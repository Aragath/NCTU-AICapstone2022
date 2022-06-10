import gym
import numpy as np
from gym import spaces
from math import floor
from kaggle_environments import make
from kaggle_environments.envs.kore_fleets.helpers import ShipyardAction, Board, Direction, Fleet
from kaggle_environments.helpers import Point
# from helpers import ShipyardAction, Board, Direction, Fleet
from typing import Union, Tuple, Dict
from reward_utils import get_board_value
from config import *

class FlightPlanHelper:
    # possible tokens in flight plan:
    #   0~9: keeps current direction for n steps
    #   N, E, S, W: directions
    #   C: convert to shipyard

    # convert plan from gym format (array)
    # to that of kore (string)

    BOARD_SIZE = 21 # GAME_CONFIG["size"]

    def get_shortest_fp(self, src: tuple, dst: tuple, max_len: int, create_shipyard: bool=False) -> str:
        # generate plan with shortest path to go from src to dst
        # fleet going out of the board comes back from the other end
        dx = dst[0] - src[0]
        if dx < 0: # dst at west
            dx = abs(dx)
            if dx < self.BOARD_SIZE / 2:
                # go west
                dx = -dx
            else:
                # go east
                dx = self.BOARD_SIZE - dx
        else: # dst at east
            dx = abs(dx)
            if dx < self.BOARD_SIZE / 2:
                # go east
                dx = dx
            else:
                # go west
                dx = -(self.BOARD_SIZE - dx)

        dy = dst[1] - src[1]
        if dy < 0: # dst at south
            dy = abs(dy)
            if dy < self.BOARD_SIZE / 2:
                # go south
                dy = -dy
            else:
                # go north
                dy = self.BOARD_SIZE - dy
        else: # dst at east
            dy = abs(dy)
            if dy < self.BOARD_SIZE / 2:
                # go north
                dy = dy
            else:
                # go south
                dy = -(self.BOARD_SIZE - dy)
        
        if dx > 0:
            fp_x = "E"
        elif dx < 0:
            fp_x = "W"
        else:
            fp_x = ""
        if abs(dx) >= 2:
            fp_x += str(abs(dx) - 1)

        if dy > 0:
            fp_y = "N"
        elif dy < 0:
            fp_y = "S"
        else:
            fp_y = ""
        if abs(dy) >= 2:
            fp_y += str(abs(dy) - 1)

        # take the axis with longer distance
        # to reduce affect of plan truncate
        if abs(dx) > abs(dy):
            fp = fp_x + fp_y
        else:
            fp = fp_y + fp_x

        if fp and create_shipyard:
            fp += "C"
        
        fp = self._truncate(fp, max_len)
        assert (not fp) or (fp[0] in "NESW")
        return fp

    @staticmethod
    def _truncate(plan: str, max_len: int):
        if not plan or len(plan) <= max_len:
            return plan
        elif plan[-1] == "C":
            return plan[:max_len-1] + "C"
        else:
            return plan[:max_len]

    @staticmethod
    def min_ship_cnt_for_fp(fp_len) -> int:
        # math.floor(2 * math.log(ship_count)) + 1
        return np.exp(np.ceil(fp_len-1) / 2)


class KoreGymEnv(gym.Env):
    """An openAI-gym env wrapper for kaggle's kore environment. Can be used with stable-baselines3.

    There are three fundamental components to this class which you would want to customize for your own agents:
        The action space is defined by `action_space` and `gym_to_kore_action()`
        The state space (observations) is defined by `state_space` and `obs_as_gym_state()`
        The reward is computed with `compute_reward()`

    Note that the action and state spaces define the inputs and outputs to your model *as numpy arrays*. Use the
    functions mentioned above to translate these arrays into actual kore environment observations and actions.

    The rest is basically boilerplate and makes sure that the kaggle environment plays nicely with stable-baselines3.

    Usage:
        >>> from stable_baselines3 import PPO
        >>>
        >>> kore_env = KoreGymEnv()
        >>> model = PPO('MlpPolicy', kore_env, verbose=1)
        >>> model.learn(total_timesteps=100000)
    """

    def __init__(self, config=None, agents=None, debug=None):
        super(KoreGymEnv, self).__init__()

        if not config:
            config = GAME_CONFIG
        if not agents:
            agents = GAME_AGENTS
        if not debug:
            debug = True

        self.agents = agents
        self.env = make("kore_fleets", configuration=config, debug=debug)
        self.config = self.env.configuration
        self.trainer = None
        self.raw_obs = None
        self.previous_obs = None

        # Define the action and state space
        # Change these to match your needs. Normalization to the [-1, 1] interval is recommended. See:
        # https://araffin.github.io/slides/rlvs-tips-tricks/#/13/0/0
        # See https://www.gymlibrary.ml/content/spaces/ for more info on OpenAI-gym spaces.
        self.action_space = spaces.Box(
            low=-1,
            high=1,
            shape=ACTION_SIZE,
            dtype=DTYPE
        )

        # ob_space_size = self.config.size ** 2 * N_2D_FEATURES + N_1D_FEATURES
        self.observation_space = spaces.Box(
            low=-1,
            high=1,
            shape=OBSERVATION_SIZE,
            dtype=DTYPE
        )

        self.strict_reward = config.get('strict', False)

        # Debugging info - Enable or disable as needed
        self.reward = 0
        self.n_steps = 0
        self.n_resets = 0
        self.n_dones = 0
        self.last_action = None
        self.last_done = False

    def reset(self) -> np.ndarray:
        """Resets the trainer and returns the initial observation in state space.

        Returns:
            self.obs_as_gym_state: the current observation encoded as a state in state space
        """
        # agents = self.agents if np.random.rand() > .5 else self.agents[::-1]  # Randomize starting position
        self.trainer = self.env.train(self.agents)
        self.raw_obs = self.trainer.reset()
        self.n_resets += 1
        return self.obs_as_gym_state

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute action in the trainer and return the results.

        Args:
            action: The action in action space, i.e. the output of the stable-baselines3 agent

        Returns:
            self.obs_as_gym_state: the current observation encoded as a state in state space
            reward: The agent's reward
            done: If True, the episode is over
            info: A dictionary with additional debugging information
        """
        kore_action = self.gym_to_kore_action(action)
        self.previous_obs = self.raw_obs
        self.raw_obs, _, done, info = self.trainer.step(kore_action)  # Ignore trainer reward, which is just delta kore
        self.reward = self.compute_reward(done)

        # Debugging info
        # with open('logs/tmp.log', 'a') as log:
        #    print(kore_action.action_type, kore_action.num_ships, kore_action.flight_plan, file=log)
        #    if done:
        #        print('done', file=log)
        #    if info:
        #        print('info', file=log)
        self.n_steps += 1
        self.last_done = done
        self.last_action = kore_action
        self.n_dones += 1 if done else 0

        return self.obs_as_gym_state, self.reward, done, info

    def render(self, **kwargs):
        self.env.render(**kwargs)

    def close(self):
        pass

    @property
    def board(self):
        return Board(self.raw_obs, self.config)

    @property
    def previous_board(self):
        return Board(self.previous_obs, self.config)

    def gym_to_kore_action(self, gym_action: np.ndarray) -> Dict[str, str]:
        gym_action = np.reshape(gym_action, (self.config.size, self.config.size, ACTION_LEN))

        """

        gym_action is board_size * board_size * ACTION_LEN
        for a cell located at (x, y), 
        gym_action[x][y] instructs the agent what to do for an ally shipyard lying here (if there is)
        (Action)

            Action[0] either specifies the mission of the launched fleet 
                      or ask the yard to build ships

            - -1 ~ -0.6: do nothing
            - -0.6 ~ -0.2: build ships
            - -0.2 ~ 0.2: shipyard builder (fleet)
            - 0.2 ~ 0.6: miner (fleet)
            - 0.6 ~ 1: attacker (fleet)

            abs(gym_action[1]): #number of ships to build/launch.
            gym_action[2]: the target to go (x axis)
            gym_action[3]: the target to go (y axis)

        Args:
            gym_action: The action produces by our stable-baselines3 agent.

        Returns:
            The corresponding kore environment actions or None if the agent wants to wait.

        """
        
        board = self.board
        me = board.current_player

        for point, cell in board.cells.items():
            shipyard = cell.shipyard
            
            if not shipyard or shipyard.player_id != me.id: 
                continue

            dst_x = gym_action[point.x][point.y][2] # in (-1, 1)
            dst_y = gym_action[point.x][point.y][3] # in (-1, 1)

            dst_x = int((dst_x + 1) * 10.49) # interger in [0, 20]
            dst_y = int((dst_y + 1) * 10.49) # interger in [0, 20]

            dst = (dst_x, dst_y)
            src = (shipyard.position.x, shipyard.position.y)
            
            number_of_ships = int(
                clip_normalize(
                    x=abs(gym_action[point.x][point.y][1]),
                    low_in=-1,
                    high_in=1,
                    low_out=1,
                    high_out=MAX_ACTION_FLEET_SIZE
                )
            )

            action = None
            mission = gym_action[point.x][point.y][0]
            if mission < -0.6: # do nothing
                pass

            elif mission >= -0.6 and mission < -0.2: # build ships
                # Limit the number of ships to the amount that is actually present in the shipyard
                max_spawn = shipyard.max_spawn
                max_purchasable = floor(me.kore / self.config["spawnCost"])
                number_of_ships = min(number_of_ships, max_spawn, max_purchasable)

                if number_of_ships:
                    action = ShipyardAction.spawn_ships(number_ships=number_of_ships)

            elif mission >= -0.2 and mission < 0.2: # shipyard builder
                ship_cnt = shipyard.ship_count
                number_of_ships = min(number_of_ships, ship_cnt)
                
                if number_of_ships >= 50: # the cost of building shipyard
                    max_len = Fleet.max_flight_plan_len_for_ship_count(number_of_ships)
                    flight_plan = FlightPlanHelper().get_shortest_fp(src=src, dst=dst, max_len=max_len, create_shipyard=True)

                    if flight_plan:
                        action = ShipyardAction.launch_fleet_with_flight_plan(number_ships=number_of_ships, flight_plan=flight_plan)

            elif mission >= 0.2 and mission < 0.6: # miner
                ship_cnt = shipyard.ship_count
                number_of_ships = min(number_of_ships, ship_cnt)

                flight_plan = FlightPlanHelper().get_shortest_fp(src=src, dst=dst, max_len=1e10) + FlightPlanHelper().get_shortest_fp(src=dst, dst=src, max_len=1e10)
                min_ship = FlightPlanHelper().min_ship_cnt_for_fp(len(flight_plan))

                if flight_plan and number_of_ships >= min_ship:
                    action = ShipyardAction.launch_fleet_with_flight_plan(number_ships=number_of_ships, flight_plan=flight_plan)

            else: # attacker
                ship_cnt = shipyard.ship_count
                number_of_ships = min(number_of_ships, ship_cnt)
                
                if number_of_ships:
                    max_len = Fleet.max_flight_plan_len_for_ship_count(number_of_ships)
                    flight_plan = FlightPlanHelper().get_shortest_fp(src=src, dst=dst, max_len=max_len)
                    
                    if flight_plan:
                        action = ShipyardAction.launch_fleet_with_flight_plan(number_ships=number_of_ships, flight_plan=flight_plan)

            shipyard.next_action = action

        return me.next_actions

    @property
    def obs_as_gym_state(self) -> np.ndarray:
        """
        Return the current observation encoded as a state in state space.

        #################### 2D features ####################
        ######### 21x21x4 (size x size x n_features) ########
        Cell:
            feat 0. #kore in the cell

        Shipyard:
            feat 1. #ships (*= -1 if the yard belongs to enemy)
            feat 2. max spawn

        Fleet:
            feat 3. #ships of mine
            feat 4. #ships of the opponent

        #################### 1D features ####################
        ################### N_1D_FEATURES ###################
        General:
            feat 6: #plays so far
            feat 7: #kore I have
            feat 8: #kore the opponent has

        """

        state_2D = np.zeros(shape=(self.config.size, self.config.size, N_2D_FEATURES))
        board = self.board
        my_id = board.current_player_id

        for point, cell in board.cells.items():
            # feat 0: #kore in the cell
            state_2D[point.y, point.x, 0] = cell.kore

            shipyard = cell.shipyard
            if shipyard:
                # feature 1: #ships owned (shipyard)
                state_2D[point.y, point.x, 1] = (
                    shipyard.ship_count 
                    if shipyard.player_id == my_id 
                    else -shipyard.ship_count
                )
                
                # feature 2: max spawn (shipyard)
                state_2D[point.y, point.x, 2] = shipyard.max_spawn

            fleet = cell.fleet
            if fleet:
                # feat 3. #ships of mine
                if fleet.player_id == my_id:
                    state_2D[point.y, point.x, 3] = fleet.ship_count
                # feat 4. #ships of the opponent
                else: 
                    state_2D[point.y, point.x, 4] = fleet.ship_count

        # express flight plan as future boards
        next_board = self.board.next()
        for i in range(10):
            for point, cell in next_board.cells.items():
                fleet = cell.fleet
                if fleet:
                    # feat 3. #ships of mine
                    if fleet.player_id == my_id:
                        state_2D[point.y, point.x, 3] += fleet.ship_count * 0.9**(i+1)
                    # feat 4. #ships of the opponent
                    else: 
                        state_2D[point.y, point.x, 4] += fleet.ship_count * 0.9**(i+1)
                
            next_board = next_board.next()

        # For better performance, bound all features in the range [-1, 1]
        # and as close to a normal distribution as possible

        # feat 0: Logarithmic scale, kore in range [0, MAX_OBSERVABLE_KORE]
        state_2D[:, :, 0] = clip_normalize(
            x=np.log2(state_2D[:, :, 0] + 1),
            low_in=0,
            high_in=np.log2(MAX_OBSERVABLE_KORE)
        )

        # feat 1: Ships in range [-MAX_OBSERVABLE_SHIPS, MAX_OBSERVABLE_SHIPS]
        state_2D[:, :, 1] = clip_normalize(
            x=state_2D[:, :, 1],
            low_in=-MAX_OBSERVABLE_SHIPS,
            high_in=MAX_OBSERVABLE_SHIPS,
        )

        # feat 2: spawn maximum in range [1, 10]
        state_2D[:, :, 2] = clip_normalize(
            x=state_2D[:, :, 2],
            low_in=MIN_SPAWN_LIMIT,
            high_in=MAX_SPAWN_LIMIT,
        )

        # feat 3: #ships of mine [0, MAX_OBSERVABLE_SHIPS]
        state_2D[:, :, 3] = clip_normalize(
            x=state_2D[:, :, 3],
            low_in=0,
            high_in=MAX_OBSERVABLE_SHIPS*2,
        )

        # feat 4: #ships of the opponent [0, MAX_OBSERVABLE_SHIPS]
        state_2D[:, :, 4] = clip_normalize(
            x=state_2D[:, :, 4],
            low_in=0,
            high_in=MAX_OBSERVABLE_SHIPS*2,
        )

        # Flatten the input (recommended by stable_baselines3.common.env_checker.check_env)
        output_state = state_2D.flatten()

        # 1D Features
        player = board.current_player
        opponent = board.opponents[0]
        progress = clip_normalize(board.step, low_in=0, high_in=GAME_CONFIG['episodeSteps'])
        my_kore = clip_normalize(np.log2(player.kore+1), low_in=0, high_in=np.log2(MAX_KORE_IN_RESERVE))
        opponent_kore = clip_normalize(np.log2(opponent.kore+1), low_in=0, high_in=np.log2(MAX_KORE_IN_RESERVE))

        return np.append(output_state, [progress, my_kore, opponent_kore])

    def compute_reward(self, done: bool, strict=False) -> float:
        """Compute the agent reward. Welcome to the fine art of RL.

         We'll compute the reward as the current board value and a final bonus if the episode is over. If the player
          wins the episode, we'll add a final bonus that increases with shorter time-to-victory.
        If the player loses, we'll subtract that bonus.

        Args:
            done: True if the episode is over
            strict: If True, count only wins/loses (Useful for evaluating a trained agent)

        Returns:
            The agent's reward
        """
        board = self.board
        previous_board = self.previous_board

        if strict:
            if done:
                # Who won?
                # Ugly but 99% sure correct, see https://www.kaggle.com/competitions/kore-2022/discussion/324150#1789804
                agent_reward = self.raw_obs.players[0][0]
                opponent_reward = self.raw_obs.players[1][0]
                return int(agent_reward > opponent_reward)
            else:
                return 0
        else:
            if done:
                # Who won?
                agent_reward = self.raw_obs.players[0][0]
                opponent_reward = self.raw_obs.players[1][0]
                if agent_reward is None or opponent_reward is None:
                    we_won = -1
                else:
                    we_won = 1 if agent_reward > opponent_reward else -1
                win_reward = we_won * (WIN_REWARD + 5 * (GAME_CONFIG['episodeSteps'] - board.step))
            else:
                win_reward = 0

            return get_board_value(board) - get_board_value(previous_board) + win_reward


def clip_normalize(x: Union[np.ndarray, float],
                   low_in: float,
                   high_in: float,
                   low_out=-1.,
                   high_out=1.) -> Union[np.ndarray, float]:
    """Clip values in x to the interval [low_in, high_in] and then MinMax-normalize to [low_out, high_out].

    Args:
        x: The array of float to clip and normalize
        low_in: The lowest possible value in x
        high_in: The highest possible value in x
        low_out: The lowest possible value in the output
        high_out: The highest possible value in the output

    Returns:
        The clipped and normalized version of x

    Raises:
        AssertionError if the limits are not consistent

    Examples:
        >>> clip_normalize(50, low_in=0, high_in=100)
        0.0

        >>> clip_normalize(np.array([-1, .5, 99]), low_in=-1, high_in=1, low_out=0, high_out=2)
        array([0., 1.5, 2.])
    """
    assert high_in > low_in and high_out > low_out, "Wrong limits"

    # Clip outliers
    try:
        x[x > high_in] = high_in
        x[x < low_in] = low_in
    except TypeError:
        x = high_in if x > high_in else x
        x = low_in if x < low_in else x

    # y = ax + b
    a = (high_out - low_out) / (high_in - low_in)
    b = high_out - high_in * a

    return a * x + b