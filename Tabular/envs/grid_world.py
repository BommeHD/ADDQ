import numpy as np
from typing import Union, Tuple, Dict, List
from copy import deepcopy

from envs.env import Env
from utils import check_for_allowed_dist, sample_from_dist

class GridWorld(Env):
    def __init__(self,
                 grid_size: Tuple[int,int] = (4,4), # Default grid size is 4x4
                 state_type_loc: Dict[str,Tuple[List[Tuple[int,int]],bool]] = None, # The dictionary mapping state names to their locations and if they are terminal
                 rewards: Dict[str,Union[int,float,List[Union[str,Dict]]]] = None, # Rewards for various state types
                 hovering = True, # Whether the player is allowed to choose to bump into the wall and by doing so hover in place
                 windy: bool = False, # Whether there is wind affecting movement
                 wind_prob: float = 0.25, # Probability of wind occurring
                 wind_dir: str = "down", # Direction of the wind (up, right, down, left)
                 slippery: bool = False, # Whether the grid is slippery
                 slip_prob: float = 0.1, # Probability of slipping
                 random_actions: bool = False, # Whether random actions occur
                 random_prob: float = 0.1, # Probability of taking a random action
                 random_vec: List[float] = None, # Probability vector for random actions
                 rng_seed: int = 42, # Random number generator seed
                 checks: str = "all_checks", # Should all checks, only initial checks, or no checks be performed
                 ) -> None:
        
        """
        Initializes the Gridworld environment with the provided parameters. The agent moves in a grid. In each
        step it can move up, right, down, or left, but without exiting the perimeter of the grid. If hovering is off
        the agent can not bump into the wall, meaning near a wall the allowed action space is limited. It starts from
        a specified state and when it reaches a goal state or a losing state the game terminates and begins again
        from the starting state. There may be wind pushing the agent in a direction instead of the intended one,
        the grid may be frozen and slippery, in which case the agent may accidentally do an adjacent action instead
        of the intended one, or there may be a probability of choosing a random action instead of the intended one.
        In the case of windy or slippery and hovering deactivated, the agent may still hover in a place if the wind or 
        the slippery action would push it over the edge of the grid. In the case of random action activated but hovering 
        deactivated the environment chooses a random action based off the passed random vector, but refactored.

        Parameters:
        - grid_size (tuple): Size of the grid, given as (rows, columns). Default is (4, 4).
        - state_type_loc (dict): A dictionary mapping state types to locations and terminal status. 
          Each entry is a tuple where the first element is a list of coordinates for that state type, 
          and the second element is a boolean indicating whether the state is terminal. Needs to contain
          locations of goal and start. Goal must be terminal, while start can not be terminal.
          Commonly used state types are:
          - goal: Highest reward, we want to find this state. Must be specified and is terminal.
          - start: Where the game starts. Must be specified.
          - hole: Hole in the grid with high negative reward. Typically terminal.
          - false_goal: Low positive reward, seems like a goal but is not. Typically terminal.
          - stoch_goal: Stochastic goal state with high reward. Alternative to goal. Typically terminal.
          - stoch: Stochastic region.
        - rewards (dict): A dictionary mapping state types to rewards. The special key "default" is used for 
          all states not specified in the dictionary. Some states may have stochastic rewards represented 
          by a distribution (e.g., normal). In this case a list containing the distribution name and a
          dictionary of keyword arguments compatible with the numpy random generator need to be passed.
        - hovering (bool): Wheter the player is allowed to bump into the wall and thus hover in the same place.
        - windy (bool): Whether wind is applied to the environment. Default is False.
        - wind_prob (float): The probability that wind will affect the environment in each step. Default is 0.25.
        - wind_dir (str): Direction of the wind as one of the following: "up", "right", "down", "left". Default is "down"
        - slippery (bool): Whether the environment is slippery, causing random movement adjacent to action. Default is False.
        - slip_prob (float): Probability that a random slip occurs in a slippery environment. Default is 0.1.
        - random_actions (bool): Whether random actions can be taken instead of the chosen action. Default is False.
        - random_prob (float): Probability of taking a random action. Default is 0.1.
        - random_vec (list): A list of probabilities for each action when taking random actions. Default is evenly distributed.
        - rng_seed (int): Seed for the random number generator. Default is 42.
        - checks (str): Checking mode. "all_checks", "only_initial_checks", and "no_checks" is available.
        """

        # Default arguments if arguments are None
        if state_type_loc is None:
            state_type_loc = { 
                     "goal": ([(3,3)],True), 
                     "start": ([(0,0)],False), 
                     "hole": ([(2,2),(2,1)],True),
                     "stoch": ([(1,1)],False)}
        if rewards is None:
            rewards = {
                     "goal": 8.5,
                     "hole": -10,
                     "stoch": ["normal",{"loc":0, "scale":1}],
                     "default": -1}
        if random_vec is None:
            random_vec = [1/4,1/4,1/4,1/4]

        # Standard initialization of arguments from the input
        self.grid_size = grid_size 
        self.state_type_loc = state_type_loc # Not modified inside the class
        self.rewards = rewards # Not modified inside the class
        self.hovering = hovering
        self.windy = windy
        self.wind_prob = wind_prob
        self.slippery = slippery
        self.slip_prob = slip_prob
        self.rng_seed = rng_seed
        self.wind_dir = wind_dir
        self.random_actions = random_actions
        self.random_prob = random_prob   
        self.random_vec = random_vec # Gets modified inside the class
        if isinstance(checks,str):
            if not (checks == "all_checks" or checks == "no_checks" or checks == "only_initial_checks"):
                raise ValueError("Checks needs to be either all_checks, no_checks, or only_initial_checks")
        else:
            raise TypeError("Flag for type and value checking must be a string!")
        self.checks = checks   
        self.init_check1_done = False
        self.init_check2_done = False
        self.init_check3_done = False

        # Basic initialization for input checks
        self.rng = np.random.default_rng(seed = self.rng_seed) # Random number generator for stochastic draws

        # Checking input constraints
        if self.checks != "no_checks":
            self.inputcheck()

        # Advanced initializations
        self.num_states = grid_size[0]*grid_size[1] # Total number of states (grid size)
        self.num_actions = 4 # Number of actions (up, down, left, right)
        self.wind_dir_num = self.action_words_to_nums(self.wind_dir) # Convert wind direction to numerical representation
        self.reset_states_num = [
            self.coord_to_state(coord) # Convert coordinates of reset states to state numbers
            for key in self.state_type_loc.keys()
            for coord in self.state_type_loc[key][0]
            if self.state_type_loc[key][1] # Only include terminal states
        ]
        self.start_state_num = self.coord_to_state(self.state_type_loc["start"][0][0]) # Convert start state coordinates to state number
        self.state_action_rewards_dict = {} # Dictionary for state rewards
        for act in [0,1,2,3]:
            for key in self.rewards.keys():
                if key == "default":
                    pass
                else:
                    for coord in self.state_type_loc[key][0]:
                        self.state_action_rewards_dict[(self.coord_to_state(coord),act)] = self.rewards[key] # Set reward for each state
        self.default_reward = self.rewards["default"] # Default reward for non-special states
        self.allowed_actions = {}
        for state in range(self.num_states):
            if hovering:
                if state in self.reset_states_num:
                    self.allowed_actions[state] = [0]
                else:
                    self.allowed_actions[state] = [0,1,2,3]
            else:
                if state == 0:
                    self.allowed_actions[state] = [1,2]
                elif state in self.reset_states_num:
                    self.allowed_actions[state] = [0]
                elif 0 < state < self.grid_size[1] - 1:
                    self.allowed_actions[state] = [1,2,3]
                elif state == self.grid_size[1] - 1:
                    self.allowed_actions[state] = [2,3]
                elif state % self.grid_size[1] == 0 and state != (self.grid_size[1] * (self.grid_size[0] - 1)):
                    self.allowed_actions[state] = [0,1,2]
                elif state == (self.grid_size[1] * (self.grid_size[0] - 1)):
                    self.allowed_actions[state] = [0,1]
                elif (state + 1) % self.grid_size[1] == 0 and (state + 1) != (self.grid_size[1] * self.grid_size[0]):
                    self.allowed_actions[state] = [0,2,3]
                elif state == self.num_states - 1:
                    self.allowed_actions[state] = [0,3]
                elif self.num_states - self.grid_size[1] < state < self.num_states - 1:
                    self.allowed_actions[state] = [0,1,3]
                else:
                    self.allowed_actions[state] = [0,1,2,3]
        game_probs = {(next_state,state,action):0 for next_state in range(self.num_states) for state in range(self.num_states) for action in self.allowed_actions[state] if not (state in self.reset_states_num)}
        for state in range(self.num_states):
            if state in self.reset_states_num:
                pass
            else:
                for action in self.allowed_actions[state]:
                    next_state = self.get_next_state_det(state,action)
                    if not (self.windy or self.slippery or self.random_actions):
                        game_probs[(next_state,state,action)] = 1
                    elif self.windy:
                        game_probs[(next_state,state,action)] = 1 - self.wind_prob
                        if self.wind_dir_num in self.allowed_actions[state]:
                            next_state_windy = self.get_next_state_det(state,self.wind_dir_num)
                        else:
                            next_state_windy = state
                        game_probs[(next_state_windy,state,action)] += self.wind_prob
                    elif self.slippery:
                        game_probs[(next_state,state,action)] = 1 - 2/3 * self.slip_prob
                        next_state_slip1 = self.get_next_state_det(state , (action + 1) % self.num_actions)
                        next_state_slip2 = self.get_next_state_det(state, (action - 1) % self.num_actions)
                        game_probs[(next_state_slip1,state,action)] += 1/3 * self.slip_prob
                        game_probs[(next_state_slip2,state,action)] += 1/3 * self.slip_prob
                    elif self.random_actions:
                        game_probs[(next_state,state,action)] = 1 - self.random_prob
                        allowed_indices_prob = [self.random_vec[act] for act in self.allowed_actions[state]]
                        factor = sum(allowed_indices_prob)
                        probs = [val * self.random_prob/factor for val in allowed_indices_prob]
                        for i,a in enumerate(self.allowed_actions[state]):
                            next_state_random_act = self.get_next_state_det(state,a)
                            game_probs[(next_state_random_act,state,action)] += probs[i]
        self.game_probabilities = {key: val for key,val in game_probs.items() if val != 0}
    
    def __str__(self):
        return "GridWorld"

    def get_next_state(self, state: int, action: int) -> Tuple[int,bool]:

        """
        Returns the next state after taking an action from a given state.
        Takes into account wind, slippery surfaces, and random actions.

        Parameters:
        - state (int): The current state, represented as a state number.
        - action (int): The action taken, where 0 = up, 1 = right, 2 = down, 3 = left.

        Returns:
        - int: The state number resulting from the action taken.
        - bool: Signals, if the game is restarted due to being in a terminal state
        """

        # Terminal and restarting?
        t = False

        # Check if state is valid integer and action is allowed
        if self.checks != "no_checks" and not self.init_check1_done:
            if isinstance(state,int) and isinstance(action,int):
                if not ((state in range(self.num_states)) and (action in range(self.num_actions))):
                    raise ValueError("State and action pair outside of state action space!")
                else:
                    if not (action in self.allowed_actions[state]):
                        raise ValueError("Action is not allowed in this state!")
            else:
                raise TypeError("State and action need to be integers!")
            if self.checks == "only_initial_check":
                self.init_check1_done = True
        
        # Reset needed? If so, next state is always start
        if state in self.reset_states_num:
            next_state = self.start_state_num
            t = True
            return next_state, t
        
        # Apply wind effect, if applicable
        if self.windy:
            if self.rng.uniform(0,1) < self.wind_prob:
                if self.wind_dir_num in self.allowed_actions[state]:
                    action = self.wind_dir_num
                else:
                    next_state = state
                    return next_state, t

        # Apply slippery effect, if applicable
        if self.slippery:
            if self.rng.uniform(0,1) < self.slip_prob:
                neighbor = self.rng.choice([-1, 0, 1], p=[1/3, 1/3, 1/3])
                if (action + neighbor) % self.num_actions in self.allowed_actions[state]:
                    action = (action + neighbor) % self.num_actions
                else:
                    next_state = state
                    return next_state, t

        # Apply random action, if applicable
        if self.random_actions:
            if self.rng.uniform(0,1) < self.random_prob:
                allowed_indices_prob = [self.random_vec[act] for act in self.allowed_actions[state]]
                factor = sum(allowed_indices_prob)
                probs = [val/factor for val in allowed_indices_prob]
                action = self.rng.choice(self.allowed_actions[state], p=probs)

        # Determine next state based on action and return it
        if action == 0: # Action "up"
            if not (state - self.grid_size[1] < 0):
                return state - self.grid_size[1], t
            else:
                return state, t  
        elif action ==1: # Action "right"
            if not (((state + 1) % self.grid_size[1]) == 0):
                return state + 1, t
            else:
                return state, t
        elif action ==2: # Action "down"
            if not (state + self.grid_size[1]) >= self.num_states:
                return state + self.grid_size[1], t
            else:
                return state, t
        elif action ==3: # Action "left"
            if not (state % self.grid_size[1] == 0):
                return state - 1, t
            else:
                return state, t
        else:
            raise ValueError("Something went wrong and the action was somehow chosen outside of the action space while sampling the next state!")
    
    def get_reward(self,state: int,action: int) -> Union[int,float]:

        """
        Returns the reward for a given state-action pair.

        Parameters:
        - state (int): The current state, represented as a state number.
        - action (int): The action taken, where 0 = up, 1 = right, 2 = down, 3 = left.

        Returns:
        - float: The reward for the state-action pair.
        """

        # Check if state is valid integer and action is allowed
        if self.checks != "no_checks" and not self.init_check2_done:
            if isinstance(state,int) and isinstance(action,int):
                if not ((state in range(self.num_states)) and (action in range(self.num_actions))):
                    raise ValueError("State and action pair outside of state action space!")
                else:
                    if not (action in self.allowed_actions[state]):
                        raise ValueError("Action is not allowed in this state!")
            else:
                raise TypeError("State and action need to be integers!")
            if self.checks == "only_initial_check":
                self.init_check2_done = True
        
        # Assign the reward for the state, or sample from the distribution specified in case of stochastic rewards
        if (state,action) in self.state_action_rewards_dict.keys():
            if isinstance(self.state_action_rewards_dict[(state,action)], (int,float)):
                return self.state_action_rewards_dict[(state,action)]
            else:
                return float(sample_from_dist(self.rng,self.state_action_rewards_dict[(state,action)][0],1,**self.state_action_rewards_dict[(state,action)][1])[0])
        else:
            if isinstance(self.default_reward,(int,float)):
                return self.default_reward
            else:
                return float(sample_from_dist(self.rng,self.default_reward[0],1,**self.default_reward[1])[0])
        
    def get_next_state_and_reward(self, state: int,action: int) -> Tuple[int,bool,Union[int,float]]:

        """
        Returns the next state and reward for a given state-action pair.

        Parameters:
        - state (int): The current state, represented as a state number.
        - action (int): The action taken, where 0 = up, 1 = right, 2 = down, 3 = left.

        Returns:
        - int: The state number resulting from the action taken.
        - bool: Signals, if the game is restarted due to being in a terminal state.
        - float: The reward for the state-action pair.
        """

        next_state, t = self.get_next_state(state,action)
        reward = self.get_reward(state,action)
        return next_state, t, reward
    
    def get_next_state_det(self, state: int, action: int) -> Tuple[int,bool]:

        """
        Returns the next state after taking an action from a given state.
        Does not take into account wind, slippery surfaces, and random actions.

        Parameters:
        - state (int): The current state, represented as a state number.
        - action (int): The action taken, where 0 = up, 1 = right, 2 = down, 3 = left.

        Returns:
        - int: The state number resulting from the action taken.
        """
        
        # Reset needed? If so, next state is always start
        if state in self.reset_states_num:
            next_state = self.start_state_num
            return next_state

        # Determine next state based on action and return it
        if action == 0: # Action "up"
            if not (state - self.grid_size[1] < 0):
                return state - self.grid_size[1]
            else:
                return state 
        elif action ==1: # Action "right"
            if not (((state + 1) % self.grid_size[1]) == 0):
                return state + 1
            else:
                return state
        elif action ==2: # Action "down"
            if not (state + self.grid_size[1]) >= self.num_states:
                return state + self.grid_size[1]
            else:
                return state
        elif action ==3: # Action "left"
            if not (state % self.grid_size[1] == 0):
                return state - 1
            else:
                return state
        
    def mean_rewards_to_state_action(self, mean_rewards: Dict = None, mc_runs: int = 100000) -> Dict[Tuple[int,int],Union[int,float]]:

        """
        Returns a dictionary containing mean rewards. Can be based on a dictionary shaped like the rewards dictionary containing 
        the means for the state action pairs or alternatively uses MC runs to determine a proxy for the mean.

        Parameters:
        - mean_rewards (Dict): Eiter an empty dictionary if mc runs should be performed to get the values or a dictionary
          of the same shape as rewards
        - mc_runs (int): The number of MC-Runs to be performed.

        Returns:
        - dict: The dictionary containing the mean reward for each state action pair.
        """

        if mean_rewards is None:
            mean_rewards = {}

        # Check if state is valid integer and action is allowed
        if self.checks != "no_checks" and not self.init_check3_done:
            if isinstance(mc_runs,int):
                if mc_runs <= 0:
                    raise ValueError("Number of Monte Carlo runs needs to be bigger than 0!")
            else:
                raise TypeError("Number of Monte Carlo runs needs to be numerical!")
            if isinstance(mean_rewards,dict):
                if mean_rewards == {}:
                    pass
                elif set(mean_rewards.keys()) == set(self.rewards.keys()) and len(mean_rewards.keys()) == len(self.rewards.keys()):
                    for key in mean_rewards.keys():
                        if not isinstance(mean_rewards[key],(int,float)):
                            raise ValueError("Mean rewards need to take numerical values!")
                else:
                    raise ValueError("The mean rewards dictionary needs to either be empty or have the same shape as the rewards dictionary!")
            else:
                raise TypeError("Mean rewards need to be passed in a dictionary!")
            if self.checks == "only_initial_checks":
                self.init_check3_done = True
        
        # Fill dictionary with MC-Runs if it is empty
        if mean_rewards == {}:
            m_rewards = deepcopy(self.rewards)
            for key in m_rewards.keys():
                if not isinstance(m_rewards[key],(int,float)):
                    mc_runs_values = []
                    for _ in range(mc_runs):
                        reward = float(sample_from_dist(self.rng,m_rewards[key][0],1,**m_rewards[key][1])[0])
                        mc_runs_values.append(reward)
                    mc_estimator = sum(mc_runs_values) / mc_runs
                    m_rewards[key] = mc_estimator
        
        # Initialization if dictionary is non-empty
        else:
            m_rewards = deepcopy(mean_rewards)

        # Fill the state action rewards_dict to return
        s_a_rewards_dict = {(state,action): m_rewards["default"] for state in range(self.num_states) for action in self.allowed_actions[state]}
        for key in m_rewards.keys():
            if key == "default":
                pass
            else:
                for coord in self.state_type_loc[key][0]:
                    for act in self.allowed_actions[self.coord_to_state(coord)]:
                        s_a_rewards_dict[(self.coord_to_state(coord),act)] = m_rewards[key] # Set mean reward for each state that is not considered a default state
        
        return s_a_rewards_dict

    def coord_to_state(self, coordinate_tuple: Tuple[int,int]) -> int:

        """
        Converts a coordinate tuple (row, col) to a state number.

        Parameters:
        - coordinate_tuple (tuple): A tuple representing the (row, col) position in the grid.

        Returns:
        - int: The state number corresponding to the given coordinates.
        """

        return (coordinate_tuple[0] * self.grid_size[1]) + coordinate_tuple[1]    
    
    def state_to_coord(self, state: int) -> Tuple[int,int]:

        """
        Converts a state number to a coordinate tuple (row, col).

        Parameters:
        - state (int): The state number to convert.

        Returns:
        - tuple: The (row, col) coordinates corresponding to the given state.
        """

        return ( int(np.floor(state / self.grid_size[1])) , state % self.grid_size[1]) 
    
    def action_words_to_nums(self,action: str) -> int:

        """
        Converts action words ("up", "right", "down", "left") to action numbers (0, 1, 2, 3).

        Parameters:
        - action (str): The action word, one of "up", "right", "down", or "left".

        Returns:
        - int: The corresponding action number (0 = up, 1 = right, 2 = down, 3 = left).
        """

        if action == "up":
            return 0
        elif action == "right":
            return 1
        elif action == "down":
            return 2
        elif action == "left":
            return 3
        else:
            raise ValueError("Only up, right, down, or left are valid as actions or directions!")

    def inputcheck(self) -> int:

        """
        Validates the input parameters to ensure they follow the expected formats and constraints.

        Raises:
        - ValueError: If any of the input parameters are invalid.
        - TypeError: If any of the input types are invalid.
        """

        # Grid size is tuple of length 2 containing positive integers
        if isinstance(self.grid_size,tuple):
            if len(self.grid_size) != 2:
                raise ValueError("Grid size must be tuple of length two!")
            else: 
                for value in self.grid_size:
                    if not isinstance(value,int):
                        raise ValueError("Grid size components must be integers!")
                    else:
                        if value <= 0:
                            raise ValueError("Grid size components must be strictly positive")
        else:
            raise TypeError("Grid size needs to be a tuple!") 
        
        # Hovering needs to be boolean
        if not isinstance(self.hovering, bool):
            raise TypeError("Variable hovering needs to be a boolean value!")
        
        # Windy, slippery, and random are boolean values
        if not isinstance(self.windy,bool):
            raise TypeError("Variable windy needs to be a boolean value!")
        if not isinstance(self.slippery,bool):
            raise TypeError("Variable slippery needs to be a boolean value!")
        if not isinstance(self.random_actions,bool):
            raise TypeError("Variable random actions needs to be a boolean value!")
        
        # Only one of windy, slippery, and random
        if self.windy and self.slippery or self.windy and self.random_actions or self.slippery and self.random_actions:
            raise ValueError("Only one of the modes windy, slippery, and random actions can not be on at the same time!")
        
        # Wind_prob, slip_prob, and random_prob are values between 0 and 1
        if isinstance(self.wind_prob, (int, float)):
            if not (0 <= self.wind_prob <= 1):
                raise ValueError("The wind probability needs to be between 0 and 1!")
        else:
            raise TypeError("Variable wind_prob needs to be numeric")  
        if isinstance(self.slip_prob, (int, float)):
            if not (0 <= self.slip_prob <= 1):
                raise ValueError("The wind probability needs to be between 0 and 1!")
        else:
            raise TypeError("Variable wind_prob needs to be numeric")
        if isinstance(self.random_prob, (int, float)):
            if not (0 <= self.random_prob <= 1):
                raise ValueError("The wind probability needs to be between 0 and 1!")
        else:
            raise TypeError("Variable random_prob needs to be numeric")  
        
        # Seed is in valid range:
        if isinstance(self.rng_seed, int):
            if not (0 <= self.rng_seed < 2**32):
                raise ValueError("The provided seed is not in the range of acceptable integer seeds!")
        else:
            raise TypeError("The seed needs to be an integer!")
        
        # Wind direction is a valid action
        if isinstance(self.wind_dir, str):
            if not (self.wind_dir in ["up","right","left","down"]):
                raise ValueError("Wind direction is not contained in the action space!")
        else:
            raise TypeError("Wind direction must be a string!") 
        
        # Random_vec needs to be a probability vector of length 4 as a list
        if isinstance(self.random_vec, list):
            if all(isinstance(prob,(int,float)) for prob in self.random_vec):
                if not np.sum(self.random_vec) == 1:
                    raise ValueError("Probability values in random_vec need to add up to 1!")
            else:
                raise TypeError("Random_vec needs to be a list containing numerical values!")
        else:
            raise TypeError("Random_vec needs to be a list!")
        
        # State_type_loc needs to be dict containing at least goal and start and map to 2-tuples of lists of tuples of valid coordinates and boolean values representing if the state terminates
        if isinstance(self.state_type_loc,dict):
            if "goal" in self.state_type_loc.keys() and "start" in self.state_type_loc.keys():
                for key in self.state_type_loc.keys():
                    if isinstance(self.state_type_loc[key],tuple):
                        if len(self.state_type_loc[key]) == 2:
                            if isinstance(self.state_type_loc[key][0],list):
                                if key == "goal" or key == "start":
                                    if len(self.state_type_loc[key][0]) != 1:
                                        raise ValueError(f"There can only be one location for {key}!")
                                for coord_tuple in self.state_type_loc[key][0]:
                                    if isinstance(coord_tuple,tuple):
                                        if len(coord_tuple) != 2:
                                            raise ValueError(f"Locations of {key} must be tuples of length two!")
                                        else: 
                                            for value in coord_tuple:
                                                if not isinstance(value,int):
                                                    raise ValueError(f"Location components of {key} must be integers!")
                                                else:
                                                    if (coord_tuple[0] >= self.grid_size[0]) or (coord_tuple[1] >= self.grid_size[1]):
                                                        raise ValueError(f"The location {coord_tuple} for the state type {key} is out of bounds!")
                                    else:
                                        raise TypeError(f"Locations of {key} need to be tuples!")
                                if not isinstance(self.state_type_loc[key][1],bool):
                                    raise TypeError(f"The value for if the state type {key} terminates or not needs to be a boolean one!")
                            else:
                                raise TypeError("Dictionary keys of state_type_loc need to map to lists of coordinates!")
                        else:
                            raise TypeError("For each keyword, the dictionary of state types and locations needs to contain a tuple of length 2 containing a list of locations and a boolean indicating if the state is terminating or not!")
                    else:
                        raise ValueError(f"State types must map to a tuple containing a list of coordinates and if the state terminates!")
            else:
                raise ValueError("The location of the goal and the start state need to be specified!")
        else:
            raise TypeError("Variable state_type_loc needs to be a dictionary!") 
        
        # No double assignments in state type locations dictionary
        coordinates = []
        for key in self.state_type_loc.keys():
            for coord in self.state_type_loc[key][0]:
                coordinates.append(coord)
        if len(coordinates) != len(set(coordinates)):
            raise ValueError("There are coordinates used for multiple state types!")
        
        # Goal needs to be terminal and Start can not be terminal
        if not self.state_type_loc["goal"][1] or self.state_type_loc["start"][1]:
            raise ValueError("Goal must be terminal and start can not be terminal!")
        
        # Rewards needs to be a dictionary mapping all appearing state types and default to their rewards, which can be either stochastic or fixed
        if isinstance(self.rewards,dict):
            if all([key in self.state_type_loc.keys() or key == "default" for key in self.rewards.keys()]):
                if all([key in self.rewards.keys() or key == "start" for key in self.state_type_loc.keys()]):
                    if "goal" in self.rewards.keys() and "default" in self.rewards.keys():
                        for key in self.rewards.keys():
                            if isinstance(self.rewards[key],list):
                                if len(self.rewards[key]) == 2:
                                    if (isinstance(self.rewards[key][0],str) and isinstance(self.rewards[key][1],dict)):
                                        check_for_allowed_dist(self.rng,self.rewards[key][0],**self.rewards[key][1])
                                else:
                                    raise ValueError("For stochastic rewards a list containing the distribution and a dictionary of keyword arguments need to be passed!")
                            elif (isinstance(self.rewards[key],(int,float))):
                                pass
                            else:
                                raise TypeError("The rewards need to be either numeric or a list containing a sample distribution and a dictionary of keyword arguments!")
                    else:
                        raise ValueError("Rewards for goal and default value need to be provided!")
                else:
                    raise ValueError("A state type with a specified location was provided that has no specified reward!")
            else:
                raise ValueError("A reward was specified for a state type for which no location was provided!")
        else:
            raise TypeError("Variable rewards needs to be a dictionary!")
        
        return 1