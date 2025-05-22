import numpy as np
from typing import Union, Tuple, Dict, List
from copy import deepcopy

from envs.env import Env
from utils import check_for_allowed_dist, sample_from_dist

class SuttonExtended(Env):
    def __init__(self,
                 grid_size: Tuple[int, ...] = (1,2,2), # Number of states for each branch
                 num_arms: Tuple[Tuple[int, ...], ...] = ((1,),(1,10),(1,1)), # Arms connecting to the states for each branch
                 rewards: Dict[Union[str,int,Tuple[int,int]],Union[int,float,List[Union[str,Dict]],List[List[Union[int,List[Union[str,Dict]]]]]]] = None, # Dictionary containing tuple of branch and round number, as well as reward structure for arms
                 rng_seed: int = 42, # Random number generator seed
                 checks: str = "all_checks", # Should all checks, only initial checks, or no checks be performed
                 ) -> None:
        
        """
        Initializes the extended Sutton example with the provided parameters. In the starting state, the 
        agent may choose to take branches according to the length of the grid size. Each branch will play
        n rounds according to its grid size parameter. In each round the agent chooses an arm, where 
        num_arms specifies how many arms can be chosen from. The agent then receives a reward specified in
        the rewards dictionary.

        Parameters:
        - grid_size (tuple): A tuple whose length determines the number of branches, numbered from 0 to
          length of tuple - 1 and whose entries determine the number of steps per branch, numbered from 1 to the entry.
        - num_arms (tuple): A tuple containing tuples of number of arms for each step in each branch.
        - rewards (dict): A dictionary mapping state coordinates to rewards. The special key "default" is used for 
          all states not specified in the dictionary. Some states may have stochastic rewards represented 
          by a distribution (e.g., normal). In this case a list containing the distribution name and a
          dictionary of keyword arguments compatible with the numpy random generator need to be passed.
        - rng_seed (int): Seed for the random number generator. Default is 42.
        - checks (str): Checking mode. "all_checks", "only_initial_checks", and "no_checks" is available.
        """

        # Default arguments if arguments are None
        if rewards is None:
            rewards = { 
                     (1,1): [[0,5,["normal",{"loc":-0.1, "scale":5}]],[6,["normal",{"loc":-0.5, "scale":10}]],[7,8,["normal",{"loc":-1, "scale":20}]]],
                     (2,1): ["normal",{"loc":0.1, "scale":0.1}],
                     "default": 0}

        # Standard initialization of arguments from the input
        self.grid_size = grid_size
        self.num_arms = num_arms
        self.rewards = rewards # Not modified inside the class
        self.rng_seed = rng_seed
        if isinstance(checks,str):
            if not (checks == "all_checks" or checks == "no_checks" or checks == "only_initial_checks"):
                raise ValueError("Checks needs to be either all_checks, no_checks, or only_initial_checks")
        else:
            raise TypeError("Flag for type and value checking must be a string!")
        self.checks = checks   
        self.init_check1_done = False
        self.init_check2_done = False   

        # Basic initialization for input checks
        self.rng = np.random.default_rng(seed = self.rng_seed) # Random number generator for stochastic draws
        self.num_branches = len(self.grid_size)

        # Checking input constraints
        if self.checks != "no_checks":
            self.inputcheck()

        # Advanced initializations
        self.num_states = 1  # Total number of states (initial state and number of steps per branch
        for steps in self.grid_size:
            self.num_states += steps
        armslist = [value for tup in self.num_arms for value in tup]
        originactions = 0
        for branch in range(self.num_branches):
            originactions += self.num_arms[branch][0]
        armslist.append(originactions)
        self.num_actions = int(np.max(armslist)) # Number of actions (maximum of number of branches and arms)
        self.reset_states_num = [] # Store terminal states
        counter = 0
        for periods in grid_size:
            counter += periods
            self.reset_states_num.append(counter)
        self.start_state_num = 0 # Starting state is always state 0
        self.allowed_actions = {}
        actions = [i for i in range(self.num_actions)]
        for state in range(self.num_states):
            if state == 0:
                self.allowed_actions[state] = actions[:originactions]
            elif state in self.reset_states_num:
                self.allowed_actions[state] = [0]
            else:
                coord = self.state_to_coord(state)
                self.allowed_actions[state] = actions[:self.num_arms[coord[0]][coord[1]]]
        self.state_action_rewards_dict = {} # Dictionary for state action rewards
        for key in self.rewards.keys():
            if key == "default":
                pass
            elif isinstance(key,int):
                state = self.reset_states_num[key]
                self.state_action_rewards_dict[(state,0)] = self.rewards[key]
            elif key[1] == 0:
                state = 0
                action_baseline = 0
                for i in range(key[0]):
                    action_baseline += self.num_arms[i][0]
                if isinstance(self.rewards[key],list):
                    if isinstance(self.rewards[key][0],str):
                        act = action_baseline
                        self.state_action_rewards_dict[(state,act)] = self.rewards[key]
                    elif isinstance(self.rewards[key][0],(int,float)):
                        act = action_baseline
                        self.state_action_rewards_dict[(state,act)] = self.rewards[key]
                    else:
                        for act_rew in self.rewards[key]:
                            if len(act_rew) == 2:
                                self.state_action_rewards_dict[(state,action_baseline + act_rew[0])] = act_rew[1]
                            else:
                                for act in range(act_rew[0],act_rew[1]+1):
                                    self.state_action_rewards_dict[(state,action_baseline + act)] = act_rew[2]
                else:
                    act = action_baseline
                    self.state_action_rewards_dict[(state,act)] = self.rewards[key]
            elif isinstance(self.rewards[key],list):
                if isinstance(self.rewards[key][0],str):
                    for act in self.allowed_actions[self.coord_to_state(key)]:
                        self.state_action_rewards_dict[(self.coord_to_state(key),act)] = self.rewards[key]
                elif isinstance(self.rewards[key][0],(int,float)):
                    for act in self.allowed_actions[self.coord_to_state(key)]:
                        self.state_action_rewards_dict[(self.coord_to_state(key),act)] = self.rewards[key]
                else:
                    for act_rew in self.rewards[key]:
                        if len(act_rew) == 2:
                            self.state_action_rewards_dict[(self.coord_to_state(key),act_rew[0])] = act_rew[1]
                        else:
                            for act in range(act_rew[0],act_rew[1]+1):
                                self.state_action_rewards_dict[(self.coord_to_state(key),act)] = act_rew[2]
            else:
                for act in self.allowed_actions[self.coord_to_state(key)]:
                    self.state_action_rewards_dict[(self.coord_to_state(key),act)] = self.rewards[key]
        self.default_reward = self.rewards["default"] # Default reward for non-special state action pairs 
        game_probs = {(next_state,state,action):0 for next_state in range(self.num_states) for state in range(self.num_states) for action in self.allowed_actions[state] if not (state in self.reset_states_num)}
        for state in range(self.num_states):
            if state in self.reset_states_num:
                pass
            else:
                for action in self.allowed_actions[state]:
                    next_state, _ = self.get_next_state(state,action)
                    game_probs[(next_state,state,action)] = 1
        self.game_probabilities = {key: val for key,val in game_probs.items() if val != 0}

    def __str__(self):
        return "SuttonExtended"

    def get_next_state(self, state: int, action: int) -> Tuple[int,bool]:

        """
        Returns the next state after taking an action from a given state.

        Parameters:
        - state (int): The current state, represented as a state number.
        - action (int): The action taken, corresponding to the chose arm.

        Returns:
        - int: The state number resulting from the action taken.
        - bool: Signals, if the game is restarted due to being in a terminal state.
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
        elif state == 0:
            branch = 0
            counter = 1
            for _ in range(action):
                if counter >= self.num_arms[branch][0]:
                    branch += 1
                    counter = 1
                else:
                    counter += 1
            return self.coord_to_state((branch,1)), t
        else:
            return state + 1, t

    def get_reward(self,state: int,action: int) -> Union[int,float]:

        """
        Returns the reward for a given state-action pair.

        Parameters:
        - state (int): The current state, represented as a state number.
        - action (int): The action taken, corresponding to the chosen arm.

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
        - action (int): The action taken, corresponding to the chosen arm.

        Returns:
        - int: The state number resulting from the action taken.
        - bool: Signals, if the game is restarted due to being in a terminal state.
        - float: The reward for the state-action pair.
        """

        next_state, t = self.get_next_state(state,action)
        reward = self.get_reward(state,action)
        return next_state, t, reward
    
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
        if self.checks != "no_checks" and not self.init_check2_done:
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
                            if isinstance(mean_rewards[key],list):
                                for block in mean_rewards[key]:
                                    if len(block) == 2:
                                        if (isinstance(block[0],int) and isinstance(block[1],(int,float))):
                                            if not block[0] in range(max(max(self.num_arms))):
                                                raise ValueError("Chosen arm for one of the mean rewards outside of range!")
                                        else:
                                            raise ValueError("Mean rewards need to be contained in lists conforming to possible arms!")
                                    elif len(block) == 3:
                                        if (isinstance(block[0],int) and isinstance(block[1],int) and isinstance(block[2],(int,float))):
                                            if not (block[0] in range(max(max(self.num_arms))) and block[1] in range(max(max(self.num_arms))) and block[0]<block[1]):
                                                raise ValueError("Chosen arms for one of the mean rewards outside of range or range of arms wrong!")
                                        else:
                                            raise ValueError("Mean rewards need to be contained in lists conforming to possible arms!")
                                    else:
                                        raise ValueError("Mean rewards need to be contained in lists conforming to possible arms with length of 2 or three!")
                            else:
                                raise ValueError("Mean rewards need to take numerical values or lists conforming to possible arms!")
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
                    if isinstance(m_rewards[key][0],str):
                        mc_runs_values = []
                        for _ in range(mc_runs):
                            reward = float(sample_from_dist(self.rng,m_rewards[key][0],1,**m_rewards[key][1])[0])
                            mc_runs_values.append(reward)
                        mc_estimator = sum(mc_runs_values) / mc_runs
                        m_rewards[key] = mc_estimator
                    else:
                        for i, block_list in enumerate(m_rewards[key]):
                            if len(block_list) == 2:
                                mc_runs_values = []
                                for _ in range(mc_runs):
                                    reward = float(sample_from_dist(self.rng,m_rewards[key][i][1][0],1,**m_rewards[key][i][1][1])[0])
                                    mc_runs_values.append(reward)
                                mc_estimator = sum(mc_runs_values) / mc_runs
                                m_rewards[key][i][1] = mc_estimator
                            elif len(block_list) == 3:
                                mc_runs_values = []
                                for _ in range(mc_runs):
                                    reward = float(sample_from_dist(self.rng,m_rewards[key][i][2][0],1,**m_rewards[key][i][2][1])[0])
                                    mc_runs_values.append(reward)
                                mc_estimator = sum(mc_runs_values) / mc_runs
                                m_rewards[key][i][2] = mc_estimator
        
        # Initialization if dictionary is non-empty
        else:
            m_rewards = deepcopy(mean_rewards)

        # Fill the state action rewards_dict to return
        s_a_rewards_dict = {(state,action): m_rewards["default"] for state in range(self.num_states) for action in self.allowed_actions[state]}
        for key in m_rewards.keys():
            if key == "default":
                pass
            elif isinstance(key,int):
                state = self.reset_states_num[key]
                s_a_rewards_dict[(state,0)] = m_rewards[key]
            elif key[1] == 0:
                state = 0
                action_baseline = 0
                for i in range(key[0]):
                    action_baseline += self.num_arms[i][0]
                if isinstance(m_rewards[key],list):
                    if isinstance(m_rewards[key][0],str):
                        act = action_baseline
                        s_a_rewards_dict[(state,act)] = m_rewards[key]
                    elif isinstance(m_rewards[key][0],(int,float)):
                        act = action_baseline
                        s_a_rewards_dict[(state,act)] = m_rewards[key]
                    else:
                        for act_rew in m_rewards[key]:
                            if len(act_rew) == 2:
                                s_a_rewards_dict[(state,action_baseline + act_rew[0])] = act_rew[1]
                            else:
                                for act in range(act_rew[0],act_rew[1]+1):
                                    s_a_rewards_dict[(state,action_baseline + act)] = act_rew[2]
                else:
                    act = action_baseline
                    s_a_rewards_dict[(state,act)] = m_rewards[key]
            elif isinstance(m_rewards[key],list):
                if isinstance(m_rewards[key][0],str):
                    for act in self.allowed_actions[self.coord_to_state(key)]:
                       s_a_rewards_dict[(self.coord_to_state(key),act)] = m_rewards[key]
                elif isinstance(m_rewards[key][0],(int,float)):
                    for act in self.allowed_actions[self.coord_to_state(key)]:
                        s_a_rewards_dict[(self.coord_to_state(key),act)] = m_rewards[key]
                else:
                    for act_rew in m_rewards[key]:
                        if len(act_rew) == 2:
                            s_a_rewards_dict[(self.coord_to_state(key),act_rew[0])] = act_rew[1]
                        else:
                            for act in range(act_rew[0],act_rew[1]+1):
                                s_a_rewards_dict[(self.coord_to_state(key),act)] = act_rew[2]
            else:
                for act in self.allowed_actions[self.coord_to_state(key)]:
                    s_a_rewards_dict[(self.coord_to_state(key),act)] = m_rewards[key]
        
        return s_a_rewards_dict

    def state_to_coord(self, state: int) -> Tuple[int,int]:

        """
        Converts a state number to a coordinate tuple (branch,round number).

        Parameters:
        - state (int): The state number to convert.

        Returns:
        - tuple: The (branch,round number) coordinates corresponding to the given state.
        """
        
        if state == 0:
            raise ValueError("This function is not implemented for the origin!")

        coord = [0,1]

        for _ in range(state - 1):
            if coord[1] == self.grid_size[coord[0]]:
                coord[0] += 1
                coord[1] = 1
            else:
                coord[1] += 1

        return tuple(coord) 

    def coord_to_state(self, coordinate_tuple: Tuple[int,int]) -> int:

        """
        Converts a coordinate tuple (branch,round number) to a state number.

        Parameters:
        - coordinate_tuple (tuple): A tuple representing the branch we are on and the round number.

        Returns:
        - int: The state number corresponding to the given coordinates.
        """

        state = 0

        for periods in self.grid_size[:coordinate_tuple[0]]:
            state += periods

        return state + coordinate_tuple[1] 

    def inputcheck(self) -> int:

        """
        Validates the input parameters to ensure they follow the expected formats and constraints.

        Raises:
        - ValueError: If any of the input parameters are invalid.
        - TypeError: If any of the input types are invalid.
        """

        # Grid size is tuple containing positive integers
        if isinstance(self.grid_size,tuple):
            for value in self.grid_size:
                if not isinstance(value,int):
                    raise ValueError("Grid size components must be integers!")
                else:
                    if value <= 0:
                        raise ValueError("Grid size components must be strictly positive")
        else:
            raise TypeError("Grid size needs to be a tuple!") 
        
        # Num arms is a tuple of tuples containing a number of positive integers corresponding to the grid size
        if isinstance(self.num_arms, tuple):
            if len(self.num_arms) != self.num_branches:
                raise ValueError("Length of the tuple for number of arms needs to correspond to the number of branches specified by the grid size!")
            else:
                for branch_no in range(self.num_branches):
                    if isinstance(self.num_arms[branch_no],tuple):
                        if len(self.num_arms[branch_no]) != self.grid_size[branch_no]:
                            raise TypeError("The tuple length of the tuples containing the number of arms needs to match the grid size!")
                        else:
                            for value in self.num_arms[branch_no]:
                                if not isinstance(value,int):
                                    raise ValueError("Number of arms must be integers!")
                                else:
                                    if value <= 0:
                                        raise ValueError("Number of arms must always be strictly positive!")
                    elif isinstance(self.num_arms[branch_no],int):
                        if self.grid_size[branch_no] != 1:
                            raise TypeError("The tuple length of the tuples containing the number of arms needs to match the grid size!")
                        else:
                            if self.num_arms[branch_no] <= 0:
                                raise ValueError("Number of arms must always be strictly positive!")
                    else:
                        raise TypeError("Number of arms per step in a branch needs to be a tuple!")
        else:
            raise TypeError("Number of arms needs to be a tuple!")
        
        # Seed is in valid range:
        if isinstance(self.rng_seed, int):
            if not (0 <= self.rng_seed < 2**32):
                raise ValueError("The provided seed is not in the range of acceptable integer seeds!")
        else:
            raise TypeError("The seed needs to be an integer!")
        
        # Rewards is a dictionary containing tuple of branch and round number or default as keys and for each of them deterministic or stochastic rewards, while the arm-reward pairs are uniquely determined
        if isinstance(self.rewards,dict):
            if all([(isinstance(key,tuple) or isinstance(key,int) or key == "default") for key in self.rewards.keys()]) and "default" in self.rewards.keys():
                for key in self.rewards.keys():
                    if key != "default":
                        if isinstance(key,tuple):
                            if len(key) == 2:
                                if isinstance(key[0],int) and isinstance(key[1],int):
                                    if (key[0] in range(self.num_branches)) and (key[1] in range(self.grid_size[key[0]])):
                                        if self.num_arms[key[0]][key[1]] == 1:
                                            if isinstance(self.rewards[key],list):
                                                if len(self.rewards[key]) == 2:
                                                    if (isinstance(self.rewards[key][0],str) and isinstance(self.rewards[key][1],dict)):
                                                        check_for_allowed_dist(self.rng,self.rewards[key][0],**self.rewards[key][1])
                                                else:
                                                    raise ValueError("For stochastic rewards a list containing the distribution and a dictionary of keyword arguments need to be passed!")
                                            elif isinstance(self.rewards[key],(int,float)):
                                                pass
                                            else:
                                                raise TypeError(f"The rewards for {key} need to be either numeric or a list containing a sample distribution and a dictionary of keyword arguments, as there is only one arm to choose from in the specified step!")
                                        else:
                                            if isinstance(self.rewards[key],list):
                                                specified_arms = []
                                                for item in self.rewards[key]:
                                                    if isinstance(item,list):
                                                        if len(item) == 3:
                                                            if isinstance(item[0],int) and isinstance(item[1],int):
                                                                if item[0] >= 0 and item[0] < item[1] and item[1] < self.num_arms[key[0]][key[1]]:
                                                                    for arm in range(item[0],item[1] + 1):
                                                                        specified_arms.append(arm)
                                                                    if isinstance(item[2],list):
                                                                        if len(item[2]) == 2:
                                                                            if (isinstance(item[2][0],str) and isinstance(item[2][1],dict)):
                                                                                check_for_allowed_dist(self.rng,item[2][0],**item[2][1])
                                                                        else:
                                                                            raise ValueError("For stochastic rewards a list containing the distribution and a dictionary of keyword arguments need to be passed!")
                                                                    elif isinstance(item[2],(int,float)):
                                                                        pass
                                                                    else:
                                                                        raise TypeError("The rewards need to be either numeric or a list containing a sample distribution and a dictionary of keyword arguments!")
                                                                else:
                                                                    raise ValueError("The lowest and highest arms associated to the rewards need to match to the range of the possible arms!")
                                                            else:
                                                                raise TypeError("The lists containing arm-specific rewards must contain the lowest and highest arm for which the following reward is specified as an integer in the places 0 and 1!")
                                                        elif len(item) == 2:
                                                            if isinstance(item[0],int):
                                                                if item[0] >= 0 and item[0] < self.num_arms[key[0]][key[1]]:
                                                                    specified_arms.append(item[0])
                                                                    if isinstance(item[1],list):
                                                                        if len(item[1]) == 2:
                                                                            if (isinstance(item[1][0],str) and isinstance(item[1][1],dict)):
                                                                                check_for_allowed_dist(self.rng,item[1][0],**item[1][1])
                                                                        else:
                                                                            raise ValueError("For stochastic rewards a list containing the distribution and a dictionary of keyword arguments need to be passed!")
                                                                    elif isinstance(item[1],(int,float)):
                                                                        pass
                                                                    else:
                                                                        raise TypeError("The rewards need to be either numeric or a list containing a sample distribution and a dictionary of keyword arguments!")
                                                                else:
                                                                    raise ValueError("The given arm associated to the reward needs to match the range of the possible arms!")
                                                            else:
                                                                raise TypeError("The list containing one arm-specific reward must contain the number of the arm as an integer in the place 0!")
                                                        else:
                                                            raise TypeError("The lists containing arm-specific rewards must contain lists of length two or three!")
                                                    else:
                                                        raise TypeError("The lists containing arm-specific rewards must contain lists!")
                                                if len(specified_arms) != len(set(specified_arms)):
                                                    raise ValueError("Multiple rewards were assigned to the same arm!")
                                            else:
                                                raise TypeError("For states in which multiple arms can be chosen the rewards must be specified by a list!")
                                    else:
                                        raise ValueError("Integers in keyword tuples must be in the range of the number of branches and steps per branch!")
                                else:
                                    raise TypeError("Keywords of the rewards dictionary must be tuples containing integers")
                            else:
                                raise TypeError("Keywords of the rewards dictionary must be tuples of length two!")
                        elif isinstance(key,int):
                            if key in range(self.num_branches):
                                if isinstance(self.rewards[key],list):
                                    if len(self.rewards[key]) == 2:
                                        if (isinstance(self.rewards[key][0],str) and isinstance(self.rewards[key][1],dict)):
                                            check_for_allowed_dist(self.rng,self.rewards[key][0],**self.rewards[key][1])
                                    else:
                                        raise ValueError("For stochastic rewards a list containing the distribution and a dictionary of keyword arguments need to be passed!")
                                elif isinstance(self.rewards[key],(int,float)):
                                    pass
                            else:
                                raise ValueError("The branch numbers for which a terminal value is specified need to be in the range of the number of branches present!")
                        else:
                            raise TypeError("Keywords of the rewards dictionary must be tuples or branch numbers!")
                    else:
                        if isinstance(self.rewards[key],list):
                            if len(self.rewards[key]) == 2:
                                if (isinstance(self.rewards[key][0],str) and isinstance(self.rewards[key][1],dict)):
                                    check_for_allowed_dist(self.rng,self.rewards[key][0],**self.rewards[key][1])
                            else:
                                raise ValueError("For stochastic rewards a list containing the distribution and a dictionary of keyword arguments need to be passed!")
                        elif isinstance(self.rewards[key],(int,float)):
                            pass
                        else:
                            raise TypeError("The rewards need to be either numeric or a list containing a sample distribution and a dictionary of keyword arguments!")
            else:
                raise TypeError("The keys need to either be default or a tuples!")
        else:
            raise TypeError("Variable rewards needs to be a dictionary!")
        if len(self.rewards.keys()) != len(set(self.rewards.keys())):
            raise ValueError("For some states multiple reward structures were assigned!")

        return 1