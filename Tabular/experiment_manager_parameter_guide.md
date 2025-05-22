# Parameter guide for executing experiments

The parameters are generally seperated into five categories:

- Parameters for controlling execution of the experiment
- Parameters for controlling the individual training steps
- Parameters to pass to the algorithms
- Parameters to pass to the policy
- Parameters to pass to the environments

The parameters in the head of the execute_experiment function are ordered with this respect. In the case of algorithms and environments, there may be parameters unique to some of the options. These will all be passed in a special argument, where all possible arguments will be explained in detail below.

## 1. Parameters for controlling execution of the experiment

### base_folder (Str)

The path to the folder in which the data should be saved. This folder needs to already exist.

### num_runs (Int)

The number of times the algorithm should be trained.

### progress (Bool)

If True, a progress bar will be displayed showing the progress for the amount of times the algorithm is trained.

### project_name (Str)

The name of the project for which the experiments are done. The results will be saved under the folder with this name in the base_folder.

### runtime_estimation (Bool):

If True, an estimate of the total runtime will be calculated and displayed.

### safe_mode (True):

If True, the inputs will be checked. The displayed messages might make it easier to understand what parameters need to be changed in which way in case of errors.

### verbose (Bool):

If True, the function is verbose, meaning more messages will be printed to the terminal.

## 2. Parameters for controlling the individual training steps

### algo (Algo)

The algorithm on which the training should be performed.

### algo_special_logs (Bool)

If True, the algorithm's special logs will be logged.

### algo_special_logs_kwargs (Dict)

If the algorithm's special logs are to be logged, this dictionary contains the necessary keywords. For the following algorithms there are special log options available:

#### ADDQ

| Keyword                | Type  | Description                                                                             |
|------------------------|-------|-----------------------------------------------------------------------------------------|
| which_sample_variances | tuple | A tuple containing a list of states and a list of corresponding actions to log the sample variances for. The first corrdinate contains the list of states, where each state needs to be a valid positive integer. The second coordinate contains a list of lists for each state, which contain the actions corresponding to the states given as positive integers. |

### bias_estimation (Bool)

If True, the various bias estimation metrics will be computed at the individual runs and the averages over the runs will be stored.

### correct_act_q_fct_mode (Str)

If the logging of the bias estimation metrics, the correct action rates, or the initial Q function estimations is on, the correct action and Q function need to be known to the function. The mode with which the correct actions and Q function will be determined can be either "manual" or "value_iteration". In the latter case, the value iteration algorithm will be performed to determine both.

### correct_act_q_fct_mode_kwargs (Dict)

The dictionary of necessary keyword arguments for the chosen mode of determination of the correct actions and Q function. For "manual", it needs the keys "correct_actions" and "correct_q_fct", which correspond to the list containing the lists of assumed best actions for each state and the dictionary mapping each state action pair to their assumend optimal Q function value. For "value_iteration", the keys need to be "n_max", "tol", "env_mean_rewards", and "mean_rewards_mc_runs", where "n_max" maps to a strictly positive integer representing the maximum amount of iterations for which the value iteration should be performed, "tol" maps to a strictly positive numerical value representing the desired maximum error after completing the value iteration, "env_mean_rewards" is a dictionary in the same shape as rewards for the chosen environment (see below) but with the mean of the rewards instead of possible stochastic rewards everywhere if the means are known and else it is an empty dictionary, and "mean_rewards_mc_runs" is the number of Monte Carlo runs per stochastic state that should be performed for determining the actual environment mean rewards in case the corresponding dictionary is left empty.

### correct_action_log (Bool)

If True, the correct action rates will be logged after each epoch and each evaluation cycle.

### correct_action_log_which (Str/List)

If the correct action rates are logged, describes which states should be considered for calculating the current policies' correct action rate. It can be passed as 'all', meaning all states will be considered or as a list of states passed as valid state numbers.

### env (Env)

The environment you want to train your algorithm on.

### eval_reseeding (Bool)

If True, the evaluation environment's seed will be reseeded after each evaluation run after the evaluation seed schedule is exhausted.

### eval_seed_schedule (List)
The list of seeds scheduled for being used during evaluations. For each instance of -1, a random seed will be drawn.

### eval_steps (int)

The number of steps that the environment should be played during each evaluation

### eval_freq (int)

The number of update steps that should be performed before the algorithm is evaluated.

### focus_state_actions (bool)
If True, the state action pairs contained in which_state_actions_focus will be focussed, meaning that their estimated Q values at evaluations and their biases at evaluations will be logged separately.

### max_steps_per_epoch (int)

The maximum amount of update steps the algorithm is allowed to perform before the epoch gets cut off. If it is -1, there is no maximum amount.

### num_steps (int)

The number of steps (as defined by the training mode) to be performed for each individual training cycle.

### policy (Policy)

The policy to be used during training

### progress_single_games (Bool)

If True, for each individual training cycle a separate progress bar will be shown (and dropped after completion), if the general progress bar is also displayed.

### training_mode (Str)

The chosen training mode. It can either be "steps" or "epoch", depending on if you want to play a fixed number of steps or a fixed number of epochs in each training cycle.

### training_reseeding (Bool)

If True, the algorithm's seed will be reseeded after each step (in the sense of the training mode) after the training seed schedule is exhausted.

### training_seed_schedule (List)
The list of seeds scheduled for being used during training. For each instance of -1, a random seed will be drawn.

### which_state_actions_focus (Tuple)

Describes for which state action pairs the Q function and the bias at evaulations are to be logged seperately. The first argument of the tuple is a list of integers corresponding to the chosen states. 'start' can be passed, corresponding to the start state. The second argument is a list of lists of actions corresponding to each of the chosen states. The actions can be either passed with their numerical value or one of the action values can be "best", meaning for this state, the best action (according to the estimated best actions) will be chosen.

## 3. Parameters to pass to the algorithms

### learning_rate_kwargs (Dict)

A dictionary that contains the keyword arguments for scheduling the learning rate. The following table summarizes the keywords and their options.

| Keyword      | Type  | Description                                                                      |
|--------------|-------|----------------------------------------------------------------------------------|
| initial_rate | float | The initial stepsize to be used at the beginning of the scheduling process. |
| mode         | str   | The mode with which the stepsize should be stepwise updated. The implemented modes are: "constant", meaning the initial rate will always be used as stepsize. "linear", meaning the initial rate and a desired end rate may be linearly interpolated between based on a set amount of steps or a slope in order to schedule the stepsizes. "rate", meaning a specified rate function is used to schedule the stepsizes until a final rate is reached. |
| mode_kwargs  | dict  | A dictionary containing the necessary keyword arguments for the chosen scheduling mode. For "constant" this needs to be a dictionary containing the keyword "final_rate" mapping to the same value as initial_rate. For "linear" it needs to be a dictionary containing the keywords "final_rate", "num_steps", and "slope", mapping to the desired final rate, the number of steps upon which it should be reached, and the slope at which this should happen. The value of the slope may either be positive, in which case the num_steps argument will be ignored and the slope will be used until the rate hits the final rate value, or it may be -1, meaning the num_steps argument (passed as a positive integer) will be used to determine the slope automatically. For "rate" it needs to be a dictionary containing the keywords "rate_fct", "iteration_num", and "final_rate", mapping to a rate function to be used, which should be a decreasing lambda funtion, the current iteration number, which needs to be set to one, and the desired final rate. |

### learning_rate_state_action_wise (Bool)

If True, it means that the learning rate schedule will be executed seperately for each state action pair as opposed to being updated for all state action pairs at the same time on each step. If the algorithm works using multiple copies, the learning rate schedule will be applied seperately for each copy.

### gamma (Float)

The discount factor to be used for the value functions given as a float between 0 and 1.

### algo_specific_params (Dict)

The rest of the possible parameters are algorithm specific. Some of them are equal for classes of algorithms. Therefore, in the following, for each class of algorithm and some individual algorithms a description of the parameters you can pass is provided.

#### 1. Q function based algorithms

This class currently encases Q, Double, and WDQ.

| Keyword           | Type            | Description                                                      |
|-------------------|-----------------|-------------------------------------------------------------------|
| q_fct_manual_init | bool            | If True, the Q function(s) will be manually initialized, if False the initial Q functions will take the value 0 on all state action pairs. |
| initial_q_fct     | dict/list(dict) | The dictionary (or list of dictionaries) containing the initialization(s) of the Q function(s). In the case of the Q algorithm one dictionary needs to be passed, in the case of the Double and WDQ algorithm either one dictionary (meaning both Q functions will be initialized with the same custom Q function), or a list of two dictionaries may be passed. The keys of the dictionary (or dictionaries) should correspond to all allowed tuples of state and action (as integers) for the chosen game and the values of the dictiony (or dictionaries) to the Q values to be initialized. |

#### 2. Categorical Distributional based algorithms

This class currently encases CategoricalQ, CategoricalDouble, and ADDQ.

| Keyword                | Type            | Description                                                  |
|------------------------|-----------------|--------------------------------------------------------------|
| num_atoms              | int             | The number of atoms used for the atom net.                   |
| range_atoms            | list            | The range in which the atom net will be laid out. It needs to be passed as a list of two numerical values, where the first one is strictly smaller than the second one. |
| atom_probs_manual_init | bool            | If True, the atom probabilies will be manually initialized, if False the initial atom probabilities will be uniform on all atoms for all state action pairs. |
| initial_atom_probs     | dict/list(dict) | The dictionary (or list of dictionaries) containing the initialization(s) of the atom probabilities. In the case of the CategoricalQ algorithm one dictionary needs to be passed, in the case of the CategoricalDouble and ADDQ algorithm either one dictionary (meaning both atom probability nets will be initialized with the same custom probability net), or a list of two dictionaries may be passed. The keys of the dictionary (or dictionaries) should correspond to all allowed tuples of state and action (as integers) for the chosen game and the values of the dictiony (or dictionaries) to the atom probabilities to be initialized. |

#### 3. WDQ

| Keyword                   | Type | Description                                                          |
|---------------------------|------|----------------------------------------------------------------------|
| interpolation_mode        | bool | The chosen interpolation mode between the Q and Double algorithm updates. The implemented modes are: "constant", meaning that always the same interpolation constant is chosen. "adaptive", meaning that an adaptive choice (based on [this paper](https://www.ijcai.org/proceedings/2017/483)) of the interpolation constant is chosen in each step. |
| interpolation_mode_kwargs | dict | The dictonary containing the necessary keyword arguments for the chosen interpolation mode. For "constant" the key "beta" needs to map to the chosen constant interpolation coefficient, passed as a numerical value between 0 and 1. For "adaptive" it needs to contain the keys "c", "current_state", "which_copy", and "which_reference_arm", all relating to equation (7) of the paper. The chosen constant "c" needs to be a positive numerical value, if "current_state" is True, instead of the next state (as detailed in the paper) the current state with its highest yielding arm wrt the Q function will be chosen, "which_copy" refers to if we use the index not chosen for the update ("other") to get the absolute values of the Q functions in the equation, or use the chosen index ("chosen") or an average of both ("average") for the calculation of the interpolation coefficient. Finally, "which_reference_arm" refers to if we use the "lowest", "median", or "second_highest" arm as reference arm a<sub>L</sub> in the calculation of the interpolation coefficient. |

#### 4. ADDQ

| Keyword                   | Type | Description                                                          |
|---------------------------|------|----------------------------------------------------------------------|
| interpolation_mode        | bool | The chosen interpolation mode between the CategoricalQ and CategoricalDouble algorithm updates. The implemented modes are: "constant", meaning that always the same interpolation constant is chosen. "adaptive", meaning that an adaptive choice (based on our paper to be published) of the interpolation constant is chosen in each step. |
| interpolation_mode_kwargs | dict | The dictonary containing the necessary keyword arguments for the chosen interpolation mode. For "constant" the key "beta" needs to map to the chosen constant interpolation coefficient, passed as a numerical value between 0 and 1. For "adaptive" it needs to contain the keys "center", "left_truncated", "which", "current_state", "bounds", and "betas", all relating to the exemplary adaptive beta choices in the paper. "center" refers to from which centerpoint the measure of dispersion computed should be taken, where the choice is between "variance" and "median". If "left_truncated" is True, then the left truncation based on the chosen centerpoint is used instead of the whole range of atoms to compute the dispersion measure, meaning only atoms to the right of the centerpoint will be contributing to the calculations. "which" refers to if we use the index not chosen for the update ("other"), the one chosen ("chosen"), or an average ("average") to compute the average of the dispersion measure and if "current_state" is True we use the current state's dispersal, if it is False the next state's one. "bounds" and "betas" refers to the choice of the interpolation constant based on the averaged and normalized measure of dispersal lying in intervals. "bounds" is a list of tuples containing the upper bound of consecutive intervals and "strict" or "not_strict" corresponding to open, respectively closed upper intervals. Therefore, the values of the first entry of the tuple need to be strictly positive numerical values that are strictly increasing in the index number. "betas" then is a list of numerical values between 0 and 1, where the value at index i corresponds to the assigned interpolation coefficient on interval i. |

## 4. Parameters to pass to the environment

### env_specific_params (Dict)

For environments, there are no parameters shared in common. Therefore, in the following, for each game a description of the parameters you can pass is provided.

#### 1. GridWorld

| Keyword        | Type  | Description                                                                    |
|----------------|-------|--------------------------------------------------------------------------------|
| grid_size      | tuple | The size of the grid given as (rows, columns). Both must be positive integers. |
| state_type_loc | dict  | The dictionary mapping state types given as strings with their name to locations and information if its a terminal state. Each entry is a tuple where the first element is a list of coordinates, given as (row,column) for that state type, where the row and column numeration starts with one, and the second element is a boolean indicating whether the state is terminal. Needs to contain the locations of "goal" and "start". The goal must be terminal, while the start can not be terminal. |
| rewards        | dict  | The dictionary mapping state types given as strings with their name (The same keywords as in the stae_type_loc dictionary need to be passed) to their respective rewards. The special key "default" is used for all states not specified in the dictionary and must be passed. Some states may have stochastic rewards represented by a distribution (e.g., normal). In this case a list containing the distribution name and a dictionary of keyword arguments compatible with the numpy random generator need to be passed. In all other cases, the reward must be a numerical value. |
| hovering       | bool  | If True, the player is allowed to choose actions that make it bump into the wall and thus hover in the same place. |
| windy          | bool  | If True, wind is applied to the environment and the player may be pushed in a direction given by wind_dir instead of the direction of the action it chooses. Can not be activated in combination with slippery and/or random_actions. |
| wind_prob      | float | The probability that wind will affect the environment in each step if turned on. Must be a numerical value between 0 and 1. |
| wind_dir       | str   | Direction of the wind as one of the following strings: "up", "right", "down", "left". |
| slippery       | float | If True, the environment is slippery, causing random movement adjacent to the player's chosen action. Can not be activated in combination with windy and/or random_actions. |
| slip_prob      | float | The probability that a random slip occurs if turned on. Must be a numerical value between 0 and 1 |
| random_actions | bool  | If True, the environment may perform random actions instead of the player's chosen action. Can not be activated in combination with windy and/or slippery. |
| random_prob    | float | The probability that a random action will be taken if turned on. Must be a numerical value between 0 and 1. |
| random_vec     | list  | A list of four probabilities, corresponding in order to the probability of randomly moving up, right, down, and left, if turned on. All values in the list need to be numerical and between 0 and 1. |

#### 2. SuttonExtended

| Keyword        | Type  | Description                                                                    |
|----------------|-------|--------------------------------------------------------------------------------|
| grid_size | tuple | A tuple whose length determines the number of branches, which will be numbered from 0 to (length of tuple - 1). The entries of the tuple determine the number of consecutive steps per branch, which will be numbered from 0 to the entry (which needs to be an integer), where 0 refers to the first step taken from the origin to get to the branch. |
| num_arms  | tuple | A tuple containing tuples specifying the number of arms in all steps previous of the end of the branch for each branch. For each branch and each step, the number must be a positive integer. Please be careful to include the comma in typles of length one (e.g. (int,)) so that the type tuple will be recognized. |
| rewards   | dict  | A dictionary mapping states to their rewards in the following way: The states can either be "default" (which needs to always been passed), meaning the value assigned to all non-specified state-action pairs, a valid branch number in the form of an integer, meaning the reward that is collected at the end of a branch with the specified number, or a tuple of branch and step number, both valid integers, meaning the specified step at the specified branch. If you wish to assign all arms the same value, you may simply specify a reward as the value corresponding to the key in the dictionary. If you want to assign rewards to individual arms, the value of the corresponding key in the dictionary needs to be a list of lists. each of the interior lists has either length 2 or 3, where the first (respectively the first two) values are integers and the second (respectively the third) is the reward. For lists of length two, the integer specifies the arm for which the reward in the list will be assigned. For lists of length three, the first two integers specify the range of arms for which the reward in the list will be assigned, meaning the first entry needs to be strictly smaller than the second. It is not possible to assign the same arm multiple values. Some states may have stochastic rewards represented by a distribution (e.g., normal). In this case a list containing the distribution name and a dictionary of keyword arguments compatible with the numpy random generator need to be passed. In all other cases, the reward must be a numerical value. |

## 5. Parameters to pass to the policy

### policy_specific_params (Dict)

There is only one implemented policy at the moment. But since there might be more in the future that do not share the same parameters, in the following, for the implemented policy a description of the parameters you can pass is provided with the option to supplement the list in the future.

#### 1. BasePolicy

| Keyword            | Type | Description                                                                 |
|--------------------|------|----------------------------------------------------------------------------|
| policy_mode        | str  | Mode for choosing the next action via the policy. The implemented modes are: "offpolicy", meaning a certain policy determined by the kwargs will always be played. "epsilon_greedy" and "epsilon_greedy_statewise", meaning the policy will decide greedy with respect to the current Q function and a certain rate in each step, which will be updated via a schedule determined by the kwargs either for all steps at once or individually, if statewise is chosen. "greedy", meaning the policy will decide greedy with respect to the current Q function. "softmax", meaning the policy will decide with respect to the softmax of the current Q function. |
| policy_mode_kwargs | dict | The dictionary containing the necessary keyword arguments for the chosen policy mode. These will be explained in the table below for each policy mode individually. |

##### Policy mode "offpolicy" 

| Keyword | Type | Description                                                                            |
|---------|------|----------------------------------------------------------------------------------------|
| type    | str  | The type of behaviour policy chosen. The implemented types are: "uniform_random", meaning the behaviour policy will be a uniform random one over all allowed actions in each state. "full_init", meaning a behaviour policy will be manually passed. |
| kwargs  | dict | The dictionary containing the necessary keyword arguments for the chosen behaviour policy type. For "uniform_random" this needs to be an empty dictionary. For "full_init" it needs to only contain the keyword "policy_list", mapping to the desired behaviour policy. The behaviour policy needs to be passed as a list that has the same length as the played game has number of states and contain the actions. For some states the behaviour policy may have stochastic preferences among the arms. In this case a list containing the distribution name "choice" and a dictionary containing the keywords "a" and "p", mapping to a list of arms "a" (passed as integers), between which the agent may choose with probabilities "p", need to be passed. In all other cases, the action must be a numerical value. |

##### Policy modes "epsilon_greedy" and "epsilon_greedy_statewise

| Keyword      | Type  | Description                                                                      |
|--------------|-------|----------------------------------------------------------------------------------|
| initial_rate | float | The initial epsilon to be used at the beginning of the scheduling process. |
| mode         | str   | The mode with which the rate should be stepwise updated. The implemented modes are: "constant", meaning the initial rate will always be used as epsilon. "linear", meaning the initial rate and a desired end rate may be linearly interpolated between based on a set amount of steps or a slope in order to schedule the epsilon. "rate", meaning a specified rate function is used to schedule the epsilon until a final rate is reached. |
| mode_kwargs  | dict  | The dictionary containing the necessary keyword arguments for the chosen scheduling mode. For "constant" this needs to be a dictionary containing the keyword "final_rate" mapping to the same value as initial_rate. For "linear" it needs to be a dictionary containing the keywords "final_rate", "num_steps", and "slope", mapping to the desired final rate, the number of steps upon which it should be reached, and the slope at which this should happen. The value of the slope may either be positive, in which case the num_steps argument will be ignored and the slope will be used until the rate hits the final rate value, or it may be -1, meaning the num_steps argument (passed as a positive integer) will be used to determine the slope automatically. For "rate" it needs to be a dictionary containing the keywords "rate_fct", "iteration_num", and "final_rate", mapping to a rate function to be used, which should be a decreasing lambda funtion, the current iteration number, which needs to be set to one, and the desired final rate. |

##### Policy mode "greedy"

For this policy mode, the dictionary "policy_mode_kwargs" needs to be left empty.

##### Policy mode "softmax"

| Keyword     | Type  | Description                                                                       |
|-------------|-------|-----------------------------------------------------------------------------------|
| temperature | float | The temperature used in the softmax function.                                     |