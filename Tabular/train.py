import numpy as np
from tqdm import tqdm
from typing import Dict, Tuple, List, Union, Any
import time
from copy import deepcopy

import algos
import envs
from utils import generate_random_seed, sample_from_dist

def train(
		algo: algos.Algo = None, # The chosen algorithm for training
		algo_kwargs: Dict = None, # The keyword arguments for the algorithm
		algo_special_logs: bool = False, # If the algorithm's special parameters should be logged
		algo_special_logs_kwargs: Dict = None, # The necessary keyword arguments for logging the algorithm's special parameters
		env: envs.Env = None, # The chosen environment 
		env_kwargs: Dict = None, # The keyword arguments for the environment
		training_mode: str = "steps", # Choose, if a number of total update-steps or a number of total epochs should be played during training
		num_steps: int = 500, # Choose the number of steps or epochs
		max_steps_per_epoch: int = -1, # Maximal number of update-steps per epoch before restarting the game
		training_seed_schedule: List[Union[int,float]] = None, # Seed schedule for initializing the random number generators involved
		training_reseeding: bool = False, # Should we reseed after an epoch finishes
		eval_freq: int = 10, # Number of timesteps after which an evaluation of the policy will occur
		eval_steps: int = 7, # Number of evaluation steps for the current policy
		eval_seed_schedule: List[Union[int,float]] = None, # Schedule for evaluation seeds
		eval_reseeding: bool = True, # Should we reseed for each evaluation
		bias_estimation: bool = True, # Should a bias estimation be run at each evaluation
		focus_state_actions: bool = True, # Should some state action pairs be focused and have their metrics separately logged
		which_state_actions_focus: Tuple[List[int],List] = None, # Which state action pairs should have their metrics separately logged
		correct_action_log: bool = True, # Should it be logged when correct actions were played
		correct_action_log_which: Union[str,List] = "all", # Over which states should the correct action rate at evaluations be considered
		correct_act_q_fct_mode: str = "value_iteration", # How should the correct actions and the correct q function be determined
		correct_act_q_fct_mode_kwargs: Dict[str,Any] = None, # The keyword arguments for the determination of the correct actions and q function
		safe_mode: bool = True, # Checks will be run, in case of error messages it will be clearer where they come from
		progress: bool = True, # If on, displays a progress bar during training
		measure_runtime: bool = False, # If on, measures the time the algorithm took to be exercised
) -> Tuple[List,Dict[str,Dict[str,List[Union[int,float]]]],Dict[str,Any],Union[int,float]]: 
	
	"""
		Performs training via an algorithm on an environment. The number of steps to perform updates for can be either 
		chosen to be a certain number of epochs or a certain number of total timesteps, where in each epoch a maximum
		number of steps is performed. Optionally, estimates of the bias can be logged as well. A seed schedule can be 
		provided or is chosen randomly, where choosing to reseed after each epoch is possible. The agent is evaluated 
		periodically with a certain number of evaluation steps. At evaluation steps, the environment can be reseeded 
		or an evaluation seed schedule can be provided. A progress bar can be activated and the runtime can be measured.

		Parameters:
		- algo (Algo): The algorithm to be used.
		- algo_kwargs (dict): The algorithm keyword arguments to be used. Please do not pass environment name and/or
		  environment keyword arguments here. Instead, pass them seperately below. Also, do not include the seed and
		  checks there.
		- algo_special_logs (bool): If True, the algorithm's special parameter can get logged.
		- algo_special_logs_kwargs (dict): If the algorithm's special parameters can get logged, the necessary keyword
		  arguments for doing so need to be insinde this dictionary.
		- env (Env): The environment the algorithm is executed on.
		- env_kwargs (Dict): The keyword arguments with which the environment is initialized. Do not include the seed
		  and the checks there.
		- training_mode (str): The training mode. It can be either steps or epoch.
		- num_steps (int): The number of training steps or epoch steps to be used.
		- max_steps_per_epoch (int): The maximum number of steps to be done per epoch. Default is -1, meaning there is no 
		  max specified.
		- training_seed_schedule (list): A list containing a single seed or a seed schedule to be used for the training 
		  process. When the end of the list is reached, reseeding will start if activated. If not, the last seed will then 
		  be used for the rest of the training time. Default seed value is -1, meaning a random seed is chosen and logged.
		- training_reseeding (bool): If True, once the provided list of scheduled seeds is exhausted, random reseeding will
		  start.
		- eval_freq (int): The frequency of steps after which a an evaluation should take place.
		- eval_steps (int): The duration of the evaluation in timesteps.
		- eval_seed_schedule (list): A list containing a single seed or a seed schedule to be used for the evaluations. When
		  the end of the list is reached, reseeding will start if activated. If not, the last seed will then be used for the
		  rest of the evaluations. Default seed value is -1, meaning a random seed is chosen and logged.
		- eval_reseeding (bool): If True, once the provided list of scheduled seeds is exhausted, random reseeding will start.
		- bias_estimation (bool): If True, an estimation of the bias will be logged after each evaluation.
		- focus_state_actions (bool): If True, the Q-value and Bias of some specified states will be logged.
		- which_state_actions_focus (list): A list of states and corresponding actions for which the Q-Value and bias estimation 
		  should be logged individually. For the actions, either a list of actions, a list containing a single action, or the 
		  string 'best' can be passed, meaning the estimated optimal action for that state. The states need to be either valid
		  integers or the string 'start'.
		- correct_action_log (bool): If True, it will be logged whether at each step the correct action was played or not and
		  the rate of correct actions after each epoch and after an evaluation cycle will be logged as well.
		- correct_action_log_which (str): The states over which the correct action rates should be averaged at evaluation. Can
		  be either 'all', meaning the average will be taken over all states, or a list of valid state numbers.
		- correct_act_q_fct_mode (str): The mode for determining the correct actions and Q function in case bias_estimation or
		  correct_action_log are on. You can choose between a manual initialization and value iteration.
		- correct_act_q_fct_mode_kwargs (dict): The necessary parameters for the chosen mode of determining the correct actions
		  and Q function.
		- safe_mode (bool): If True, the safe mode will be activated, where checks are performed and any error messages will 
		  be easier understood.
		- progress (bool): If True, a progress bar will show the amount of work left.
		- measure_runtime (bool): If True, measures the time the training of the algorithm took with the provided parameters.

		Returns:
		- final_policy (list): The policy after completing the training. It is a list that contains the chosen arm for each 
		  state.
		- final_q_fct (dict): The Q function after completing the training. It is a dictionary containing the Q function value
		  for each state action tuple.
		- estimated_correct_policy (list): The policy that was estimated or passed to be correct in case bias estimation or 
		  logging of the correct actions was turned on. It is a list that contains the chosen arm for each state.
		- estimated_correct_q_fct (dict): The Q function that was estimated or passed to be correct in case bias estimation or
		  logging of the correct actions was turned on. It is a dictionary containing the Q function value for each state action 
		  tuple.
		- results (dict): Dictionary that contains all relevant result metrics. It has three main entries:
			- at_step: Dictionary containing timesteps, rewards, and if activated if a correct action was played at the timesteps.
			- at_epoch: Dictionary containing epoch numbers, epoch starting times, epoch durations, information on if the 
			  epoch was capped, the epoch results, if activated the correct action rates, and the chosen training seeds.
			- at_eval: Dictionary containing evaluation times, evaluation scores, if activated the initial Q function values at the
			  best arm, if activated the  correct action rates, if activated the several different evaluation biases, and the 
			  chosen evaluation seeds. The evaluation scores are tuples consisting of the summed discounted scores, the number 
			  of terminal states reached and the time at which the last terminal state was reached.
		- parameters (dict): A Dictionary containing all passed parameters for reference and reproducibility.
		- runtime (float): The time it took to train the algorithm with the given parameters. If the value is -1, the runtime was
		  not measured.
		"""
	
	# Default arguments if arguments are None
	if algo is None:
		algo = algos.Q
	if algo_kwargs is None:
		algo_kwargs = {}
	if env is None:
		env = envs.GridWorld
	if env_kwargs is None:
		env_kwargs = {}
	if training_seed_schedule is None:
		training_seed_schedule = {}
	if eval_seed_schedule is None:
		eval_seed_schedule = {}
	if which_state_actions_focus is None:
		which_state_actions_focus = ([0],[["best"]])
	if correct_act_q_fct_mode_kwargs is None:
		correct_act_q_fct_mode_kwargs = {"n_max": 1000, "tol": 0.0001, "env_mean_rewards": {},"env_mean_rewards_mc_runs": 1000000}
	if algo_special_logs_kwargs is None:
		algo_special_logs_kwargs = {}
	
	# Initialize outputs
	final_policy = []
	final_q_fct = {}
	estimated_correct_policy = []
	estimated_correct_q_fct = {}
	results = {
		"at_step": {
			"timesteps": [],
			"rewards": [],
			**({"correct actions": []} if correct_action_log else {})
		},
		"at_epoch": {
			"epoch numbers": [],
			"epoch starting times": [],
			"epoch durations": [],
			"capped": [],
			"epoch results": [],
			"training seeds": [],
			**({"correct action rates": []} if correct_action_log else {})
		},
		"at_eval": {
			"evaluation times": [],
			"evaluation scores": [],
			"evaluation seeds": [],
			**({"correct action rates at evaluations": [], "correct action rates at chosen states": [],"total evaluation biases": [],"total evaluation biases at best arms": [], "total normalized evaluation biases": [],"total normalized evaluation biases at best arms": [], "total squared evaluation biases": [],"total squared evaluation biases at best arms": [], "total squared normalized evaluation biases": [],"total squared normalized evaluation biases at best arms": [], "Q function values at chosen states and actions": [], "evaluation biases at chosen states and actions": []} if (bias_estimation and correct_action_log and focus_state_actions) else (
				{"Q function values at chosen states and actions": [], "evaluation biases at chosen states and actions": [], "correct action rates at evaluations": [], "correct action rates at chosen states": []} if (correct_action_log and focus_state_actions) else (
					{"total evaluation biases": [],"total evaluation biases at best arms": [], "total normalized evaluation biases": [],"total normalized evaluation biases at best arms": [], "total squared evaluation biases": [],"total squared evaluation biases at best arms": [], "total squared normalized evaluation biases": [],"total squared normalized evaluation biases at best arms": [], "Q function values at chosen states and actions": [], "evaluation biases at chosen states and actions": []} if (bias_estimation and focus_state_actions) else (
						{"correct action rates at evaluations": [], "correct action rates at chosen states": [],"total evaluation biases": [],"total evaluation biases at best arms": [], "total normalized evaluation biases": [],"total normalized evaluation biases at best arms": [], "total squared evaluation biases": [],"total squared evaluation biases at best arms": [], "total squared normalized evaluation biases": [],"total squared normalized evaluation biases at best arms": []} if (bias_estimation and correct_action_log) else (
							{"Q function values at chosen states and actions": [], "evaluation biases at chosen states and actions": []} if focus_state_actions else (
								{"correct action rates at evaluations": [], "correct action rates at chosen states": []} if correct_action_log else (
									{"total evaluation biases": [],"total evaluation biases at best arms": [], "total normalized evaluation biases": [],"total normalized evaluation biases at best arms": [], "total squared evaluation biases": [],"total squared evaluation biases at best arms": [], "total squared normalized evaluation biases": [],"total squared normalized evaluation biases at best arms": []} if bias_estimation else {}
								)
							)
						)
					)
				)
			))
		}
	}
	parameters = {
		"algo": algo,
		"algo_kwargs": algo_kwargs,
		"env": env,
		"env_kwargs": env_kwargs,
		"training_mode": training_mode,
		"num_steps": num_steps,
		"max_steps_per_epoch": max_steps_per_epoch,
		"training_seed_schedule": training_seed_schedule,
		"training_reseeding": training_reseeding,
		"eval_freq": eval_freq,
		"eval_steps": eval_steps,
		"eval_seed_schedule": eval_seed_schedule,
		"eval_reseeding": eval_reseeding,
		"bias_estimation": bias_estimation,
		"focus_state_actions": focus_state_actions,
		"which_state_actions_focus": which_state_actions_focus,
		"correct_action_log": correct_action_log,
		"correct_action_log_which": correct_action_log_which,
		"correct_act_q_fct_mode_kwargs": correct_act_q_fct_mode_kwargs,
		"safe_mode": safe_mode,
		"progress": progress,
		"measure_runtime": measure_runtime,
	}
	runtime = -1
	
	# Find intended seeds for algorithm and evaluation environment initialization
	if training_seed_schedule[0] != -1:
		training_seed = training_seed_schedule[0]
	else:
		training_seed = generate_random_seed()
	if eval_seed_schedule[0] != -1:
		eval_seed = eval_seed_schedule[0]
	else:
		eval_seed = generate_random_seed()

	# If safe mode is on, initial checks will be run, if not, no checks will be run
	if safe_mode:
		checks = "only_initial_checks"
	else:
		checks = "no_checks"
		
	# Initialize training algorithm and environment
	training_algo = algo(
		env = env,
		env_kwargs = env_kwargs,
		rng_seed = training_seed,
		checks = checks,
		special_logs_kwargs = algo_special_logs_kwargs,
		**algo_kwargs
	)
	evaluation_environment = env(
		rng_seed = eval_seed,
		checks = checks,
		**env_kwargs
	)

	# If special logs are activated, add them to the initialization of the output
	if algo_special_logs:
		special_log_keys = training_algo.get_special_log_keys()
		if len(special_log_keys) != 0:
			results["special"] = {}
			for item in special_log_keys:
				if item[0] in results["special"].keys():
					results["special"][item[0]][item[1]] = []
				else:
					results["special"][item[0]] = {}
					results["special"][item[0]][item[1]] = []

	# If bias estimation, correct action logging, or focussing state actions is on, determine the correct actions and the correct q functions
	if bias_estimation or correct_action_log or focus_state_actions:

		# If manual initialisation simply take the passed values
		if correct_act_q_fct_mode == "manual":
			estimated_correct_policy = deepcopy(correct_act_q_fct_mode_kwargs["correct_actions"])
			estimated_correct_q_fct = deepcopy(correct_act_q_fct_mode_kwargs["correct_q_fct"])
		
		# If value iteration was chosen, perform value iteration
		elif correct_act_q_fct_mode == "value_iteration":

			# Determine the stopping criterion of value iteration
			epsilon = (correct_act_q_fct_mode_kwargs["tol"] * (1 - training_algo.gamma)) / training_algo.gamma

			# Initialize Q function, counter for maximal steps, and delta higher than epsilon
			val_it_q_fct = {(state, action): 0 for state in range(evaluation_environment.num_states) for action in evaluation_environment.allowed_actions[state]}
			steps_done = 0
			delta = epsilon + 1

			# Get the p_values and the mean rewards
			game_probabilities_dict = evaluation_environment.game_probabilities
			mean_rewards_dict = evaluation_environment.mean_rewards_to_state_action(correct_act_q_fct_mode_kwargs["env_mean_rewards"],correct_act_q_fct_mode_kwargs["env_mean_rewards_mc_runs"])

			# While error epsilon has not been reached and the maximum amount of steps have not yet been taken, perform the value iteration steps
			while (delta > epsilon) and steps_done <= correct_act_q_fct_mode_kwargs["n_max"]:

				# Update the Q function
				new_val_it_q_fct = {}
				for state in range(evaluation_environment.num_states):
					for action in evaluation_environment.allowed_actions[state]:

						# Determine reward of state action pair
						r = mean_rewards_dict[(state,action)]

						# Determine all possible next states with their probabilities
						next_act_prob_dict = {key: prob_val for key,prob_val in game_probabilities_dict.items() if (key[1] == state and key[2] == action)}
						possible_next_states = [key[0] for key in next_act_prob_dict.keys()]
						
						# Determine maximal Q values for all possible next states
						next_states_q_values = {next_state: {ke: val for ke, val in val_it_q_fct.items() if ke[0] == next_state} for next_state in possible_next_states}
						next_states_max_q_values = [max(next_states_q_values[next_state].values()) for next_state in possible_next_states]

						# Determine the expected value of the Q function at the next state and best action
						sum = 0
						for i,next_state in enumerate(possible_next_states):
							sum += game_probabilities_dict[(next_state,state,action)] * next_states_max_q_values[i]

						# Determine the next q function value
						new_val_it_q_fct[(state,action)] = r + training_algo.gamma * sum

				# Update the steps done and the delta
				steps_done += 1
				differences = {key: abs(val_it_q_fct[(key[0],key[1])] - new_val_it_q_fct[(key[0],key[1])]) for key in val_it_q_fct.keys()}
				delta = max(differences.values())

				# Update the current q function estimate
				val_it_q_fct = new_val_it_q_fct

			# Assign the estimated correct Q function and the estimated correct actions as its greedy actions
			estimated_correct_q_fct = val_it_q_fct
			# Compute greedy actions
			greedy_policy = []
			for state in range(evaluation_environment.num_states):
				state_q_fct = {key: value for key, value in val_it_q_fct.items() if key[0] == state}
				max_val = max(state_q_fct.values())
				arg_max = [key[1] for key, value in state_q_fct.items() if value == max_val]
				greedy_policy.append(arg_max)
			estimated_correct_policy = greedy_policy

		else:
			raise ValueError("The mode of determination for the correct actions and Q function you passed seems to not exist. If you tried implementing a new one, make sure to specify how to determine them in your train function!")

	# Start with timestep and epoch zero
	timestep = 0
	epoch = 0
	counter = 1
	eval = 0

	# Measure time if needed
	if measure_runtime:
		start_time = time.time()

	# Iterator, depending on if I want a progress bar or not
	if progress:
		iterator = tqdm(total = num_steps, desc="Executing algorithm steps", leave = False)
		
	# While we did not reach the end, continue training
	while training_mode == "epoch" or training_mode == "steps":
		
		# Start epoch again once finished and update epoch number
		epoch += 1
		timestep += 1
		
		# Log epoch number, starting time, and training seed
		results["at_epoch"]["epoch numbers"].append(epoch)
		results["at_epoch"]["epoch starting times"].append(timestep)
		results["at_epoch"]["training seeds"].append(training_seed)

		while training_mode == "epoch" or training_mode == "steps":
		
			# Log timestep number
			results["at_step"]["timesteps"].append(timestep)

			# Execute the algorithm step, obtain the reward and chosen action during training
			state, chosen_action, _, reward_during_step, epoch_done = training_algo.step()

			# Look up if the chosen action is correct and log it if correct_action_log was chosen
			if correct_action_log:
				correct = 0
				if chosen_action in estimated_correct_policy[state]:
					correct = 1
				results["at_step"]["correct actions"].append(correct)

			# Log reward at timestep 
			results["at_step"]["rewards"].append(reward_during_step)

			# If some of the special parameters need to get logged log them
			if algo_special_logs:
				if "special" in results.keys():
					if "at_step" in results["special"].keys():
						special_results_to_log = training_algo.get_special_logs_at_step()
						if special_results_to_log != []:
							for label_value in special_results_to_log:
								results["special"]["at_step"][label_value[0]].append(label_value[1])
						
			# If timestep is divisible by evaluation frequency, do an evaluation cycle
			if timestep % eval_freq == 0:

				# Update the number of the evaluation
				eval += 1

				# Log Evaluation time and seed
				results["at_eval"]["evaluation times"].append(timestep)
				results["at_eval"]["evaluation seeds"].append(eval_seed)

				# Get Greedy Policy at the moment
				eval_policy = training_algo.get_greedy_policy()

				# Initialize the reward and terminal state lists, set the first state to starting state, and the score to zero
				eval_rewards = []
				eval_terminal = []
				eval_state = evaluation_environment.start_state_num
				eval_score = 0
				if correct_action_log:
					correct_actions_played_at_eval = 0

				# Evaluate for eval_steps time
				for _ in range(eval_steps):
					chosen_action = eval_policy[eval_state]
					if correct_action_log:
						if chosen_action in estimated_correct_policy[eval_state]:
							correct_actions_played_at_eval += 1
					eval_state, ter, rew = evaluation_environment.get_next_state_and_reward(eval_state,chosen_action)
					eval_rewards.append(rew)
					eval_terminal.append(ter)

				# Compute the evaluation score metrics
				terminal_indices = [i for i, value in enumerate(eval_terminal) if value == True]
				num_terminal_states_reached = len(terminal_indices)
				if num_terminal_states_reached != 0:
					avg_eval_period_length = 0
					for num_terminal_index, terminal_index in enumerate(terminal_indices):
						if num_terminal_index != 0:
							period_duration = terminal_indices[num_terminal_index] - terminal_indices[num_terminal_index - 1]
						else:
							period_duration = terminal_index + 1
							avg_eval_period_length += period_duration
						reward_temp = 0
						for i in range(period_duration):
							reward_temp += eval_rewards[terminal_index - period_duration + 1 + i] * (training_algo.gamma ** i)
						eval_score += reward_temp
					avg_eval_period_length = avg_eval_period_length / num_terminal_states_reached
				else:
					avg_eval_period_length = eval_steps
					for i in range(eval_steps):
						eval_score += eval_rewards[i] * (training_algo.gamma ** i)

				# Log evaluation scores
				results["at_eval"]["evaluation scores"].append((eval_score,num_terminal_states_reached,avg_eval_period_length))

				# If some of the special parameters need to get logged log them
				if algo_special_logs:
					if "special" in results.keys():
						if "at_eval" in results["special"].keys():
							special_results_to_log = training_algo.get_special_logs_at_eval()
							if special_results_to_log != []:
								for label_value in special_results_to_log:
									results["special"]["at_eval"][label_value[0]].append(label_value[1])

				# If necessary, get the value of the current Q function value and bias at chosen state action pairs
				if focus_state_actions:
					q_fct_at_eval_time = training_algo.get_q_fct()
					relevant_q_fct_values_list_all = []
					relevant_bias_list_all = []
					for index in range(len(which_state_actions_focus[0])):
						state = which_state_actions_focus[0][index]
						if state == "start":
							state = evaluation_environment.start_state_num
						relevant_q_fct_values_dict = {key: val for key, val in q_fct_at_eval_time.items() if (key[0] == state and key[1] in estimated_correct_policy[key[0]])}
						relevant_q_fct_values_list = []
						relevant_bias_list = []
						for i in range(len(which_state_actions_focus[1][index])):
							action = which_state_actions_focus[1][index][i]
							if action != "best":
								relevant_q_fct_values_list.append(q_fct_at_eval_time[(state,action)])
								relevant_bias_list.append(q_fct_at_eval_time[(state,action)] - estimated_correct_q_fct[(state,action)])
							elif action == "best" and len(estimated_correct_policy[state]) == 1:
								act_to_regard = estimated_correct_policy[state][0]
								relevant_q_fct_values_list.append(q_fct_at_eval_time[(state,act_to_regard)])
								relevant_bias_list.append(q_fct_at_eval_time[(state,act_to_regard)] - estimated_correct_q_fct[(state,act_to_regard)])
							else:
								# Implementation so that the better Q function value of both will be logged
								relevant_estimated_correct_q_fct_values_dict = {key: val for key, val in estimated_correct_q_fct.items() if (key[0] == state and key[1] in estimated_correct_policy[key[0]])}
								diff_to_correct = [abs(list(relevant_q_fct_values_dict.values())[index] - list(relevant_estimated_correct_q_fct_values_dict.values())[index]) for index in range(len(relevant_q_fct_values_dict.values()))]
								min_diff_index = diff_to_correct.index(min(diff_to_correct))
								act_to_regard = list(relevant_q_fct_values_dict.keys())[min_diff_index][1]
								relevant_q_fct_values_list.append(q_fct_at_eval_time[(state,act_to_regard)])
								relevant_bias_list.append(q_fct_at_eval_time[(state,act_to_regard)] - estimated_correct_q_fct[(state,act_to_regard)])
						relevant_q_fct_values_list_all.append(relevant_q_fct_values_list)
						relevant_bias_list_all.append(relevant_bias_list)
					results["at_eval"]["Q function values at chosen states and actions"].append(relevant_q_fct_values_list_all)
					results["at_eval"]["evaluation biases at chosen states and actions"].append(relevant_bias_list_all)	

				# If necessary, compute the correct action rate for the greedy policy and at the chosen states and actions
				if correct_action_log:
					correct_action_rate_eval_at_chosen = 0
					for i,action in enumerate(eval_policy):
						if action in estimated_correct_policy[i]:
							if correct_action_log_which == "all":
								correct_action_rate_eval_at_chosen += 1
							elif i in correct_action_log_which:
								correct_action_rate_eval_at_chosen += 1
					if correct_action_log_which == "all":
						correct_action_rate_eval_at_chosen = correct_action_rate_eval_at_chosen / len(eval_policy)
					else:
						correct_action_rate_eval_at_chosen = correct_action_rate_eval_at_chosen / len(correct_action_log_which)
					results["at_eval"]["correct action rates at chosen states"].append(correct_action_rate_eval_at_chosen)
					correct_action_rate_eval = correct_actions_played_at_eval / eval_steps
					results["at_eval"]["correct action rates at evaluations"].append(correct_action_rate_eval)


				# If necessary, compute the biases, log the total bias and the chosen biases
				if bias_estimation:
					# Get the Q function at evaluation time
					q_fct_at_eval_time = training_algo.get_q_fct()
					bias_dict = {}
					# Get the bias dictionary
					for state_action in q_fct_at_eval_time.keys():
						if q_fct_at_eval_time[state_action] == 0:
							bias_dict[state_action] = 0
						else:
							bias_dict[state_action] = q_fct_at_eval_time[state_action] - estimated_correct_q_fct[state_action]
					# Get the start state and the optimal arm there to compute the normalization factor
					start_state = evaluation_environment.start_state_num
					optimal_arms_at_start_state = estimated_correct_policy[start_state].copy()
					if len(optimal_arms_at_start_state) == 1:
						optimal_arm_at_start_state = optimal_arms_at_start_state[0]
					else:
						optimal_arm_at_start_state = int(sample_from_dist(training_algo.env.rng,"choice",1,**{"a": optimal_arms_at_start_state, "p": [1/len(optimal_arms_at_start_state) for _ in optimal_arms_at_start_state]})[0])
					if q_fct_at_eval_time[(start_state,optimal_arm_at_start_state)] == 0:
						normalization_factor = 0
					else:
						normalization_factor = q_fct_at_eval_time[(start_state,optimal_arm_at_start_state)] - estimated_correct_q_fct[(start_state,optimal_arm_at_start_state)]
					# Get the normalized bias dictionary
					normalized_bias_dict = {}
					for state_action in bias_dict:
						if bias_dict[state_action] == 0:
							normalized_bias_dict[state_action] = 0
						else:
							normalized_bias_dict[state_action] = bias_dict[state_action] - normalization_factor
					# Get the total bias and log it
					total_bias = 0
					for bias in bias_dict.values():
						total_bias += bias
					results["at_eval"]["total evaluation biases"].append(total_bias)
					# Get the total squared bias and log it
					total_squared_bias = 0
					for bias in bias_dict.values():
						total_squared_bias += bias ** 2
					results["at_eval"]["total squared evaluation biases"].append(total_squared_bias)
					# Get list of biases at the best arms. If there are multiple best arms then get the one where there is the lowest bias except for zero
					rel_bias_list = []
					for sta, best_act_list in enumerate(estimated_correct_policy):
						relevant_bias = 0
						for arms in best_act_list:
							if relevant_bias == 0:
								relevant_bias = bias_dict[(sta,arms)]
							elif bias_dict[(sta,arms)] < relevant_bias and bias_dict[(sta,arms)] != 0:
								relevant_bias = bias_dict[(sta,arms)]
						rel_bias_list.append(relevant_bias)					
					# Get the total bias at best arms and log it
					filtered_total_bias = 0
					for bias in rel_bias_list:
						filtered_total_bias += bias
					results["at_eval"]["total evaluation biases at best arms"].append(filtered_total_bias)
					# Get the total squared bias at best arms and log it
					filtered_total_squared_bias = 0
					for bias in rel_bias_list:
						filtered_total_squared_bias += bias ** 2
					results["at_eval"]["total squared evaluation biases at best arms"].append(filtered_total_squared_bias)
					# Get the total normalized bias and log it
					total_normalized_bias = 0
					for norm_bias in normalized_bias_dict.values():
						total_normalized_bias += norm_bias
					results["at_eval"]["total normalized evaluation biases"].append(total_normalized_bias)
					# Get the total squared normalized bias and log it
					total_squared_normalized_bias = 0
					for norm_bias in normalized_bias_dict.values():
						total_squared_normalized_bias += norm_bias ** 2
					results["at_eval"]["total squared normalized evaluation biases"].append(total_squared_normalized_bias)
					# Get list of normalized biases at best arms. If there are multiple best arms then get the one where there is the lowest bias except for zero
					rel_normalized_bias_list = []
					for sta, best_act_list in enumerate(estimated_correct_policy):
						relevant_normalized_bias = 0
						for arms in best_act_list:
							if relevant_normalized_bias == 0:
								relevant_normalized_bias = normalized_bias_dict[(sta,arms)]
							elif normalized_bias_dict[(sta,arms)] < relevant_normalized_bias and normalized_bias_dict[(sta,arms)] != 0:
								relevant_normalized_bias = normalized_bias_dict[(sta,arms)]
						rel_normalized_bias_list.append(relevant_bias)
					# Get the total normalized bias at best arms and log it
					filtered_total_normalized_bias = 0
					for norm_bias in rel_normalized_bias_list:
						filtered_total_normalized_bias += norm_bias
					results["at_eval"]["total normalized evaluation biases at best arms"].append(filtered_total_normalized_bias)
					# Get the total squared normalized bias at best arms and log it
					filtered_total_squared_normalized_bias = 0
					for norm_bias in rel_normalized_bias_list:
						filtered_total_squared_normalized_bias += norm_bias ** 2
					results["at_eval"]["total squared normalized evaluation biases at best arms"].append(filtered_total_squared_normalized_bias)

				# Evaluation reseeding or advancing the seed according to schedule if necessary
				if len(eval_seed_schedule) > eval:
					eval_seed = eval_seed_schedule[eval]
					if eval_seed == -1:
						eval_seed = generate_random_seed()
					evaluation_environment.rng = np.random.default_rng(seed = eval_seed)
				elif eval_reseeding:
					eval_seed = generate_random_seed()
					evaluation_environment.rng = np.random.default_rng(seed = eval_seed)

			# If our training goal is a certain amount of steps, update the counter after each step
			if training_mode == "steps":
				if progress:
					iterator.update(1)
				counter += 1
				# If the counter reaches the maximum amount of steps, mark the epoch as capped and exit the epoch loop
				if counter > num_steps:
					results["at_epoch"]["capped"].append(1)
					break

			# If the epoch is done, exit the epoch loop and mark it as either capped or not capped
			if epoch_done:
				results["at_epoch"]["capped"].append(0)
				break
			elif (timestep - results["at_epoch"]["epoch starting times"][-1] + 1) >= max_steps_per_epoch and max_steps_per_epoch != -1:
				results["at_epoch"]["capped"].append(1)
				break
			else:
				# If no breaking conditions are fulfilled, update the timestep and continue the epoch training loop
				timestep += 1
		
		# Calculate epoch duration, epoch result
		results_during_epoch = results["at_step"]["rewards"][results["at_epoch"]["epoch starting times"][-1] - 1 : ]
		duration = len(results_during_epoch)
		epoch_reward = 0
		for i in range(duration):
			epoch_reward += results_during_epoch[i] * (training_algo.gamma ** i)

		# Log epoch duration and epoch result
		results["at_epoch"]["epoch durations"].append(duration)
		results["at_epoch"]["epoch results"].append(epoch_reward)

		# If some of the special parameters need to get logged log them
		if algo_special_logs:
			if "special" in results.keys():
				if "at_epoch" in results["special"].keys():
					special_results_to_log = training_algo.get_special_logs_at_epoch()
					if special_results_to_log != []:
						for label_value in special_results_to_log:
							results["special"]["at_epoch"][label_value[0]].append(label_value[1])

		# If necessary, calculate correct action rates and log them
		if correct_action_log:
			correct_actions_during_epoch = results["at_step"]["correct actions"][results["at_epoch"]["epoch starting times"][-1] - 1 : ]
			correct_action_rate = 0
			for i in range(duration):
				correct_action_rate += correct_actions_during_epoch[i] / duration
			results["at_epoch"]["correct action rates"].append(correct_action_rate)

		# If the counter has already reached the maximum amount of steps, exit the loop
		if counter > num_steps:
			break

		if training_mode == "epoch":
			if progress:
				iterator.update(1)
			counter += 1
			# If the counter reaches the maximum amount of steps, exit the loop
			if counter > num_steps:
				break		

		# Epoch reseeding or advancing the seed according to schedule if necessary
		if len(training_seed_schedule) > epoch:
			training_seed = training_seed_schedule[epoch]
			if training_seed == -1:
				training_seed = generate_random_seed()
			training_algo.policy.rng = np.random.default_rng(seed = training_seed)
			training_algo.env.rng = np.random.default_rng(seed = training_seed)
		elif training_reseeding:
			training_seed = generate_random_seed()
			training_algo.policy.rng = np.random.default_rng(seed = training_seed)
			training_algo.env.rng = np.random.default_rng(seed = training_seed)

	# After all training steps are done, obtain the greedy policy and estimated Q function
	final_policy = training_algo.get_greedy_policy()
	final_q_fct = training_algo.get_q_fct()

	if measure_runtime:
		end_time = time.time()
		runtime = end_time - start_time

	return final_policy, final_q_fct, estimated_correct_policy, estimated_correct_q_fct, results, parameters, runtime