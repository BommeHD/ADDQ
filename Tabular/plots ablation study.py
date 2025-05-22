import plots

plots.selected_plots_one_experiment(
        plottitle = "Ablation study",
        subplottitles=["Summed total biases","Summed total squared biases","Summed total biases\nat best actions","Summed total squared biases\nat best actions","empty","State 1 Action Up", "State 1 Action Right", "State 1 Action Down","State 1 Action Left","empty","State 6 Action Up", "State 6 Action Right", "State 6 Action Down","State 6 Action Left"],
        plotlist = ["Mean bias metrics at evals","empty","Mean Q function values at chosen at evals","empty","Mean Q function values at chosen at evals"],
        correct_action_log = True,
        focus_state_actions = True,
        bias_estimation = True,
        algo_special_logs = True,
        squared_normalized_best_arms_list = [[(False,False,False),(True,False,False),(False,False,True),(True,False,True)]],
        which = [[(1,0),(1,1),(1,2),(1,3)],[(6,0),(6,1),(6,2),(6,3)]],
        max_steps_per_epoch = -1,
        index = [[[],[],[]]],
        input_path = "plots/.results_to_plot/ablation_study_GridWorld.pkl",
        plot_folder = "plots",
        project_name = "paper",
        individual_figsize = (4,4),
        num_rows = 5,
        save_string = "Plot 3")