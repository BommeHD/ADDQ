#!/bin/bash

python train.py --algo "qrdqn" --env AsterixNoFrameskip-v4 --hyperparams buffer_size:100 n_timesteps:1000 --eval-freq 100 --log-interval 100 --eval-episodes 5 --n-eval-envs 2 --seed 1 -f results/example_project/dev_run/dev_run1 -tb results/tbs/example_project/dev_run/dev_run1 --verbose 1