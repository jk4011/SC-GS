#!/bin/bash
SECONDS=0
set -e        # exit when error
set -o xtrace # print command

wandb_group_name=10k_init0.5

#                           GPU object                      idx_from idx_to cam_idx wandb_group_name
bash train_dfa.sh 6   "beagle_dog(s1)"            520      525    16      ${wandb_group_name} &
bash train_dfa.sh 6   "beagle_dog(s1_24fps)"      190      215    32      ${wandb_group_name} &
bash train_dfa.sh 6   "wolf(Howling)"             10       60     24      ${wandb_group_name} &
bash train_dfa.sh 6   "bear(walk)"                110      140    16      ${wandb_group_name} &
bash train_dfa.sh 7   "cat(run)"                  25       30     32      ${wandb_group_name} &

# bash train_dfa.sh 7   "cat(walk_final)"           10       20     32      ${wandb_group_name} &
# bash train_dfa.sh 5   "wolf(Run)"                 20       25     16      ${wandb_group_name} 
# bash train_dfa.sh 5   "cat(walkprogressive_noz)"  25       30     32      ${wandb_group_name} 
bash train_dfa.sh 6   "duck(eat_grass)"           5        15     32      ${wandb_group_name} &

bash train_dfa.sh 7   "duck(swim)"                145      160    16      ${wandb_group_name} &
# bash train_dfa.sh 0   "whiteTiger(roaringwalk)"   15       25     32      ${wandb_group_name} 
# bash train_dfa.sh 0   "fox(attitude)"             95       145    24      ${wandb_group_name} 
# bash train_dfa.sh 0   "wolf(Walk)"                85       95     16      ${wandb_group_name} &

# bash train_dfa.sh 1   "fox(walk)"                 70       75     24      ${wandb_group_name} 
# bash train_dfa.sh 1   "panda(walk)"               15       25     32      ${wandb_group_name} 
# bash train_dfa.sh 1   "lion(Walk)"                30       35     32      ${wandb_group_name} 
# bash train_dfa.sh 1   "panda(acting)"             95       100    32      ${wandb_group_name} &

# bash train_dfa.sh 7   "panda(run)"                5        10     32      ${wandb_group_name} &
# bash train_dfa.sh 2   "lion(Run)"                 50       55     24      ${wandb_group_name} 
bash train_dfa.sh 7   "duck(walk)"                200      230    16      ${wandb_group_name} &
# bash train_dfa.sh 2   "whiteTiger(run)"           25       70     32      ${wandb_group_name} &

# bash train_dfa.sh 3   "wolf(Damage)"              0        110    32      ${wandb_group_name} 
# bash train_dfa.sh 3   "cat(walksniff)"            70       150    32      ${wandb_group_name} 
# bash train_dfa.sh 3   "bear(run)"                 0        2      16      ${wandb_group_name} 
# bash train_dfa.sh 3   "fox(run)"                  25       30     32      ${wandb_group_name} &

wait