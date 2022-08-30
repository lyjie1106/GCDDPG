# Deep Deterministic Policy Gradient

Implementation of Deep Deterministic Policy Gradient with Hindsight Experience Replay(HER), Curriculum Hindsight
Experience Replay(CHER),Energy-Based Hindsight Experience Prioritization(EBP)
The Algorithm is for Gym-Robotics' Fetch environments and a customised gym-simple-minigrid environment.

## Usage:

```shell
# train
mpirun -np $(nproc) python3 -u main.py --config ./path/to/config/file
# tensorboard data
tensorboard --logdir=./path/to/the/log/folder --port 8123