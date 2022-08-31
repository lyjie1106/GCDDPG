# Deep Deterministic Policy Gradient

Implementation of Deep Deterministic Policy Gradient with Hindsight Experience Replay(HER), Curriculum Hindsight
Experience Replay(CHER),Energy-Based Hindsight Experience Prioritization(EBP)
The Algorithm is for Gym-Robotics' Fetch environments and a customised gym-simple-minigrid environment.

## Requirements:
- Python3
- PyTorch
- CUDA
- mpi4py == 3.0.3
- gym == 0.21.0
- gym-minigrid == 1.0.3
- gym-simple-minigrid == 2.0.0
- mujoco_py == 2.1.2.14
- numpy
- matplotlib

## Usage:

```shell
# train
mpirun -np $(nproc) python3 -u main.py --config ./path/to/config/file
Example: mpirun -np $(nproc) python3 -u main.py --config ./baseline/config/test.config
# tensorboard data read
tensorboard --logdir=./path/to/the/log/folder --port 8123
Example: tensorboard --logdir=./data/experiment-1/FetchPickAndPlace/HER/08-16_03:13-FetchPickAndPlace-v1/log --port 8123
# play
python3 play.py --model ./path/to/model/folder
Example: python3 play.py --model ./data/experiment-1/FetchPickAndPlace/HER/08-16_03:13-FetchPickAndPlace-v1