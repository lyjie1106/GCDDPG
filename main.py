import argparse
import configparser
import os
import sys

from baseline.ddpg.train import launch
from baseline.common.utils import str_to_int, str_to_float

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
sys.path.append('/home/lujie/GCRL/PycharmProjects/GCRL-baseline')


def get_args():
    parser = argparse.ArgumentParser(prog='Deep Deterministic Policy Gradient',
                                     description='DDPG with different relabeling algorithm')
    parser.add_argument('--config', dest='config_file_path', required=True, help='File path of configuration file')
    args = parser.parse_args()
    return args


def read_config(config_file_path):
    cfg = configparser.ConfigParser()
    cfg.read(config_file_path)
    config_dict = {'Env_type': cfg.get('Env', 'Env_type'), 'Env_name': cfg.get('Env', 'Env_name'),
                   'MAX_EPOCHS': str_to_int(cfg.get('Train', 'MAX_EPOCHS')),
                   'MAX_CYCLES': str_to_int(cfg.get('Train', 'MAX_CYCLES')),
                   'MAX_EPISODES': str_to_int(cfg.get('Train', 'MAX_EPISODES')),
                   'NUM_TRAIN': str_to_int(cfg.get('Train', 'NUM_TRAIN')),
                   'GAMMA': str_to_float(cfg.get('Train', 'GAMMA')), 'LR_A': str_to_float(cfg.get('Train', 'LR_A')),
                   'LR_C': str_to_float(cfg.get('Train', 'LR_C')), 'TAU': str_to_float(cfg.get('Train', 'TAU')),
                   'MEMORY_CAPACITY': str_to_int(cfg.get('ExperienceReplay', 'MEMORY_CAPACITY')),
                   'BATCH_SIZE': str_to_int(cfg.get('ExperienceReplay', 'BATCH_SIZE')),
                   'K_future': str_to_int(cfg.get('ExperienceReplay', 'K_future')),
                   'Sampler': cfg.get('ExperienceReplay', 'Sampler')}
    if config_dict['Sampler'] == 'CHER':
        config_dict['LR_CHER'] = str_to_float(cfg.get('CHER_Setting','LR_CHER'))
        config_dict['LAMDA_0'] = str_to_float(cfg.get('CHER_Setting', 'LAMDA_0'))
        config_dict['FIXED_LAMDA'] = str_to_float(cfg.get('CHER_Setting', 'FIXED_LAMDA'))
        config_dict['SIZE_A'] = str_to_int(cfg.get('CHER_Setting', 'SIZE_A'))
        config_dict['SIZE_k'] = str_to_int(cfg.get('CHER_Setting', 'SIZE_k'))
    elif config_dict['Sampler']=='EBPHER':
        config_dict['G'] = str_to_float(cfg.get('EBP_Setting', 'G'))
        config_dict['M'] = str_to_float(cfg.get('EBP_Setting', 'M'))
        config_dict['Delta_t'] = str_to_float(cfg.get('EBP_Setting', 'Delta_t'))
        config_dict['Weight_potential'] = str_to_float(cfg.get('EBP_Setting', 'Weight_potential'))
        config_dict['Weight_kinetic'] = str_to_float(cfg.get('EBP_Setting', 'Weight_kinetic'))
        config_dict['Max_energy'] = str_to_float(cfg.get('EBP_Setting', 'Max_energy'))

    # verify config
    flag = False
    if config_dict['Env_type'] not in ['minigrid', 'mujoco']:
        print('Error: wrong env type ')
        flag = True
    elif config_dict['Env_type'] == 'minigrid' and config_dict['Env_name'] not in ['ModifiedEmptyRoomEnv-v0']:
        print('Error: {} is not an available {} env'.format(config_dict['Env_name'], config_dict['Env_type']))
        flag = True
    elif config_dict['Env_type'] == 'mujoco' and config_dict['Env_name'] not in ['FetchPickAndPlace-v1', 'FetchPush-v1',
                                                                                 'FetchReach-v1', 'FetchSlide-v1']:
        print('Error: {} is not an available {} env'.format(config_dict['Env_name'], config_dict['Env_type']))
        flag = True
    elif config_dict['GAMMA'] >= 1 or config_dict['GAMMA'] <= 0:
        print('Error: GAMMA value not in (0,1)')
        flag = True
    elif config_dict['LR_A'] < 0 or config_dict['LR_C'] < 0:
        print('Error: Negative learning rate ')
        flag = True
    elif config_dict['TAU'] >= 1 or config_dict['TAU'] <= 0:
        print('Error: TAU value not in (0,1)')
        flag = True
    elif config_dict['Sampler'] not in ['Vanilla','HER', 'CHER', 'EBPHER']:
        print('Error: {} is not an available sampler'.format(config_dict['Sampler']))
        flag = True
    elif config_dict['Sampler'] == 'EBPHER' and config_dict['Env_name'] not in ['FetchPickAndPlace-v1', 'FetchPush-v1',
                                                                                'FetchSlide-v1']:
        print('Error: Sampler EBPHER only for env FetchPickAndPlace-v1,FetchPush-v1,FetchSlide-v1')
        flag = True
    elif config_dict['Sampler'] == 'CHER' and config_dict['BATCH_SIZE']<=config_dict['SIZE_A']:
        print('Error: Batch Size must > SIZE_A')
        flag = True

    if flag:
        sys.exit()
    return config_dict


if __name__ == '__main__':
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['IN_MPI'] = '1'
    args = get_args()
    config = read_config(args.config_file_path)
    launch(config)
