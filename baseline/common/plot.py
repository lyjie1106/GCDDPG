import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator
import os

def read_data(dir_name):
    data_dir_path = '../../data'
    all_success_rate_list = []
    for dir_temp in os.listdir('{}/{}'.format(data_dir_path, dir_name)):
        filename = os.listdir('{}/{}/{}/log'.format(data_dir_path, dir_name, dir_temp))[0]
        ea = event_accumulator.EventAccumulator('{}/{}/{}/log/{}'.format(data_dir_path, dir_name, dir_temp, filename))
        ea.Reload()

        success_rate = ea.scalars.Items('success_rate')
        success_rate_list = [i.value for i in success_rate]
        all_success_rate_list.append(success_rate_list)
    all_success_rate_list = np.array(all_success_rate_list)

    avg = np.mean(all_success_rate_list, axis=0)
    std = np.std(all_success_rate_list, axis=0)
    upper_bound = list(map(lambda x: x[0] + x[1], zip(avg, std)))
    lower_bound = list(map(lambda x: x[0] - x[1], zip(avg, std)))
    return avg,upper_bound,lower_bound

if __name__ == '__main__':
    iters = list(range(50))
    labels=['cher','her','vanilla']

    avg0, upper_bound0, lower_bound0 = read_data('emptyroom-cher')
    avg1, upper_bound1, lower_bound1 = read_data('emptyroom-her')
    avg2, upper_bound2, lower_bound2 = read_data('emptyroom-vanilla')

    avg_list=np.array([avg0,avg1,avg2])
    upper_bound_list=np.array([upper_bound0,upper_bound1,upper_bound2])
    lower_bound_list=np.array([lower_bound0,lower_bound1,lower_bound2])
    color = ['orange','blue','green']


    for i in range(3):
        plt.plot(iters,avg_list[i],color=color[i],label=labels[i])
        plt.fill_between(iters,lower_bound_list[i],upper_bound_list[i],color=color[i],alpha=0.05)
    plt.ylim((0, 1))
    plt.xlim((0, 50))
    plt.legend()
    plt.show()