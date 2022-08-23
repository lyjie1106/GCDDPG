import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator
import os
type='time'
env = 'gamma'


def read_data(dir_name):
    data_dir_path = '../../data/experiment-5/'+env
    all_success_rate_list = []
    for dir_temp in os.listdir('{}/{}'.format(data_dir_path, dir_name)):
        filename = os.listdir('{}/{}/{}/log'.format(data_dir_path, dir_name, dir_temp))[0]
        ea = event_accumulator.EventAccumulator('{}/{}/{}/log/{}'.format(data_dir_path, dir_name, dir_temp, filename))
        ea.Reload()

        success_rate = ea.scalars.Items(type)
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

    labels=['gamma-0.9','gamma-0.95','gamma-0.99']
    #legend = ['kinetic=0,potential=1','kinetic=1,potential=1','kinetic=1,potential=0','vanilla']

    avg_list,upper_bound_list,lower_bound_list = [],[],[]
    for i in labels:
        avg, upper_bound, lower_bound = read_data(i)
        avg_list.append(avg)
        upper_bound_list.append(upper_bound)
        lower_bound_list.append(lower_bound)
    avg_list=np.array(avg_list)
    upper_bound_list=np.array(upper_bound_list)
    lower_bound_list=np.array(lower_bound_list)


    color = ['orange','blue','green','red']


    for i in range(len(labels)):
        plt.plot(iters,avg_list[i],color=color[i],label=labels[i])
        plt.fill_between(iters,lower_bound_list[i],upper_bound_list[i],color=color[i],alpha=0.1)
    #plt.ylim((0, 1))
    plt.xlim((0, 50))
    plt.legend()
    plt.savefig(env+'.png')
    plt.show()