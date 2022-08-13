import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator
import os


if __name__ == '__main__':
    data_dir_path = '../../data'
    dir_name='emptyroom-vanilla'


    iters = list(range(50))
    all_success_rate_list=[]
    for dir_temp in os.listdir('{}/{}'.format(data_dir_path,dir_name)):
        filename = os.listdir('{}/{}/{}/log'.format(data_dir_path,dir_name,dir_temp))[0]
        ea = event_accumulator.EventAccumulator('{}/{}/{}/log/{}'.format(data_dir_path,dir_name,dir_temp,filename))
        ea.Reload()

        success_rate = ea.scalars.Items('success_rate')
        success_rate_list = [i.value for i in success_rate]
        all_success_rate_list.append(success_rate_list)
    all_success_rate_list = np.array(all_success_rate_list)

    avg = np.mean(all_success_rate_list,axis=0)
    std = np.std(all_success_rate_list,axis=0)

    #fig = plt.figure(figsize=(20, 10))
    upper_bound = list(map(lambda x:x[0]+x[1],zip(avg,std)))
    lower_bound = list(map(lambda x: x[0]-x[1], zip(avg, std)))
    plt.plot(iters,avg)
    plt.fill_between(iters,lower_bound,upper_bound,alpha=0.3)
    plt.ylim((0, 1))
    plt.xlim((0, 50))
    plt.show()