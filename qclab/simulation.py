import numpy as np
from argparse import Namespace


class Trajectory:
    def __init__(self):
        self.data_dic = {}  # dictionary to store data
    def add_to_dic(self, name, data):
        self.data_dic.__setitem__(name, data)
        return
    def new_observable(self, name, shape, type):
        self.data_dic[name] = np.zeros(shape, dtype=type)
        return
    def add_observable_dict(self, ind, dic):
        for key in dic.keys():
            if key in self.data_dic.keys():
                self.data_dic[key][ind] += dic[key]
        return
    
class Data:
    def __init__(self):
        self.data_dic = {}
        self.seed_list = np.array([], dtype=int)
    def add_data(self, traj_obj):  # adds data from a traj_obj
        for key, val in traj_obj.data_dic.items():
            if key in self.data_dic:
                self.data_dic[key] = self.data_dic[key] + val
            else:
                self.data_dic[key] = val
        self.seed_list = np.append(self.seed_list, traj_obj.seed)
        return
    def sum_data(self, data_obj):  # adds data from a data_obj
        for key, val in data_obj.data_dic.items():
            if key in self.data_dic:
                self.data_dic[key] = self.data_dic[key] + val
            else:
                self.data_dic[key] = val
        self.seed_list = np.concatenate((self.seed_list, data_obj.seed_list))
        return