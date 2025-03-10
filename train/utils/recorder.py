import numpy as np
import torch
from train.utils.plotting import plot_figure
import matplotlib.pyplot as plt

# recorder class for registering metrics and visualize the result
# recorder can hold loss, accuracy (IOU or dice score) and update the result by itself
class Recorder():
    def __init__(self):
        self._registry = {} # key and value collection
        self._statistic = {} # statistics are collected only on the _registry pull
        self._registry_scalar = {} # temporary storage for key and value (aggregating loss and metrics through each iterations)
    
    # add key and value for metrics
    def register(self, key_list):
        for i, key in enumerate(key_list, 0):
            if key in self._registry.keys():
                print("keys are already in the registry")
            self._registry[key] = [] # add key and initialize the value as empty list
    
    def register_scalar(self, key_list):
        for i, key in enumerate(key_list, 0):
            if key in self._registry_scalar.keys():
                print("Keys are already in the registry scalar")
            else:
                self._registry_scalar[key] = 0.0 # initialize as scalar value (0)
    
    def append(self, key, value):
        if key not in self._registry.keys():
            raise Exception(f"given key: {key} not in the registry!")
        self._registry[key].append(value)
    
    def add(self, key, value):
        if key not in self._registry_scalar.keys():
            raise Exception(f"given key: {key} not in the registry_scalar!")
        self._registry_scalar[key] += value
        
    def append_statistic(self, key, value):
        if key not in self._statistic.keys():
            self._statistic[key] = [] # if key not registered in statistic, initialize with empty list([])
        self._statistic[key].append(value)
    
    # initialize the corresponding value in _registry as empty list ([])
    def initialize(self, key_list):
        for key in key_list:
            if key not in self._registry.keys():
                raise Exception(f"given key: {key} not in the registry!")
            self._registry[key] = []
    
    # initialize the corresponding value in _registry_scalar as 0
    def initialize_scalar(self, key_list):
        for key in key_list:
            if key not in self._registry_scalar.keys():
                raise Exception(f"given key: {key} not in the registry_scalar!")
            self._registry_scalar[key] = 0.0
        
    # initialize the collected statistic
    def initialize_statistic(self, key_list:list):
        for key in key_list:
            if key not in self._statistic.keys():
                raise Exception(f"given key: {key} not in the statistic!")
            self._statistic[key] = []
            
    # collect statistic upon given key list from self._registry
    # stat: 'avg' for average, 'std': for standard deviation
    def collect_statistic(self, key_list:list, stat='avg'):
        for key in key_list:
            value = self._registry[key]
            
            if key not in self._statistic.keys():
                self._statistic[key] = [] # if key not in the _statistic pull, initialize with the empty list
            
            if stat=='avg':
                self._statistic[key].append(np.mean(value))
            else:
                raise Exception("Not Implemented Yet!")
    
    # getter for self._registry
    def get(self, key):
        if key not in self._registry.keys():
            raise Exception(f"given key: {key} not in the registry!")
        return self._registry[key]

    # getter for self._registry_scalar
    def get_scalar(self, key):
        if key not in self._registry_scalar.keys():
            raise Exception(f"given key: {key} not in the registry_scalar!")
        return self._registry_scalar[key]
    
    # getter for self._statistic
    def get_statistic(self, key):
        if key not in self._statistic.keys():
            raise Exception(f"given key: {key} not in the statistic!")
        return self._statistic[key]
          
    # save plots for visualize the result
    def save_plots(self, key_list:list, title_list:list, x_ticks_list:list, x_label_list:list, y_label_list:list, save_path_list:list, fname_list:list):
        try:
            for i, key in enumerate(key_list, 0):
                if type(key)==tuple or type(key)==list:
                    
                    data_list = []
                    data_label_list = []
                    for k in key: # each key is tuple
                        data_list.append(self._statistic[k])
                        data_label_list.append(k)
                    
                    plot_figure(figsize=(12, 8), title=title_list[i], data_list=data_list, 
                                data_label_list=data_label_list, x_ticks=x_ticks_list[i], x_label=x_label_list[i], 
                                y_label=y_label_list[i], save_path=save_path_list[i], filename=fname_list[i])
                else:
                    # raise Exception("Not Implemented Yet!")
                    print("Skipping plotting: each key in key_list must be given as tuple or list")
        except:
            print("Error occured...: Skipping plotting")