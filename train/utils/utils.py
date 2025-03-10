import os
import yaml
import time

# copy files from source to destination (that are given in files_list as (src, dest) tuple)
# files_list is the list that contain (src, dest) tuple
def copy_files(files_list):
    for file in files_list:
        os.system(f'cp {file[0]} {file[1]}')
        
# level: "debug", "info", "warning", "error", "critical"
def log_with_r0(opts, logger, message, level):
    if opts.rank == 0:
        logger.log(level, message)

# print message with rank 0 processor
def print_with_r0(opts, message):
    if opts.rank == 0:
        print(message)

# filter yaml file to compatible with python format
# 1. convert string None to python None
# 2. convert string exponential notation to float ('1e-8' -> 1e-8)
# 3. convert most inner list to tuple
def load_yaml(path):
    def traverse_dict(input_dict):
        # start_t = time.time()
        
        # we will use FIFO style search using queue
        queue_k = []
        queue_v = []
        tree_keys = input_dict.keys()
        search_flag = True # at first, we need to search what keys they have (search root keys)
        sub_dict = input_dict # sub_dict is intialized as input_dict

        while True:
            if search_flag: # if pass then, continue to next iteration
                for tree_key in tree_keys:
                    queue_k.insert(0, tree_key)
                    queue_v.insert(0, sub_dict[tree_key])

            if not isinstance(queue_v[0], dict):
                leaf = queue_v.pop(0)
                if leaf == 'None': # 1. if leaf element is string 'None' we convert it to python None
                    leaf = None
                elif isinstance(leaf, str): # 2. if leaf element is string, we try to convert to float format if possible (convert '1e-8' to float 1e-8)
                    try:
                        leaf = float(leaf)
                    except:
                        pass
                # print("leaf value: ", leaf) # no more depth search, left node
                key = queue_k.pop(0)
                sub_dict[key] = leaf
                search_flag=False
            else:
                key = queue_k.pop(0)
                sub_dict = queue_v.pop(0)
                tree_keys = sub_dict.keys()
                search_flag=True # more depth search is needed

            if len(queue_v)==0 and not search_flag: # consuming all the element in queue means we search all the trees
                break
            
        # end_t = time.time()
        # print("Elasped time: ", end_t-start_t)
            
    with open(path, 'r') as fr:
        configs = yaml.load(fr, Loader=yaml.Loader)
    
    traverse_dict(configs) # traverse the dictionary and filter the contents
    return configs
        
        
        
    
    