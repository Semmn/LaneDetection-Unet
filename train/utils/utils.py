import os
import yaml
import regex

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
def load_yaml(path):
    # traverse nested dictionary with the path to the value (hierarchical key set)
    def traverse_leaves_with_keys(d, parent_key=()):
        for key, value in d.items():
            current_path = parent_key + (key, )
            if isinstance(value, dict):
                yield from traverse_leaves_with_keys(value, current_path)
            else:
                yield current_path, value
    
    def get_list_depth(lst):
        if not isinstance(lst, list):
            return 0
        if not lst:
            return 1
        return 1 + max(get_list_depth(item) for item in lst)
    
    def convert_innermost_lists_to_tuples(obj):
        if isinstance(obj, list):
            if not any(isinstance(item, list) for item in obj):
                return tuple(obj)
            
            return [convert_innermost_lists_to_tuples(item) for item in obj]
        return obj
        
    with open(path, 'r') as fr:
        configs = yaml.load(fr, Loader=yaml.Loader)
    
    for path, leaf in traverse_leaves_with_keys(configs):
        root = configs
        # we need to access to the node that precedes the terminal node
        for i in range(len(path)-1):
            root = root[path[i]]
            
        parent_key = path[-1]
        value = root[parent_key]
        
        p = r'\d{0,1}e[+,-]\d{0,5}' # regex expression for finding scientific notation
        p = regex.compile(p)
        # define the rules to filter the yaml value
        if value == 'None': # convert string 'None' to python None
            root[parent_key] = None
        elif type(value)==str and len(p.findall(value)) != 0: # convert string scientific notation to float value
            root[parent_key] = float(value)
        elif isinstance(value, list): # convert innermost lists to tuples
            depth = get_list_depth(value)
            if depth >= 2: # if depth of the list is bigger than 2
                root[parent_key] = convert_innermost_lists_to_tuples(value)
        
    return configs
        
        
        
    
    