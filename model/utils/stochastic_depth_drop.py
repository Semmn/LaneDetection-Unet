# Author: Yang Seung Yu
# Stochastic depth drop probability list generator
# return [[],[],[],[]] for drop probability list and drop mode list. outer list is for stage, inner list is for block

# create linearly increasing depth drop probability list, mode
def create_linear_p(block_nlist:list, dp_mode:str, last_p:float):
    
    droprate = []  # probability of stochastic drop depth for all blocks in all stages
    drop_mode = [] # mode of stochastic drop depth for all blocks in all stages
    
    l_count = 0
    num_of_l = sum(block_nlist)
    
    for s_index, num_block in enumerate(block_nlist,0): # starts from stage 0
        p_list = []
        mode_list = []
        for b_index in range(num_block): # starts from block 0
            p = (l_count/num_of_l) * (1-last_p)
            p_list.append(p)
            mode_list.append(dp_mode)
            l_count+=1
        droprate.append(p_list)
        drop_mode.append(mode_list)
    return droprate, drop_mode

# create constant depth drop probability, mode
def create_uniform_p(block_nlist:list, dp_mode:str, uniform_p:int):
    droprate = []
    drop_mode = []
    
    for s_index, num_block in enumerate(block_nlist, 0):
        p_list = []
        mode_list = []
        for b_index in range(num_block):
            p_list.append(uniform_p)
            mode_list.append(dp_mode)
        droprate.append(p_list)
        drop_mode.append(mode_list)

    return droprate, drop_mode