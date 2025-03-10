import matplotlib.pyplot as plt
from train.utils.logger import checkdir

def plot_figure(figsize, title, data_list, data_label_list, x_ticks, x_label, y_label, save_path, filename):
    
    checkdir(save_path) # check if save path exists, if not, create directories
    
    if len(data_list) != len(data_label_list):
        raise ValueError("data_list and data_label_list should have same length!")
    
    plt.figure(figsize=figsize)
    plt.title(title)
    
    for i, data in enumerate(data_list, 0):
        plt.plot(range(len(data)), data, label=data_label_list[i])

    maxlen = max([len(data) for data in data_list])
    
    plt.xticks(range(0, maxlen, x_ticks))
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.savefig(save_path+f'/{filename}.png')
    
    plt.close('all') # memory saving
