import logging
import os


# this functions creates directories if not exist. when directories are created, it prints out the message
def checkdir(path):
    if not os.path.exists(path):
        print("creating directories: ", path)
        os.makedirs(path)

# this function prepares logger for monitoring training, validating, common process (record options used for training, model weight loading etc...)
# train_logger: will record training loss, metrics, etc...
# val_logger: will record validation loss, metrics, etc...
# process_logger: will record process information such as model weight loading, options used for training, etc...
def set_logger_for_training(opts):
    
    train_logger = logging.getLogger('train')
    val_logger = logging.getLogger('val')
    process_logger = logging.getLogger('process')

    train_file_handler = logging.FileHandler(opts.save_path+'/train.log')
    val_file_handler = logging.FileHandler(opts.save_path+'/val.log')
    process_file_handler = logging.FileHandler(opts.save_path+'/process.log')
    
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s', datefmt="%Y-%m-%d %H:%M:%S")
    train_file_handler.setFormatter(formatter)
    val_file_handler.setFormatter(formatter)
    process_file_handler.setFormatter(formatter)

    train_logger.setLevel(logging.INFO)
    val_logger.setLevel(logging.INFO)
    process_logger.setLevel(logging.INFO)

    train_logger.addHandler(train_file_handler)
    val_logger.addHandler(val_file_handler)
    process_logger.addHandler(process_file_handler)
    
    return train_logger, val_logger, process_logger