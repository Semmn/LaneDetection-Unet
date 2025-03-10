import glob

# this functions constructs the culane data list to train and validate
# it returns the image and mask path for train, validation, test dataset and also lane information
# train_y_lane, val_y_lane is the list that contains the information about lane existence (0: no lane, 1: lane)
def read_culane_segdata(basepath):
    
    with open(basepath+'/list/train_gt.txt', 'r') as fr:
        train_list = fr.readlines()
    with open(basepath+'/list/val_gt.txt', 'r') as fr:
        val_list = fr.readlines()
    with open(basepath+'/list/test.txt', 'r') as fr:
        test_list = fr.readlines()


    train_x, train_y, train_y_lane = [], [], [] # train_y_lane denotes whether lane exist or not.
    val_x, val_y, val_y_lane = [], [], []
    test_x = []

    for path in train_list:
        tokens = path.split(' ')
        train_x.append(basepath+tokens[0])
        train_y.append(basepath+tokens[1])
        train_y_lane.append([int(tokens[2]), int(tokens[3]), int(tokens[4]), int(tokens[5].strip())])

    for path in val_list:
        tokens = path.split(' ')
        val_x.append(basepath+tokens[0])
        val_y.append(basepath+tokens[1])
        val_y_lane.append([int(tokens[2]), int(tokens[3]), int(tokens[4]), int(tokens[5].strip())])

    for path in test_list:
        test_x.append(basepath+path.strip())
        
    return (train_x, train_y), (val_x, val_y), (test_x, None), (train_y_lane, val_y_lane)


# this functions reads segmentation data from multi-dataset (currently CULANE, TUSIMPLE, LLAMAS, BDD100K are considered)
def read_multi_segdata(path_dict):
    
    CULANE_PATH=path_dict['CULANE']
    TUSIMPLE_PATH=path_dict['TUSIMPLE']
    LLAMAS_PATH=path_dict['LLAMAS']
    BDD100K_PATH=path_dict['BDD100K']
    
    with open(CULANE_PATH+'/list/train_gt.txt', 'r') as fr:
        cu_train_info = fr.readlines()
    with open(CULANE_PATH+'/list/val_gt.txt', 'r') as fr:
        cu_val_info = fr.readlines()
    with open(CULANE_PATH + '/list/test.txt', 'r') as fr:
        cu_test_info = fr.readlines()

    cu_train_x = []
    cu_train_y = []
    cu_val_x = []
    cu_val_y = []
    cu_test_x = []

    for path in cu_train_info:
        tokens = path.split(' ')
        cu_train_x.append(CULANE_PATH+tokens[0])
        cu_train_y.append(CULANE_PATH+tokens[1])
        
    for path in cu_val_info:
        tokens = path.split(' ')
        cu_val_x.append(CULANE_PATH+tokens[0]) # 9675
        cu_val_y.append(CULANE_PATH+tokens[1])

    for path in cu_test_info:
        tokens = path.split(' ')
        cu_test_x.append(CULANE_PATH+tokens[0])

    with open(TUSIMPLE_PATH+'/train_set/seg_label/list/train_val_gt.txt', 'r') as fr:
        tu_trainval_info = fr.readlines()
    with open(TUSIMPLE_PATH+'/train_set/seg_label/list/test_gt.txt', 'r') as fr:
        tu_test_info = fr.readlines()

    tu_trainval_x = []
    tu_trainval_y = []
    tu_test_x = []
    tu_test_y = []

    for path in tu_trainval_info:
        tokens = path.split(' ')
        tu_trainval_x.append(TUSIMPLE_PATH+ '/train_set' + tokens[0])
        tu_trainval_y.append(TUSIMPLE_PATH+ '/train_set' + tokens[1])

    for path in tu_test_info:
        tokens = path.split(' ')
        tu_test_x.append(TUSIMPLE_PATH+ '/test_set' + tokens[0]) # images must be the images in test set.
        tu_test_y.append(TUSIMPLE_PATH+ '/train_set' + tokens[1])  # segmentation labels are in the /train_set folder.

    llamas_train_x = glob.glob(LLAMAS_PATH+'/color_images/train/*/*.png')
    llamas_val_x = glob.glob(LLAMAS_PATH + '/color_images/valid/*/*.png')
    llamas_test_x = glob.glob(LLAMAS_PATH + '/color_images/test/*/*.png')

    llamas_train_y = glob.glob(LLAMAS_PATH+'/labels/train/*/*.png')
    llamas_val_y = glob.glob(LLAMAS_PATH+ '/labels/valid/*/*.png')

    llamas_train_x = sorted(llamas_train_x)
    llamas_val_x = sorted(llamas_val_x)
    llamas_test_x = sorted(llamas_test_x)
    
    llamas_train_y = sorted(llamas_train_y)
    llamas_val_y = sorted(llamas_val_y)
    
    bdd_train_x = glob.glob(BDD100K_PATH+'/images/100k/train/*.jpg')
    bdd_test_x = glob.glob(BDD100K_PATH+ '/images/100k/test/*.jpg')
    bdd_val_x = glob.glob(BDD100K_PATH+'/images/100k/val/*.jpg')

    # bdd_train_y = glob.glob(BDD100K_PATH+'/labels/lane/masks/train/*.png') # (H, W) format
    # bdd_val_y = glob.glob(BDD100K_PATH+'/labels/lane/masks/val/*.png')

    bdd_train_y = glob.glob(BDD100K_PATH+'/labels/lane/colormaps/train/*.png') # (H, W, C) format -> visualizable
    bdd_val_y = glob.glob(BDD100K_PATH+'/labels/lane/colormaps/val/*.png')

    bdd_train_x = sorted(bdd_train_x)
    bdd_val_x = sorted(bdd_val_x)
    bdd_test_x = sorted(bdd_test_x)

    bdd_train_y = sorted(bdd_train_y)
    bdd_val_y = sorted(bdd_val_y)

    return {'CULANE': (cu_train_x, cu_train_y, cu_val_x, cu_val_y, cu_test_x), 'TUSIMPLE': (tu_trainval_x, tu_trainval_y, tu_test_x, tu_test_y),
        'LLAMAS': (llamas_train_x, llamas_train_y, llamas_val_x, llamas_val_y, llamas_test_x), 'BDD100K': (bdd_train_x, bdd_train_y, bdd_val_x, bdd_val_y, bdd_test_x)}



