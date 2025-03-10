import torch
import logging
import yaml
from train.utils.loss import LaneDICELoss, DICELoss, MaskedPixelReconstructLoss
from train.utils.utils import print_with_r0, log_with_r0, load_yaml
import timm

# this function filters checkpoint['model_state_dict'] key name.
# use this function when the saved checkpoint file has 'module.' prefix in each layer name. (in the case of using torch distributed training, and inferencing on non-distributd environment)
def filter_module(checkpoint):
    # filtering weight name 'module.' on checkpoint file name (if not using torch.distributed training, 'module.' on key string need to be filtered.)
    for key in list(checkpoint['model_state_dict'].keys()):
        if 'module.' in key:
            checkpoint['model_state_dict'][key.replace('module.', '')] = checkpoint['model_state_dict'][key]
            del checkpoint['model_state_dict'][key]
            
    return checkpoint # filtered checkpoint ('module.' will be removed in the key name)

# this function load weight from checkpoint file when there is size mismatch between tensors
# layer mismatch error can be solved by setting strict=True in torch.load_state_dict from model
# but size can not be solved by just setting strict=True. in this case, we can solve this problem by altering checkpoint file
# logger will log any event when size mismatch will happen. (info: model layers are successfully loaded, warning: there is size mismatch or skipping, error: error will be occured as one of allow_size_mismatch=True or strict=True)
# optimizer state, scheduler state will not be loaded in this function.
def load_weight(cp_path, model, strict, allow_size_mismatch, logger, opts):
    checkpoint = torch.load(cp_path, map_location=torch.device('cuda:{}'.format(opts.gpu))) # checkpoint state dictionary
    model_dict = model.state_dict() # model state dictionary
    
    if logger != None:
        for key in checkpoint['model_state_dict']:
            if key in model_dict.keys():
                if model_dict[key].size() != checkpoint['model_state_dict'][key].size():
                    if allow_size_mismatch:
                            logger.warning(f"Size mismatch: Model has {model_dict[key].size()} and checkpoint has {checkpoint['model_state_dict'][key].size()} for key {key}. this layer will be initialized as model initial weight as allow_size_mismatch={allow_size_mismatch}")
                            checkpoint['model_state_dict'][key] = model_dict[key] # swap the original checkpoint weight parameters as model intialized weight parameters
                            # or you can directly remove the key from checkpoint['model_state_dict'] and set the strict to False
                            # checkpoint['model_state_dict'].pop(key)
                            # strict=False
                    else:
                        logger.error(f"Size mismatch: Model has {model_dict[key].size()} and checkpoint has {checkpoint['model_state_dict'][key].size()} for key {key}. error will be occured as allow_size_mismatch={allow_size_mismatch}")
                else:
                    logger.info(f'layer {key} is successfully loaded.')
            else:
                if key not in model_dict.keys():
                    if strict:
                        logger.error(f"Model does not have layer: {key}. as strict={strict} error will be occured!")
                    else:
                        logger.warning(f"Model does not have layer: {key}. as strict={strict} skipping this layer...")
    else: # if logger is not given, just load the weight without logging (other than rank0 process will process this..)
        for key in checkpoint['model_state_dict']:
            if key in model_dict.keys():
                if model_dict[key].size() != checkpoint['model_state_dict'][key].size():
                    if allow_size_mismatch:
                        checkpoint['model_state_dict'][key] = model_dict[key] # swap the original checkpoint weight parameters as model intialized weight parameters
    model.load_state_dict(checkpoint['model_state_dict'], strict=strict) # error stack trace will follow default pytorch stack trace.

# this function load model weight with direct checkpoint that are given as function parameter.
# use this function when filtering on checkpoint['model_state_dict'] is needed. filtered checkpoint file can be given.
# ex) saved checkpoints during pytorch multi-distributed training will have 'module.' prefix in each layer name. if you use checkpoint on non-distributed environment, you need to filter this key name
def load_weight_with_cp(checkpoint, model, strict, allow_size_mismatch, logger):
    model_dict = model.state_dict() # model state dictionary
    
    if logger != None:        
        for key in checkpoint['model_state_dict']:
            if key in model_dict.keys():
                if model_dict[key].size() != checkpoint['model_state_dict'][key].size():
                    if allow_size_mismatch:
                            logger.warning(f"Size mismatch: Model has {model_dict[key].size()} and checkpoint has {checkpoint['model_state_dict'][key].size()} for key {key}. this layer will be initialized as model initial weight as allow_size_mismatch={allow_size_mismatch}")
                            checkpoint['model_state_dict'][key] = model_dict[key] # swap the original checkpoint weight parameters as model intialized weight parameters
                    else:
                        logger.error(f"Size mismatch: Model has {model_dict[key].size()} and checkpoint has {checkpoint['model_state_dict'][key].size()} for key {key}. error will be occured as allow_size_mismatch={allow_size_mismatch}")
                else:
                    logger.info(f'layer {key} is successfully loaded.')
            else:
                if key not in model_dict.keys():
                    if strict:
                        logger.error(f"Model does not have layer: {key}. as strict={strict} error will be occured!")
                    else:
                        logger.warning(f"Model does not have layer: {key}. as strict={strict} skipping this layer...")
    else: # only rank 0 will log the event. other ranks will just load the model into their process.
        for key in checkpoint['model_state_dict']:
            if key in model_dict.keys():
                if model_dict[key].size() != checkpoint['model_state_dict'][key].size():
                    if allow_size_mismatch:
                        checkpoint['model_state_dict'][key] = model_dict[key] # swap the original checkpoint weight parameters as model intialized weight parameters
                    
    model.load_state_dict(checkpoint['model_state_dict'], strict=strict) # error stack trace will follow default pytorch stack trace.



# this function sets optimizer, scheduler, model weight with considering options from argument parser
# if opts.pretraind_weight_path is set, model will be loaded with pretrained weight and optimizer, scheduler will be initialized with default values (start epoch will be 0)
# if opts.load_weight_path is set, model will be loaded with checkpoint file and model weight, optimizer, scheduler will be loaded with checkpoint file (start epoch will be also loaded)
# if both are set, only opts.load_weight_path will be used and opts.pretrained_weight_path will be ignored

# checkpoint dir structure
# torch.save({
#     'epoch': epoch,
#     'model_state_dict' : convnext_unet.state_dict(),
#     'optimizer_state_dict': optimizer.state_dict(),
#     'scheduler_state_dict' : scheduler.state_dict(),
#     'consumed_batch': (epoch+1) * len(train_dataloader) # number of total consumed iteration (number of total consumed batches during training)
# }, f'{opts.save_path}/weights/{currtime}_epoch{epoch+1}.pth') # epochs are saved as 1-index ordering (index 0 means initial state)
def set_weights(opts, model, dataloader, logger):
    
    if opts.load_weight_path != 'None' and opts.pretrained_weight_path != 'None':
        print_with_r0(opts, "Warnings: if opts.load_weight_path are set, opts.pretrained_weight_path will be ignored!")
        log_with_r0(opts, logger, "Warnings: if opts.load_weight_path are set, opts.pretrained_weight_path will be ignored!", logging.WARNING)
    
    loading_flag = False
    if opts.load_weight_path != 'None' or opts.pretrained_weight_path != 'None':
        loading_flag = True
        
        # checkpoint file loadings
        if opts.load_weight_path != 'None':
            load_path = str(opts.load_weight_path)
        elif opts.load_weight_path=='None' and opts.pretrained_weight_path!='None':
            load_path = str(opts.pretrained_weight_path)
        
        # record to the logger
        print_with_r0(opts, "Load path: {}".format(load_path))
        log_with_r0(opts, logger, "load path are set: {}".format(load_path), logging.INFO)
        
        checkpoints = torch.load(load_path, map_location=torch.device('cuda:{}'.format(opts.gpu)))
        print_with_r0(opts, "\nLoaded checkpoint from epoch %d.\n" % (checkpoints['epoch']+1))
        
    with open(opts.train_config) as f:
        train_config = yaml.load(f, Loader=yaml.Loader)
        
    scheduler_type = train_config['train']['scheduler']['scheduler_type']
    loss_type = train_config['train']['loss']['loss_type']
    optimizer_type = train_config['train']['optimizer']['optimizer_type']
    loss_config = train_config['train']['loss'][loss_type]
    scheduler_config = train_config['train']['scheduler'][scheduler_type]
    optimizer_config = train_config['train']['optimizer'][optimizer_type] # this will be implemented later..
    
    if loss_type == 'lane_dice':
        num_cls = loss_config['num_cls']
        criterion = LaneDICELoss(num_cls=num_cls)
    elif loss_type == 'dice':
        weights = loss_config['weights']
        num_cls = loss_config['num_cls']
        criterion = DICELoss(weights=weights, num_cls=num_cls)
    elif loss_type == 'cross_entropy': # pytorch based cross entropy function
        weights = loss_config['weights']
        ignore_index = loss_config['ignore_index']
        reduction = loss_config['reduction']
        label_smoothing = loss_config['label_smoothing']
        criterion = torch.nn.CrossEntropyLoss(weight=weights, ignore_index=ignore_index, reduction=reduction, label_smoothing=label_smoothing)
    elif loss_type == 'mse': # this will be later implemented.
        reduction = loss_config['reduction']
        criterion = torch.nn.MSELoss(reduction=reduction)
    elif loss_type =='masked_mse':
        criterion = MaskedPixelReconstructLoss()
    else:
        raise Exception(f"Not Supported Loss function: {loss_type} Currently only 'lane_dice', 'dice', 'cross_entropy', 'mse', 'masked_mse' are supported")
    
    if optimizer_type == 'adamw' or optimizer_type == 'adam':
        lr = optimizer_config['lr']
        betas = optimizer_config['betas']
        betas = (betas[0], betas[1])
        eps = optimizer_config['eps']
        weight_decay = optimizer_config['weight_decay']
        amsgrad = optimizer_config['amsgrad']
        maximize = optimizer_config['maximize']
        foreach = optimizer_config['foreach']
        capturable = optimizer_config['capturable']
        differentiable = optimizer_config['differentiable']
        fused = optimizer_config['fused']
        if optimizer_type == 'adamw':
            optimizer = torch.optim.AdamW(model.parameters(),lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                                          amsgrad=amsgrad, maximize=maximize, foreach=foreach, capturable=capturable, differentiable=differentiable,
                                          fused=fused)
        else:
            optimizer = torch.optim.Adam(model.parameters(),lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                                          amsgrad=amsgrad, maximize=maximize, foreach=foreach, capturable=capturable, differentiable=differentiable,
                                          fused=fused)
    elif optimizer_type == 'sgd':
        lr = optimizer_config['lr']
        momentum = optimizer_config['momentum']
        dampening = optimizer_config['dampening']
        weight_decay = optimizer_config['weight_decay']
        nesterov = optimizer_config['nesterov']
        maximize = optimizer_config['maximize']
        foreach = optimizer_config['foreach']
        differentiable = optimizer_config['differentiable']
        fused = optimizer_config['fused']
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, dampening=dampening, weight_decay=weight_decay,
                                    nesterov=nesterov, maximize=maximize, foreach=foreach, differentiable=differentiable, 
                                    fused=fused)
    elif optimizer_type == 'rmsprop':
        lr = optimizer_config['lr']
        alpha = optimizer_config['alpha']
        eps = optimizer_config['eps']
        weight_decay = optimizer_config['weight_decay']
        momentum = optimizer_config['momentum']
        centered = optimizer_config['centered']
        capturable = optimizer_config['capturable']
        foreach = optimizer_config['foreach']
        maximize = optimizer_config['maximize']
        differentiable = optimizer_config['differentiable']
        optimizer = torch.optim.RMSprop(model.parameters(), lr=lr, alpha=alpha, eps=eps, weight_decay=weight_decay,
                                        momentum=momentum, centered=centered, capturable=capturable, foreach=foreach,
                                        maximize=maximize, differentiable=differentiable)
    else:
        raise Exception(f"Not Supported Optimizer: {optimizer_type} Currently only 'adamw', 'adam', 'sgd', 'rmsprop' are supported")
    
    if scheduler_type == 'cosine_warmup_restart': # returns timm based cosine lr scheduler
        lr_min = scheduler_config['lr_min'] # minimum learning rate for cosine decay restart
        decay = scheduler_config['decay']   # decay rate for each cycle
        mul = scheduler_config['cycle_mul'] # multiplication for cycle duration
        limit = scheduler_config['cycle_limit'] # maximum number of cycles
        warmup = scheduler_config['warmup_lr']  # number of warmup epochs
        warmup_init = scheduler_config['warmup_lr_init'] # warmup initial learning rate
        step_size = scheduler_config['step_size'] # step size of cycle
        scheduler = timm.scheduler.cosine_lr.CosineLRScheduler(optimizer, t_initial=step_size, lr_min=lr_min, cycle_mul=mul,
                                                               cycle_decay=decay, cycle_limit=limit, warmup_t=warmup, warmup_lr_init=warmup_init)
        end_epoch = scheduler_config['epoch']
    elif scheduler_type == 'one_cycle':
        if loading_flag and opts.load_weight_path != 'None':
            last_epoch = checkpoints['consumed_batch']
        else:
            last_epoch=-1 # using pre-training or start from scratch
            
        lr=scheduler_config['max_lr'] # upper learning rate boundaries in the cycle for each parameter group
        epochs=scheduler_config['epoch'] # number of epochs to train for
        steps_per_epoch = len(dataloader) # number of steps per epoch
        pct = scheduler_config['pct'] # the percentage of cycle
        anneal_strategy = scheduler_config['anneal_strategy'] # 'cos' or 'linear' -> 'cos' for cosine annealing, 'linear' for linear annealing
        div = scheduler_config['div'] # divider factor for determining intial learning rate (initial_lr = max_lr/div)
        f_div = scheduler_config['f_div'] # minimum learning rate via min_lr = initial_lr/final_div_factor
        three_phase = scheduler_config['three_phase'] # if set to true, use a thired phase of the schedule to annihilate the learning rate according to 'final_div_factor' instead of modifying the second phase
        
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, epochs=epochs, steps_per_epoch=steps_per_epoch, pct_start=pct,
                                                        anneal_strategy=anneal_strategy, div_factor=div, final_div_factor=f_div, three_phase=three_phase, last_epoch=last_epoch)
        end_epoch = scheduler_config['epoch']
    else:
        raise Exception(f"Not Supported Scheduler: {scheduler_type} Currently learning rate scheduler must be one of 'cosine_warmup_restart', 'one_cycle'")
    
    if loading_flag:
        if opts.rank == 0: # only rank 0 will access to the logger
            logger = logger
        else:
            logger = None
        
        if opts.load_weight_path!='None': # load checkpoint from training must not allow size mismatch or layer skipping
            load_weight_with_cp(checkpoints, model, strict=True, allow_size_mismatch=False, logger=logger)
            optimizer.load_state_dict(checkpoints['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoints['scheduler_state_dict'])
        elif opts.load_weight_path=='None' and opts.pretrained_weight_path!='None': # load checkpoint from pre-training must allow size mismatch and layer skipping
            load_weight_with_cp(checkpoints, model, strict=True, allow_size_mismatch=True, logger=logger)
            start_epoch=0 # pretraiend with loading will be start from 0 epoch
    else:
        start_epoch=0
    
    return model, criterion, optimizer, scheduler, start_epoch, end_epoch, scheduler_type

