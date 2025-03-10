import os
import torch
from datetime import timedelta

# this functions blocks the print when it is not in the master process
def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """    
    import builtins as __builtin__
    builtin_print = __builtin__.print
    
    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)
    
    __builtin__.print = print

# distributed training initialization function
def init_for_distributed(opts):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        opts.rank = int(os.environ['RANK'])
        opts.world_size = int(os.environ['WORLD_SIZE'])
        opts.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        opts.rank = int(os.environ['SLURM_PROCID'])
        opts.gpu = opts.rank % torch.cuda.device_count()
    else:
        print('Not using distributed training')
        opts.distributed = False
        return
    
    torch.cuda.set_device(opts.gpu)
    opts.dist_backend = 'nccl'
    
    os.environ['TORCH_NCCL_BLOCKING_WAIT'] = '0' # not to enforce timeout
    
    # 2. init_process_group
    torch.distributed.init_process_group(backend=opts.dist_backend, timeout=timedelta(seconds=72000000), init_method='env://', world_size=opts.world_size, rank=opts.rank)
    
    torch.distributed.barrier()
    
    setup_for_distributed(opts.rank==0)
    print(opts)

