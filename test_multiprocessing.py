import sys
import torch
import numpy as np
from modelling.dynamics_construction import load_ensemble
import torch.multiprocessing
from functools import partial
from tqdm import tqdm
from torch.multiprocessing import Pool
# from multiprocessing import Pool
import util
torch.multiprocessing.set_sharing_strategy('file_system')
# multiprocessing.set_sharing_strategy(
# torch.multiprocessing.set_start_method('spawn')



def main(ncpus):
    size = 1000000
    data = np.random.random((size, 6))
    data = torch.Tensor(data)
    ensemble = load_ensemble(sys.argv[2], obs_dim=4, act_dim=2)
    # for model in ensemble:
        # model.share_memory()
    plan_fn = partial(util.dummy_CEM,
                      ensemble=ensemble,
                      obs_dim=4,
                      action_dim=6,
                      epsilon=0.1,
                      quantile=0.9)
    outputs = []
    with Pool(processes=ncpus) as pool:
        outputs = list(tqdm(pool.imap(plan_fn, data), total=size))
    print(outputs)

if __name__ == '__main__':
    main(int(sys.argv[1]))
