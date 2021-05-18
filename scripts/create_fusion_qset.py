"""
Randomly create a set of starts to test trajectories on.
"""
import argparse
import os
import pickle as pkl

import h5py
import numpy as np
from tqdm import tqdm


ACTION_MAX = 2500000
TEST_SHOTS = [164352, 161544, 161164, 164365, 161171, 161555, 161558, 163610,
              161442, 161452, 164555, 163022, 162776, 163034, 163292, 162916,
              161509, 161513, 161009, 161528]

def create_single_act_qset(options):
    if options.pudb:
        import pudb; pudb.set_trace()
    with open(os.path.join(options.shot_dir, 'headers.pkl'), 'rb') as f:
        headers = pkl.load(f)
    beta_idx = headers.index('efsbetan')
    ssize = options.state_size
    sstride = options.state_stride
    obsmooth = options.obs_smooth_amt
    np.random.seed(options.seed)
    os.makedirs(options.save_dir, exist_ok=True)
    shotnames = [sn for sn in os.listdir(options.shot_dir) if '.npy' in sn]
    qset = dict(
        observations=[],
        actions=[],
        rewards=[],
        next_observations=[],
        terminals=[],
        full_states=[],
        next_full_states=[],
        shot_ids=[],
        starts=[],
    )
    for sn in tqdm(shotnames):
        valid_shot = True
        for ts in TEST_SHOTS:
            if str(ts) in sn:
                valid_shot = False
                break
        if not valid_shot and not options.dont_filter_tests:
            continue
        shot = np.load(os.path.join(options.shot_dir, sn))
        shot_id = int(sn.split('_')[0])
        idx = 0
        while idx + ssize + sstride < shot.shape[1]:
            obend = idx + ssize
            obstrt = obend - obsmooth
            if options.add_slope_to_obs:
                qset['observations'].append(np.append(
                    np.mean(shot[:10, obstrt:obend], axis=1),
                    np.mean(shot[:10, obstrt:obend], axis=1)
                        - np.mean(shot[:10, idx:idx+obsmooth], axis=1)
                ))
            else:
                qset['observations'].append(
                        np.mean(shot[:10, obstrt:obend], axis=1))
            nxtend = idx + ssize + sstride
            nxtstrt = idx + sstride
            if options.raw_actions:
                qset['actions'].append(np.mean(shot[10:, nxtstrt:nxtend]))
            else:
                qset['actions'].append(np.mean(
                    shot[10:, nxtstrt:nxtend] / ACTION_MAX * 2 - 1))
            if options.add_slope_to_obs:
                qset['next_observations'].append(np.append(
                    np.mean(shot[:10, nxtend - obsmooth:nxtend], axis=1),
                    np.mean(shot[:10, nxtend - obsmooth:nxtend], axis=1)
                    - np.mean(shot[:10, nxtstrt:nxtstrt+obsmooth], axis=1)
                ))
            else:
                qset['next_observations'].append(
                        np.mean(shot[:10, nxtend - obsmooth:nxtend], axis=1))
            qset['rewards'].append(
                    3 - np.abs(qset['next_observations'][-1][beta_idx]
                               - options.beta_target))
            qset['terminals'].append(False)
            qset['full_states'].append(shot[:, idx:idx+ssize])
            qset['next_full_states'].append(shot[:, idx+sstride:idx+sstride+ssize])
            qset['shot_ids'].append(shot_id)
            qset['starts'].append(idx == 0)
            idx += sstride
    # Save set and headers.
    with h5py.File(os.path.join(options.save_dir, 'qset.hdf5'), 'w') as hdata:
        for k, v in qset.items():
            vv = np.array(v)
            if len(vv.shape) == 1:
                vv = vv.reshape(-1, 1)
            hdata.create_dataset(k, data=vv)
    with open(os.path.join(options.save_dir, 'headers.pkl'), 'wb') as f:
        pkl.dump(headers, f)


def load_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('--shot_dir', required=True)
    parser.add_argument('--save_dir', required=True)
    parser.add_argument('--state_size', type=int, default=200)
    parser.add_argument('--state_stride', type=int, default=200)
    parser.add_argument('--obs_smooth_amt', type=int, default=50)
    parser.add_argument('--beta_target', type=float, default=2)
    parser.add_argument('--dont_filter_tests', action='store_true')
    parser.add_argument('--raw_actions', action='store_true')
    parser.add_argument('--add_slope_to_obs', action='store_true')
    parser.add_argument('--pudb', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    return parser.parse_args()


if __name__ == '__main__':
    create_single_act_qset(load_options())
