"""
Do PPO fine tuning.
"""
import argparse
import os

from graph_tool import load_graph
import gym
import torch
from torch.utils.data import DataLoader, TensorDataset

from modelling.critic_construction import load_value_function
from modelling.dynamics_construction import load_ensemble
from modelling.policy_construction import ppo_fine_tune, load_policy
from modelling.utils.graph_util import make_boltzmann_policy_dataset
from modelling.utils.torch_utils import ModelUnroller
from util import get_starts_from_graph


def run(args):
    # Load in everything.
    env = gym.make(args.env)
    graph = load_graph(os.path.join(args.graph_dir, 'vi.gt'))
    obs_dim = env.observation_space.low.size
    act_dim = env.action_space.low.size
    policy = load_policy(args.policy_dir, obs_dim, act_dim,
                         cuda_device=args.cuda_device)
    vf = load_value_function(args.vf_dir,
                             cuda_device=args.cuda_device)
    dynamics = load_ensemble(args.dyn_dir, obs_dim, act_dim,
                             cuda_device=args.cuda_device)
    # Assemble dataset of starts.
    if 'maze' in args.env:
        starts = get_starts_from_graph(graph, env)
    else:
        starts = None
    data, _, _ = make_boltzmann_policy_dataset(
            graph=graph,
            # n_collects=1000,
            n_collects=graph.num_vertices(),
            temperature=0,
            gamma=args.gamma,
            max_ep_len=env._max_episode_steps,
            starts=starts,
    )
    dataset = DataLoader(
        TensorDataset(torch.Tensor(data['observations'])),
        batch_size=1024,
        shuffle=False,
        pin_memory=True,
    )
    # Create the model unroller.
    unroller = ModelUnroller(
        env_name=args.env,
        model=dynamics,
    )
    # Do the fine tuning.
    ppo_params = vars(args)
    del ppo_params['env']
    del ppo_params['graph_dir']
    del ppo_params['policy_dir']
    del ppo_params['vf_dir']
    del ppo_params['dyn_dir']
    del ppo_params['pudb']
    ppo_fine_tune(
        dataset=dataset,
        policy=policy,
        vf=vf,
        model_unroller=unroller,
        env=env,
        **ppo_params,
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', required=True)
    parser.add_argument('--graph_dir', required=True)
    parser.add_argument('--policy_dir', required=True)
    parser.add_argument('--vf_dir', required=True)
    parser.add_argument('--dyn_dir', required=True)
    parser.add_argument('--save_path', required=True)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--horizon', type=int, default=5)
    parser.add_argument('--start_unroll_batches_per_epoch', type=int,
                        default=int(1e3))
    # parser.add_argument('--start_unroll_batches_per_epoch', type=int,
    #                     default=int(100))
    parser.add_argument('--num_policy_batch_update_per_epoch', type=int,
                        default=25)
    parser.add_argument('--ppo_update_batch_size', type=int, default=256)
    parser.add_argument('--outer_loops', type=int, default=1)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--lmbda', type=float, default=0.95)
    parser.add_argument('--epsilon', type=float, default=0.2)
    parser.add_argument('--save_freq', type=int, default=-1)
    parser.add_argument('--max_ep_len', type=int, default=1000)
    parser.add_argument('--num_eval_eps', type=int, default=10)
    parser.add_argument('--cuda_device', type=str, default='')
    parser.add_argument('--pudb', action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    if args.pudb:
        import pudb; pudb.set_trace()
    run(args)
