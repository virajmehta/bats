import argparse
import os
from pathlib import Path

import graph_tool
import torch
import pickle
import numpy as np
from fusion.env_creator import create_single_act_target_env
from modelling.bisim_construction import load_bisim
from modelling.utils.torch_utils import set_cuda_device, ModelUnroller


MODEL_PARAMS = dict(
    state_model_dir='/zfsauton/project/public/ysc/trained_models/200_200',
    disrupt_model_dir='/zfsauton/project/public/ichar/FusionModels/tearing',
    shot_dir='/zfsauton/project/public/ichar/FusionModels/single_start',
)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', type=Path)
    parser.add_argument('output_file', type=Path)
    parser.add_argument('ensemble_path')
    parser.add_argument('obs_dim', type=int)
    parser.add_argument('action_dim', type=int)
    parser.add_argument('latent_dim', type=int)
    parser.add_argument('epsilon', type=float)
    parser.add_argument('quantile', type=float)
    parser.add_argument('max_stitch_length', type=int)
    parser.add_argument('env_name')
    parser.add_argument('mean_file', nargs='?', default=None)
    parser.add_argument('std_file', nargs='?', default=None)
    parser.add_argument('-ub', '--use_bisimulation', action='store_true')
    parser.add_argument('-uapi', '--use_all_planning_itrs',
                        action='store_true')
    return parser.parse_args()


def prepare_model_inputs_torch(obs, actions):
    return torch.cat([obs, actions], dim=-1)


def make_init_action(actions, horizon):
    action_horizon = len(actions)
    action_dim = len(actions[0])
    action = torch.Tensor(actions)
    if horizon == action_horizon:
        return action
    elif horizon < action_horizon:
        return action[:horizon, ...]
    else:
        padding = torch.zeros((horizon - action_horizon, action_dim))
        return torch.cat((action, padding))


def CEM(row, obs_dim, action_dim, latent_dim, environment, epsilon,
        max_stitch_length, quantile, mean, std, env_name, device=None,
        use_all_iterations=False, **kwargs):
    '''
    attempts CEM optimization to plan a single step from the start state to the end state.
    initializes the mean with the init action.

    if we are using a standard dynamics model:
        row: a numpy array of size D = 2 + 2 * obs_dim + action_dim, assumed to be of the form:
        start state (int); end state (int); start_obs (obs_dim floats); end_obs (obs_dim floats);
        init_action (action_dim floats);
    if we are using the bisimulation metric:
        row: a numpy array of size D = 2 + 2 * latent_dim + action_dim, assumed to be of the form:
        start_state (int); end state (int); start_z (latent_dim floats); end_obs (latent_dim floats);
            init_action (action_dim floats);

    if successful, returns the action and predicted reward of the transition.
    if unsuccessful, returns None, None
    for now we assume that the actions are bounded in [-1, 1]
    '''
    # TODO: figure out what the row is, adjust planning for max_stitches
    start_idx, end_idx, start_state, end_obs, actions = row
    action_upper_bound = kwargs.get('action_upper_bound', 1.)
    action_lower_bound = kwargs.get('action_lower_bound', -1.)
    threshold = epsilon
    for horizon in range(1, max_stitch_length + 1):
        init_action = make_init_action(actions, horizon)
        initial_variance_divisor = kwargs.get('initial_variance_divisor', 4)
        max_iters = kwargs.get('max_iters', 6)
        popsize = kwargs.get('popsize', 256)
        num_elites = kwargs.get('num_elites', 64)
        alpha = kwargs.get('alpha', 0.25)
        mean = init_action
        var = torch.ones_like(mean) * ((action_upper_bound - action_lower_bound) / initial_variance_divisor) ** 2
        best_found, best_qscore = None, float('inf')
        for i in range(max_iters):
            samples = torch.fmod(torch.randn(size=(popsize, horizon, action_dim), device=device),
                                 2) * torch.sqrt(var) + mean
            samples = torch.clip(samples, action_lower_bound, action_upper_bound)
            start_states = start_state.repeat(popsize, 1)
            model_obs, model_rewards, model_terminals =\
                    [[] for _ in range(3)]
            for acts in samples:
                trajobs, trajrews, trajterms = [[] for _ in range(3)]
                env.reset()
                env.state_log = start_state[:env.num_signals]
                env.act_log = start_state[env.num_signals:]
                for act in acts:
                    nxt, rew, done, _ = env.step(act)
                    trajobs.append(nxt)
                    trajrews.append(rew)
                    trajterms.append(done)
                model_obs.append(np.array(trajobs))
                model_rewards.append(np.array(trajrews))
                model_terminals.append(np.array(trajterms))
            model_obs = np.array(model_obs)
            model_rewards = np.array(model_rewards)
            model_terminals = np.array(model_terminals)
            good_indices = np.nonzero(~model_terminals.any(axis=1))
            if len(good_indices) == 0:
                return None
            model_obs = model_obs[good_indices[0], ...]
            model_rewards = model_rewards[good_indices[0], ...]
            samples = samples[good_indices[0], ...]

            # this is because the dynamics models are written to predict the reward in the first component of the output
            # and the next state in all the following components
            displacements = model_obs[:, -1, :] - end_obs[None, None, :]
            if std is not None:
                displacements /= std
            distances = np.linalg.norm(displacements, axis=-1)
            min_idx = np.argmin(distances)
            min_dist = distances[min_idx]
            if min_dist < threshold:
                threshold = min_dist
                # success!
                if min_dist < best_qscore:
                    obs_history = model_obs[min_index, 1:-1, :]
                    rewards = model_rewards[min_idx, ...]
                    actions = samples[min_idx, ...]
                    model_errs = np.array([min_dist])
                    best_found = (start_idx, end_idx, actions,
                            min_quantile, rewards, obs_history, model_errs)
                if not use_all_iterations:
                    return best_found
            elites = samples[np.argsort(distances)[:num_elites], ...]
            new_mean = torch.mean(elites, dim=0)
            new_var = torch.var(elites, dim=0)
            mean = alpha * mean + (1 - alpha) * new_mean
            var = alpha * var + (1 - alpha) * new_var
    return None


def main(args):
    device_num = ''
    set_cuda_device(device_num)
    device = torch.device('cpu' if device_num == '' else 'cuda:' + device_num)
    with args.input_file.open('rb') as f:
        input_data = pickle.load(f)
    mean = np.load(args.mean_file) if args.mean_file else None
    std = np.load(args.std_file) if args.std_file else None
    env = create_single_act_target_env(
        horizon=5,
        init_at_start_only=True,
        num_cached_starts=False,
        **MODEL_PARAMS,
    )
    bisim_model = None
    outputs = []
    for row in input_data:
        data = CEM(row, args.obs_dim, args.action_dim, args.latent_dim,
                   env, args.epsilon, args.max_stitch_length,
                   args.quantile, mean, std, args.env_name, device=device,
                   use_all_planning_itrs=args.use_all_planning_itrs)
        if data is not None:
            outputs.append(data)
    with args.output_file.open('wb') as f:
        pickle.dump(outputs, f)


if __name__ == '__main__':
    args = parse_arguments()
    main(args)
