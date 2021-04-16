import argparse
import torch
import pickle
import numpy as np
from pathlib import Path
from modelling.dynamics_construction import load_ensemble
from modelling.bisim_construction import load_bisim
from modelling.utils.torch_utils import set_cuda_device, ModelUnroller


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
    parser.add_argument('max_stitches', type=int)
    parser.add_argument('env_name')
    parser.add_argument('mean_file', nargs='?', default=None)
    parser.add_argument('std_file', nargs='?', default=None)
    parser.add_argument('-ub', '--use_bisimulation', action='store_true')
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


def CEM(row, obs_dim, action_dim, latent_dim, ensemble, bisim_model, epsilon, max_stitches, quantile, mean, std,
        env_name, device=None, **kwargs):
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
    assert (bisim_model is None) != (ensemble is None), "Can't pass both an ensemble and a bisim model"
    model = ModelUnroller(env_name, bisim_model if ensemble is None else ensemble)
    start_idx, end_idx, start_state, end_state, actions = row
    start_state = torch.Tensor(start_state).to(device)
    end_state = torch.Tensor(end_state).to(device)
    action_upper_bound = kwargs.get('action_upper_bound', 1.)
    action_lower_bound = kwargs.get('action_lower_bound', -1.)
    threshold = epsilon
    for horizon in range(1, max_stitches + 1):
        init_action = make_init_action(actions, horizon)
        initial_variance_divisor = kwargs.get('initial_variance_divisor', 4)
        max_iters = kwargs.get('max_iters', 6)
        popsize = kwargs.get('popsize', 256)
        num_elites = kwargs.get('num_elites', 64)
        alpha = kwargs.get('alpha', 0.25)
        mean = init_action
        var = torch.ones_like(mean) * ((action_upper_bound - action_lower_bound) / initial_variance_divisor) ** 2
        # X = stats.truncnorm(-2, 2, loc=np.zeros_like(mean), scale=np.ones_like(var))
        for i in range(max_iters):
            # lb_dist, ub_dist = mean - action_lower_bound, action_upper_bound - mean
            # constrained_var = np.minimum(np.minimum(np.square(lb_dist / 2), np.square(ub_dist / 2)), var)

            # samples = X.rvs(size=[popsize, action_dim]) * np.sqrt(var) + mean
            samples = torch.fmod(torch.randn(size=(popsize, horizon, action_dim), device=device),
                                 2) * torch.sqrt(var) + mean
            samples = torch.clip(samples, action_lower_bound, action_upper_bound)
            start_states = start_state.repeat(popsize, 1)
            p = 1 if bisim_model else 2
            model_obs, model_actions, model_rewards, model_terminals = model.model_unroll(start_states, samples)

            # this is because the dynamics models are written to predict the reward in the first component of the output
            # and the next state in all the following components
            displacements = model_obs[:, :, -1, :] - end_state
            if std is not None:
                displacements /= std
            distances = torch.linalg.norm(displacements, dim=-1, ord=p)
            quantiles = torch.quantile(distances, quantile, dim=0)
            min_quantile = quantiles.min()
            if min_quantile < threshold:
                threshold = min_quantile
                # success!
                min_index = quantiles.argmin()
                rewards = np.quantile(model_rewards[min_index, ...], 0.3, dim=0)
                actions = samples[min_index, ...].cpu().nump()
                outputs = (start_idx, end_idx, actions, min_quantile, rewards)
                break
                # return np.array([row[0], row[1], *samples[min_index, :].tolist(), min_quantile, reward])
            elites = samples[torch.argsort(quantiles)[:num_elites], ...]
            new_mean = torch.mean(elites, axis=0)
            new_var = torch.var(elites, axis=0)
            mean = alpha * mean + (1 - alpha) * new_mean
            var = alpha * var + (1 - alpha) * new_var
    return outputs


def main(args):
    device_num = ''
    set_cuda_device(device_num)
    device = torch.device('cpu' if device_num == '' else 'cuda:' + device_num)
    with args.input_file.open('rb') as f:
        input_data = pickle.load(f)
    mean = np.load(args.mean_file) if args.mean_file else None
    std = np.load(args.std_file) if args.std_file else None
    if args.use_bisimulation:
        bisim_model = load_bisim(Path(args.ensemble_path))
        bisim_model.to(device)
        ensemble = None
    else:
        ensemble = load_ensemble(args.ensemble_path, args.obs_dim, args.action_dim, cuda_device=device_num)
        bisim_model = None
        for model in ensemble:
            model.to(device)
    outputs = []
    # input_data = torch.Tensor(input_data)
    # input_data = input_data.to(device)
    for row in input_data:
        data = CEM(row, args.obs_dim, args.action_dim, args.latent_dim, ensemble, bisim_model, args.epsilon,
                   args.max_stitches, args.quantile, mean, std, args.env_name, device=device)
        if data is not None:
            outputs.append(data)
    with args.output_file.open('wb') as f:
        pickle.dump(outputs, f)


if __name__ == '__main__':
    args = parse_arguments()
    main(args)
