import os, sys
import numpy as np
from argparse import Namespace
import pickle as pkl
import torch
from catboost import CatBoostRegressor

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
import NNKit


##################################################################
def check_same(mat_1, mat_2):
    """
    mat_1: base matrix
    mat_2: matrix that you want to equal to base
    """
    if len(mat_2.shape) == len(mat_1.shape) + 1:
        max_diff_list = []
        for idx in range(mat_2.shape[0]):
            curr_mat = mat_2[idx]
            curr_max_diff = np.max(np.abs(mat_1 - mat_2))
            max_diff_list.append(curr_max_diff)
        max_diff = np.max(max_diff_list)
    elif len(mat_2.shape) == len(mat_1.shape):
        assert mat_2.size == mat_1.size
        mat_2 = mat_1.reshape(mat_2.shape)
        max_diff = np.max(np.abs(mat_1 - mat_2))
    if max_diff <= 1e-10:
        return True
    else:
        return False
##################################################################

def make_signal_namespace(signal_list, signals_dir, common_x_norm_dir=None):
    """
    Creates a Namespace list for input to CatBoostStatePredictor
    Args:
        signal_list: e.g. ['R0', 'aminor',..., ]
        signals_dir: directory where model dirs are located
            parent directory of signal directories, e.g.
            - R0
                - model_file.cbm
                - X_mean.npy, ...
            - ip
                - model_file.cbm
                - X_mean.npy, ...
            ...

    Returns: a list of Namespace objects, one per signal
    """
    dir_items = os.listdir(signals_dir)
    out = []
    for sig in signal_list:
        for item in dir_items:
            if sig in item:
                curr_sig_dir = item
                break
        sig_items = os.listdir(os.path.join(signals_dir, curr_sig_dir))

        if common_x_norm_dir is not None:
            x_norm_dir = common_x_norm_dir
        else:
            x_norm_dir = None
        y_norm_dir = None

        curr_unc_model_file = None  # None if there is not unc model
        for item in sig_items:
            if ('model' in item) and ('unc' not in item):
                curr_model_file = os.path.join(signals_dir, curr_sig_dir, item)
            if '{}_mean'.format(sig) in item:
                y_norm_dir = os.path.join(signals_dir, curr_sig_dir)
            if x_norm_dir is None and 'X_mean' in item:
                x_norm_dir = os.path.join(signals_dir, curr_sig_dir)
            if 'unc_model' in item:
                curr_unc_model_file = os.path.join(signals_dir, curr_sig_dir, item)
        curr_namespace = Namespace(signal=sig,
                                   model_file=curr_model_file,
                                   x_norm_dir=x_norm_dir,
                                   y_norm_dir=y_norm_dir,
                                   unc_model_file=curr_unc_model_file)
        out.append(curr_namespace)
    return out

def get_features_vec(states, actions):
    """
    Transform raw values into frames
    Args:
        states, actions arrays must be (num_ex, num_signals, time_len)
    Note: DOES NOT perform any normalization
    """

    # TODO: right now, assumes state, action is
    # Figure out where halves and thirds are.
    num_ex = states.shape[0]
    assert num_ex == actions.shape[0]

    num_state_sigs = states.shape[1]
    state_num = states.shape[2]
    state_half_pivot = state_num // 2
    state_third_pivots = [state_num // 3, 2 * (state_num // 3)]

    num_action_sigs = actions.shape[1]
    action_num = actions.shape[2]
    action_half_pivot = action_num // 2
    action_third_pivots = [action_num // 3, 2 * (action_num // 3)]

    # Generate the features.
    state_features = []
    action_features = []

    # state first
    state_features += get_subframe_features_vec(states[:, :, :])
    state_features += get_subframe_features_vec(states[:, :, :state_half_pivot])
    state_features += get_subframe_features_vec(states[:, :, state_half_pivot:])
    state_features += get_subframe_features_vec(
            states[:, :, :state_third_pivots[0]]
    )
    state_features += get_subframe_features_vec(
            states[:, :, state_third_pivots[0]:state_third_pivots[1]]
    )
    state_features += get_subframe_features_vec(
            states[:, :, state_third_pivots[1]:]
    )
    state_features = np.concatenate(state_features, axis=-1)
    state_features = state_features.reshape(num_ex,-1)

    # actions now
    action_features += get_subframe_features_vec(actions[:, :, :])
    action_features += get_subframe_features_vec(actions[:, :, :action_half_pivot])
    action_features += get_subframe_features_vec(actions[:, :, action_half_pivot:])
    action_features += get_subframe_features_vec(
            actions[:, :, :action_third_pivots[0]]
    )
    action_features += get_subframe_features_vec(
            actions[:, :, action_third_pivots[0]:action_third_pivots[1]]
    )
    action_features += get_subframe_features_vec(
            actions[:, :, action_third_pivots[1]:]
    )
    action_features = np.concatenate(action_features, axis=-1)
    action_features = action_features.reshape(num_ex,-1)

    features = np.concatenate([state_features, action_features], axis=-1)

    return features


def get_subframe_features_vec(subframe, dt=1):
    """
    subframe: (num_exs, num_signals, time_len)

    Output:
        (num_exs, num_signals, 3)
    """

    num_exs, num_signals, time_len = subframe.shape

    mean = np.mean(subframe, axis=2)
    variance = np.var(subframe, axis=2)
    subtimes = dt * np.arange(time_len)

    temp_subframe = subframe.reshape(-1, time_len)
    try:
        slope = np.polyfit(subtimes, temp_subframe.T, 1)[0]
        # slope = np.polynomial.polynomial.Polynomial.fit(
        #     subtimes, temp_subframe.T, 1)[0]
    except:
        import pdb; pdb.set_trace()
    slope = slope.reshape(variance.shape)
    out = np.stack([mean, variance, slope], axis=2)
    return [out]

def get_ens_pred_conf_bound(unc_preds, taus, target_qs, fidelity=10000):
    """
    unc_preds is a 3D numpy array of dimensions (ens_size, num_tau, num_x)
    each dim_1 corresponds to tau 0.01, 0.02, ..., and dim_2 are for the 
    set of x being predicted over
    """
    # taus = np.arange(0.01, 1, 0.01)
    num_ens, num_tau, num_x = unc_preds.shape

    mean_pred = np.mean(unc_preds, axis=0)  # shape (num_tau, num_x)
    std_pred = np.std(unc_preds, axis=0, ddof=1)
    stderr_pred = std_pred/np.sqrt(num_ens)  # shape (num_tau, num_x)
    gt_med = (taus > 0.5).reshape(num_tau, -1)
    lt_med = ~gt_med
    out = lt_med * (mean_pred - (1.96 * stderr_pred)) + \
          gt_med * (mean_pred + (1.96 * stderr_pred))

    return out

class FusionStatePredictor():

    def predict(self, state, action):
        raise NotImplementedError('Abstract Method')

    def multi_predict(self, states, actions):
        raise NotImplementedError('Abstract Method')


class CatBoostStatePredictor(FusionStatePredictor):
    def __init__(self, input_signal_order, state_signals_used, action_signals_used,
                 signal_namespace, num_actions, history_len, pred_len,
                 use_same_featurization=False):
        """
        Args:
            signal_namespace: list of Namespaces, each Namespace is e.g.
                                Namespace(signal='R0', model_file='**/**/model.**')
            num_state_signals: number of state signals that model takes in
            num_actions: dimension of actions
            history_len: history window is H long
            pred_len: predicts P into the future
        """
        self.input_signal_order = input_signal_order
        self.state_signals_used = state_signals_used
        self.action_signals_used = action_signals_used
        self.use_same_featurization = use_same_featurization

        self.num_signals = len(signal_namespace)
        self.num_actions = num_actions
        self.history_len = history_len
        self.pred_len = pred_len
        self.signal_namespace = signal_namespace
        self.signal_names = []
        self.signal_models = {}
        self.num_state_signals=len(state_signals_used)

        for item in self.signal_namespace:
            curr_signal = item.signal
            curr_model_file = item.model_file
            curr_unc_model_file = item.unc_model_file
            x_norm_dir = item.x_norm_dir
            y_norm_dir = item.y_norm_dir
            curr_signal_env = CatBoostSignalPredictor(
                                input_signal_order=self.input_signal_order,
                                state_signals_used=self.state_signals_used,
                                action_signals_used=self.action_signals_used,
                                signal_name=curr_signal,
                                num_state_signals=len(state_signals_used),
                                num_actions=self.num_actions,
                                history_len=self.history_len,
                                pred_len=self.pred_len,
                                model_file=curr_model_file,
                                unc_model_file=curr_unc_model_file,
                                x_norm_dir=x_norm_dir,
                                y_norm_dir=y_norm_dir)
            self.signal_names.append(curr_signal)
            self.signal_models[curr_signal] = curr_signal_env

    def predict(self, hist_state_act, action, order=None, unc_info=None):
        """
        Predict next state for each signal model
        Args:
            hist_state_act:
                 a state array of shape (num_signals + num_actions x H)
            action: an array of size P which is the action for
                    the next P timesteps
            order: list of signals that you want to predict next state for

        Returns: an array that specifies next state, shape (num_signals, 1)
        """
        hist_state_act = np.expand_dims(hist_state_act, 0)
        action = np.expand_dims(action, 0)

        temp_model = self.signal_models[self.signal_names[0]]
        state_idx = temp_model.state_idx  # list of idxs corresponding to state vars
        state = hist_state_act[:, state_idx, :]
        unnorm_model_input = get_features_vec(state, action)

        out_signals = []
        out_uncertainties = []
        signal_list = order if order is not None else self.signal_names
        for signal in signal_list:
            X_mean, X_std = self.signal_models[signal].X_mean, \
                            self.signal_models[signal].X_std
            model_input = (unnorm_model_input - X_mean) / X_std
            # get delta of next state
            next_state_delta = self.signal_models[signal].model.predict(model_input)

            if self.signal_models[signal].normalized:
                next_state_delta = next_state_delta * float(self.signal_models[signal].y_std)
                next_state_delta += float(self.signal_models[signal].y_mean)

            next_state = \
                hist_state_act[0, self.signal_models[signal].signal_idx, -1] \
                + next_state_delta

            # deal with uncertainty outputs
            out_signals.append(next_state)

        out_signals = np.asarray(out_signals).reshape(self.num_signals, 1)

        return out_signals

    def multi_predict(self, hist_state_acts, actions, order=None):
        """
        Predict next state for each signal model for a number of frames.
        Args:
            state_and_hist_act:
                numpy array for the state and actions of shape:
                (num_frames X num_signals + num_actions x H)
            actions: numpy array for th next P time steps, with shape:
                (num_frames X num_actions X time_steps)
            action: List of arrays of size P which is the action for
                    the next P timesteps
            order: list of signals that you want to predict next state for

        Returns: an array that specifies next state, shape (num_signals, 1)
        """

        num_transitions = hist_state_acts.shape[0]
        if num_transitions != len(actions):
            ValueError('hist_state_acts and actions must have same number'
                       'of rows, given: %d and %d'
                       % (num_transitions, len(actions)))

        temp_model = self.signal_models[self.signal_names[0]]
        state_idx = temp_model.state_idx
        action_idx = temp_model.action_idx
        states = hist_state_acts[:, state_idx, :]

        unnorm_model_input = get_features_vec(states, actions)

        out_signals = []
        signal_list = order if order is not None else self.signal_names
        for signal in signal_list:
            X_mean, X_std = self.signal_models[signal].X_mean, self.signal_models[signal].X_std
            model_input = (unnorm_model_input - X_mean)/X_std
            next_state_delta = self.signal_models[signal].model.predict(model_input)

            if self.signal_models[signal].normalized:
                next_state_delta = next_state_delta * float(self.signal_models[signal].y_std)
                next_state_delta += float(self.signal_models[signal].y_mean)

            next_state = [sh[self.signal_models[signal].signal_idx, -1] for sh in hist_state_acts]
            next_state = np.asarray(next_state) + next_state_delta
            out_signals.append(next_state)

        out_signals = np.asarray(out_signals)
        return out_signals.reshape(self.num_signals, num_transitions)


class SignalPredictor():

    def predict(self, state, action):
        raise NotImplementedError('Abstract Method')


class CatBoostSignalPredictor(SignalPredictor):
    def __init__(self, input_signal_order, state_signals_used, action_signals_used,
                signal_name, num_state_signals, num_actions, history_len, pred_len,
                model_file, unc_model_file, x_norm_dir, y_norm_dir):
        """
        initializes a CatBoost model for rollouts
        Args:
            signal_name: name of the signal that this model predicts
            num_state_signals: number of state signals that model takes in
            num_actions: dimension of actions
            history_len: model takes in history of (states/actions) of length H
            pred_len: model predicts P into the future
            model_file: file of the predictive model used
            unc_model_file: file of UQ model used, None if there is no unc_model with env
            normalization_dir: directory of numpy arrays that contain
                                1) X_mean.npy 2) X_std.npy 3) y_mean.npy,
                                4) y_std.npy that were used to normalize training
                               None if normalization wasn't used during model train
        """

        """
        when making predictions with the model, the signals in the input data
        must be in the followings order
        """

        self.input_signal_order = input_signal_order
        self.state_signals_used = state_signals_used # must be in model order
        self.action_signals_used = action_signals_used # must be in model order

        self.signal_name = signal_name
        self.signal_idx = self.input_signal_order.index(self.signal_name)
        self.num_state_signals = num_state_signals
        self.num_actions = num_actions

        if self.state_signals_used is not None: # includes hist actions as part of state
            self.state_idx = [self.input_signal_order.index(x)
                              for x in self.state_signals_used]
        else:
            self.state_idx = None

        if self.action_signals_used is not None:
            actions_in_input_order = [x for x in self.input_signal_order if
                                      x in self.action_signals_used]
            self.action_idx = [actions_in_input_order.index(x)
                               for x in self.action_signals_used]
        else:
            self.action_idx = None

        self.history_len = history_len
        self.pred_len = pred_len
        self.model_file = model_file
        self.unc_model_file = unc_model_file # here, unc_model_file will point to a pickle file that is a list of NN's
        self.x_norm_dir = x_norm_dir
        self.y_norm_dir = y_norm_dir

        self._dt = 1

        self.model = self._load_in_model()

        if self.unc_model_file is not None:
            self.unc_model = self._load_in_unc_model()
            self.unc_num_ens = len(self.unc_model)
        else:
            self.unc_model = None

        if x_norm_dir is not None:
            self.normalized = True
            self._load_in_normalization()

    def _get_features(self, state, action):
        """
        Transform raw values into frames
        Args:
            state, action arrays must be (num_signals x time_len)
        """

        # TODO: right now, assumes state, action is
        # Figure out where halves and thirds are.
        state_num = state.shape[1]
        state_half_pivot = state_num // 2
        state_third_pivots = [state_num // 3, 2 * (state_num // 3)]

        action_num = action.shape[1]
        action_half_pivot = action_num // 2
        action_third_pivots = [action_num // 3, 2 * (action_num // 3)]
        # Generate the features.
        features = []
        for f_idx, fname in enumerate(self.state_idx):
            features += self._get_subframe_features(state[f_idx, :])
            features += self._get_subframe_features(state[f_idx, :state_half_pivot])
            features += self._get_subframe_features(state[f_idx, state_half_pivot:])
            features += self._get_subframe_features(
                    state[f_idx, :state_third_pivots[0]]
            )
            features += self._get_subframe_features(
                    state[f_idx, state_third_pivots[0]:state_third_pivots[1]]
            )
            features += self._get_subframe_features(
                    state[f_idx, state_third_pivots[1]:]
            )

        for f_idx, fname in enumerate(self.action_idx):
            features += self._get_subframe_features(action[f_idx, :])
            features += self._get_subframe_features(action[f_idx, :action_half_pivot])
            features += self._get_subframe_features(action[f_idx, action_half_pivot:])
            features += self._get_subframe_features(
                    action[f_idx, :action_third_pivots[0]]
            )
            features += self._get_subframe_features(
                    action[f_idx, action_third_pivots[0]:action_third_pivots[1]]
            )
            features += self._get_subframe_features(
                    action[f_idx, action_third_pivots[1]:]
            )

        features = np.asarray(features).reshape(1,-1)
        if self.normalized:
            assert features.size == self.X_mean.size
            features = (features - self.X_mean)/self.X_std

        return features

    def _get_subframe_features(self, subframe):
        mean = np.mean(subframe)
        variance = np.var(subframe)
        subtimes = self._dt * np.arange(len(subframe))
        slope = np.polyfit(subtimes, subframe, 1)[0]
        return [mean, variance, slope]

    def _load_in_model(self):
        model = CatBoostRegressor()
        model.load_model(self.model_file)
        return model

    def _load_in_unc_model(self):
        model = pkl.load(open(self.unc_model_file, 'rb'))
        return model

    def _load_in_normalization(self):
        self.X_mean, self.X_std, self.y_mean, self.y_std = \
            None, None, None, None

        for item in os.listdir(self.x_norm_dir):
            if 'X_mean' in item:
                self.X_mean = np.load(os.path.join(self.x_norm_dir, item))
            elif 'X_std' in item:
                self.X_std = np.load(os.path.join(self.x_norm_dir, item))

        for item in os.listdir(self.y_norm_dir):
            if '{}_mean'.format(self.signal_name) in item:
                self.y_mean = np.load(os.path.join(self.y_norm_dir, item))
            elif '{}_std'.format(self.signal_name) in item:
                self.y_std  = np.load(os.path.join(self.y_norm_dir, item))

        if (self.X_mean is None) or (self.X_std is None) or \
            (self.y_mean is None) or (self.y_std is None):
            raise RuntimeError('normalization dir does not have the correct files')

        assert self.X_mean.shape == self.X_std.shape
        assert self.y_mean.shape == self.y_mean.shape

    def _make_model_input(self, state, hist_action, action):
        """
        Args:
            state: shape (num_signals x H)
            hist_action: shape is (num_actions x H)
            action: shape is (num_actions x P)

        Returns: array of shape (1, ***) for input to model
        """
        assert(state.shape == (self.num_signals, self.history_len)
               and (hist_action.shape == (self.num_actions,self.history_len))
               and (action.shape == (self.num_actions, self.pred_len)))

        time_list = list(np.arange(0, self.history_len, 50)) + \
                         [self.history_len - 1]
        time_idx = np.array(time_list)
        train_state = state[:, time_idx]
        train_input = np.concatenate([train_state.flatten(),
                                     hist_action.flatten(),
                                     action.flatten()], axis=0).reshape(1, -1)
        if self.normalized:
            assert train_input.size == self.X_mean.size
            train_input = (train_input - self.X_mean)/self.X_std
        return train_input

    def predict_unc(self, model_input, format, alpha=0.05):
        """
        Predicts uncertainty about future state
        Currently outputs centered 1-alpha prediction intervals
        Args:
            model_input: dimensions (num_x, dim_x)
            alpha: by default set to 0.05 for 95% PI
        """
        num_input = model_input.shape[0]

        if format == 'pi':
            quantiles = np.array([(alpha/2), 1-(alpha/2)]).reshape(-1, 1)
            quantiles_rep = np.tile(quantiles, (num_input, 1)).reshape(-1, 1)
            model_input_rep = np.repeat(model_input, quantiles.size, axis=0)
            unc_model_input = np.concatenate([model_input_rep,
                                              quantiles_rep], axis=1)
        elif format == 'particle':
            quantiles = np.random.uniform(size=(num_input, 1))
            unc_model_input = np.concatenate([model_input, quantiles], axis=1)

        unc_model_input = (torch.from_numpy(unc_model_input)).float()

        if self.unc_num_ens == 1:
            with torch.no_grad():
                unc_pred = self.unc_model[0](unc_model_input).numpy().reshape(num_input, -1)
        elif self.unc_num_ens > 1:
            ens_out = []
            for ens_idx in range(self.unc_num_ens):
                with torch.no_grad():
                    single_out = self.unc_model[ens_idx](unc_model_input)
                ens_out.append(single_out.numpy().reshape(num_input, -1).T)
            ens_out_concat = np.stack(ens_out, axis=0)
            unc_pred = get_ens_pred_conf_bound(ens_out_concat, quantiles.flatten(),
                                               quantiles.flatten(),
                                               fidelity=10000)

        return unc_pred

    def predict_from_features(self, hist_state_act, model_input):
        """
        predicts the state P into the future given already engineered features.
        Args:
            model_input: The input to put into the model.
        """
        curr_signal_state = hist_state_act[self.signal_idx, -1]
        assert len(self.model.feature_names_) == model_input.size
        next_state_delta = self.model.predict(model_input)

        if self.normalized:
            assert next_state_delta.shape == self.y_mean.shape
            next_state_delta = (next_state_delta * self.y_std) + self.y_mean

        next_state = float(curr_signal_state) + float(next_state_delta)
        return next_state


    def predict(self, hist_state_act, action):
        """
        predicts the state P into the future
        Args:
            hist_state_act:
              a state array of shape (num_signals + num_actions x H)
            action: an array of size P which is the action for
                    the next P timesteps
        """
        state = hist_state_act[self.state_idx, :].reshape(self.num_state_signals,
                                                                self.history_len)
        action = action[self.action_idx, :].reshape(self.num_actions, self.pred_len)
        model_input = self._get_features(state, action)
        return self.predict_from_features(hist_state_act, model_input)


    def multi_predict(self, hist_state_acts, actions):
        """
        predicts the state P into the future
        Args:
            state_and_hist_act:
              a state array of shape (num_signals + num_actions x H)
            action: an array of size P which is the action for
                    the next P timesteps
        """
        model_input = []
        for idx in range(len(actions)):
            hist_state_act = hist_state_acts[idx]
            action = actions[idx]
            state = hist_state_act[self.state_idx, :].reshape(self.num_state_signals,
                                                                    self.history_len)
            curr_signal_state = hist_state_act[self.signal_idx, -1]
            action = action[self.action_idx, :].reshape(self.num_actions, self.pred_len)

            model_input.append(self._get_features(state, action))
        model_input = np.asarray(model_input).reshape(len(actions), -1)


        next_state_delta = self.model.predict(model_input)

        if self.normalized:
            next_state_delta = next_state_delta * float(self.y_std)
            next_state_delta += float(self.y_mean)

        next_state = [sh[self.signal_idx, -1] for sh in hist_state_acts]
        next_state = np.asarray(next_state) + next_state_delta
        return next_state


if __name__ == '__main__':
    signal_list = ['efsbetan', 'kappa', 'tritop', 'tribot', 'R0', 'efsvolume', 'aminor', 'dssdenest', 'ip', 'efsli']
    signal_namespace = make_signal_namespace(signal_list, '../notebooks/state_prediction/models')
    test_env = CatBoostStatePredictor(signal_namespace, 1, 300, 200)

    rand_state_hist_act = np.random.random(size=(11, 300))
    rand_action = np.random.random(size=200)

    test_env.predict(rand_state_hist_act, rand_action)
