from collections import deque
from copy import deepcopy
import numpy as np
import os
import gym
from gym import spaces
from scipy.interpolate import interp1d
from simple_pid import PID

SIGNAL_BOUNDS = {
    'R0': (1.47, 2.02),
    'aminor': (0.38, 0.67),
    'dssdenest': (-0.25, 30.69),
    'efsbetan': (-666.46, 226.88),
    'efsli': (0, 513.15),
    'efsvolume': (0, 23.25),
    'ip': (-99779.68, 1896687.27),
    'kappa': (0.67, 2.08),
    'tribot': (-0.38, 1),
    'tritop': (-0.38, 1),
}
# The average and standard deviation of signal data.
SIGNAL_STATS = {
    'R0': (1.74, 0.04),
    'aminor': (0.59, 0.02),
    'dssdenest': (3.64, 1.84),
    'efsbetan': (1.32, 0.94),
    'efsli': (1.03, 0.45),
    'efsvolume': (17.67, 2.64),
    'ip': (985558.82, 281681.72),
    'kappa': (1.74, 0.14),
    'tribot': (0.5, 0.22),
    'tritop': (0.38, 0.22),
}


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

def make_disrupt_model_input(signal_list, model_dir):
    dir_items = os.listdir(model_dir)
    out = {'data_in_columns': signal_list}
    for item in dir_items:
        if 'dranges' in item:
            out['data_min_max_path'] = item
        elif 'headers' in item:
            out['cb_column_path'] = item
        elif item[-4:] == '.cbm':
            out['cb_model_path'] = item
    return out

class BaseFusionEnv(gym.Env):
    def __init__(
            self,
            signal_list,
            dyn_model,
            dis_model,
            init_dir,
            horizon,
            dyn_time_scale=(200,200),
            dis_time_scale=(100,250),
            init_at_start_only=False,
            num_cached_starts=None,
            preloaded_inits=None,
            normalize_observations=False,
            standardize_observations=True,
    ):
        """
        Args:
            signal_list: (list) of signals in order desired
            dyn_model: Model for the dynamics.
            dis_model: Model for the disruption.
            init_dir: Directory of shots to base starts on.
            horizon: Length of time to run for.
            dyn_time_scale: Time scales for dynamics model where the first
                number is amount of history for input and second is the
                number of ms in the future to predict.
            dis_time_scale: Time scales for disruption model where the first
                number is amount of history for input and second is the
                number of ms in the future to predict.
            init_at_start_only: Whether to only initialize environment at the
                start of recorded shots.
            num_cached_starts: Number of start states to cache in deque.
            preloaded_inits: Set of inits to use instead of reading from dir.
            normalize_observations: Whether to normalize observations.
            standardize_observations: Whether to standardize observations.
        """
        self.signal_list = signal_list
        self.beta_idx = signal_list.index('efsbetan')
        self.num_signals = len(self.signal_list)
        self.state_model = dyn_model
        self.disrupt_model = dis_model
        self.init_dir = init_dir
        self.horizon = horizon
        self.dyn_time_scale = dyn_time_scale
        self.dis_time_scale = dis_time_scale
        self.init_at_start_only = init_at_start_only
        self.num_cached_starts = num_cached_starts
        self.preloaded_inits = preloaded_inits
        self.normalize_observations = normalize_observations
        self.standardize_observations = standardize_observations

        # self.state_size = np.sum(dyn_time_scale)
        self.state_size = dyn_time_scale[0]
        self.inits = deque()
        self._fill_inits()

    def reset(self):
        self.t = 0
        self.curr_state_idx = 0
        if len(self.inits) == 0:
            self._fill_inits()
        init_draw = self.inits.popleft()
        self.state_log = deepcopy(init_draw[0])
        self.act_log = deepcopy(init_draw[1])
        if self.num_cached_starts is None:
            self.inits.append(init_draw)

        # Fill in the disruption probabilities so far.
        self.stab_log = [0 for _ in range(self.dis_time_scale[0])]
        frame = self.state_log[:, -self.dis_time_scale[0]:].T
        x_ends = np.asarray([len(self.stab_log), self.state_log.shape[1]])
        xs = np.linspace(x_ends[0], x_ends[1], x_ends[1] - x_ends[0] + 1)
        ys = np.asarray([self.stab_log[-1], self.disrupt_model.predict(frame)])
        self.stab_log += list(interp1d(x_ends, ys)(xs[1:]))

    def step(self, action):
        """Take step in the environment.
        Args:
            action: ndarray of the next X actions.
        """
        info = {}
        # Extend the action history.
        self.act_log = np.hstack([self.act_log,
            action.reshape(self.act_log.shape[0], -1)])

        # Make prediction of the future state.
        pred_pt = self.curr_state_idx + self.dyn_time_scale[1]
        curr_frame = np.vstack([
            self.state_log[:, pred_pt:pred_pt + self.dyn_time_scale[0]],
            self.act_log[:, pred_pt:pred_pt + self.dyn_time_scale[0]],
        ])
        future_state = self.state_model.predict(curr_frame, action).flatten()
        # For state, fill in between by doing interpolation.
        last_state = self.state_log[:, -1]
        xs = np.linspace(0, self.dyn_time_scale[1] - 1, self.dyn_time_scale[1])
        rolled = np.asarray(
            interp1d(np.stack([0, self.dyn_time_scale[1]], axis=0),
                     np.stack([last_state, future_state], axis=-1))(xs)
        )
        self.state_log = np.hstack([
            self.state_log,
            rolled,
        ])
        # Roll stability forward.
        frame = self.state_log[:, - self.dis_time_scale[0]:].T
        x_ends = np.asarray([len(self.stab_log), self.state_log.shape[1]])
        xs = np.linspace(x_ends[0], x_ends[1], x_ends[1] - x_ends[0] + 1)
        ys = np.asarray([self.stab_log[-1], self.disrupt_model.predict(frame)])
        self.stab_log += list(interp1d(x_ends, ys)(xs[1:]))
        # Calculate the next rewards.
        reward, rew_info = self._get_reward()
        info.update(rew_info)
        # Return the observations
        self.t += 1
        self.curr_state_idx += self.dyn_time_scale[1]
        done = self.t >= self.horizon
        observations = self.state_log[:, -self.state_size:]
        info['raw_observation'] = observations
        if self.normalize_observations:
            mins = np.asarray([SIGNAL_BOUNDS[s][0] for s in self.signal_list])
            maxs = np.asarray([SIGNAL_BOUNDS[s][1] for s in self.signal_list])
            mins, maxs = mins.reshape(-1, 1), maxs.reshape(-1, 1)
            observations = (observations - mins) / (maxs - mins)
        elif self.standardize_observations:
            means = np.asarray([SIGNAL_STATS[s][0] for s in self.signal_list])
            stds = np.asarray([SIGNAL_STATS[s][1] for s in self.signal_list])
            means, stds = means.reshape(-1, 1), stds.reshape(-1, 1)
            observations = (observations - means) / stds
        return observations, reward, done, info


    def _get_reward(self):
        """Calculate the current reward for after a step has been made."""
        return 0, {}


    def _fill_inits(self):
        """Fill a deque of inits to use upon reset."""
        if not self.preloaded_inits is None:
            self.inits = deepcopy(self.preloaded_inits)
            return
        shot_files = [f for f in os.listdir(self.init_dir) if '.npy' in f]
        shot_idx = -1
        filling = True
        while filling:
            valid_shot = False
            while not valid_shot:
                if self.num_cached_starts is None:
                    shot_idx += 1
                else:
                    shot_idx = np.random.randint(len(shot_files))
                shot = np.load(os.path.join(self.init_dir,
                                            shot_files[shot_idx]))
                valid_shot = shot.shape[1] > self.state_size
            if self.init_at_start_only:
                self.inits.append((shot[:self.num_signals, :self.state_size],
                                   shot[self.num_signals:, :self.state_size]))
            else:
                start_idx = np.random.randint(shot.shape[1] - self.state_size)
                end_idx = start_idx + self.state_size
                self.inits.append((shot[:self.num_signals, start_idx:end_idx],
                                   shot[self.num_signals:, start_idx:end_idx]))
            if self.num_cached_starts is None:
                filling = shot_idx < len(shot_files) - 1
            else:
                filling = len(self.inits) < self.num_cached_starts


class SingleActTargetFusionEnv(BaseFusionEnv):
    def __init__(
            self,
            signal_list,
            dyn_model,
            dis_model,
            init_dir,
            horizon,
            dyn_time_scale=(200,200),
            dis_time_scale=(100,250),
            init_at_start_only=False,
            num_cached_starts=None,
            preloaded_inits=None,
            normalize_observations=False,
            standardize_observations=True,
            beta_target=1.5,
            rew_coefs=(9, 10),
            ob_smooth_amt=2,
            pos_reward=True,
    ):
        """
        Args:
            signal_list: (list) of signals in order desired
            dyn_model: Model for the dynamics.
            dis_model: Model for the disruption.
            inits: List of tuples where first item is np array of init states
                and second is np array of actions.
            horizon: Length of time to run for.
            dyn_time_scale: Time scales for dynamics model where the first
                number is amount of history for input and second is the
                number of ms in the future to predict.
            dis_time_scale: Time scales for disruption model where the first
                number is amount of history for input and second is the
                number of ms in the future to predict.
            beta_target: Target beta to reach.
            rew_coefs: Coefficients for reward in form of (a, b) where the
                reward is (-beta_L2 - a * sigmoid(b * (disrupt_prob - 0.5))).
        """
        # Have normalized action space but this will really be between
        # 0 and 2500000.
        self.beta_target = beta_target
        self.rew_coefs = rew_coefs
        self.action_space = spaces.Box(
                low=-1,
                high=1,
                shape=(1,),
                dtype=np.float32,
        )
        # Observation space is the last recorded signal + slope between this
        # and last prediction.
        lows, highs = [np.array(l) for l in
                       zip(*[SIGNAL_BOUNDS[s] for s in signal_list])]
        self.observation_space = spaces.Box(
                low=np.tile(lows, 2),
                high=np.tile(highs, 2),
                dtype=np.float32,
        )
        self.ob_smooth_amt = ob_smooth_amt
        self.pos_reward = pos_reward
        super(SingleActTargetFusionEnv, self).__init__(
                signal_list,
                dyn_model,
                dis_model,
                init_dir,
                horizon,
                dyn_time_scale=dyn_time_scale,
                dis_time_scale=dis_time_scale,
                init_at_start_only=init_at_start_only,
                num_cached_starts=num_cached_starts,
                preloaded_inits=preloaded_inits,
                normalize_observations=normalize_observations,
                standardize_observations=standardize_observations,
        )


    def reset(self):
        """Reset and get the first state.."""
        super(SingleActTargetFusionEnv, self).reset()
        last_obs = np.mean(self.state_log[:, -self.ob_smooth_amt:], axis=1)
        slopes = (last_obs - np.mean(self.state_log[:, :self.ob_smooth_amt], axis=1)) / self.state_log.shape[1]
        return np.append(last_obs, slopes)


    def step(self, action):
        """Take step in the environment."""
        # Unnormalize action.
        action = np.clip(action, -1, 1)
        action = float(2500000 * (action + 1) / 2)
        # Create constant actions for each of the beams.
        true_act = action * np.ones(shape=(
                self.act_log.shape[0],
                self.dyn_time_scale[1]),
        )
        s, r, d, info = super(SingleActTargetFusionEnv, self).step(true_act)
        # Transform observation into last observations + slopes.
        last_obs = np.mean(s[:, -self.ob_smooth_amt:], axis=1)
        slopes = (last_obs - np.mean(s[:, :self.ob_smooth_amt], axis=1)) / s.shape[1]
        nxt_state = np.append(last_obs, slopes)
        return nxt_state, r, d, info

    def _get_reward(self):
        """Calculate the current reward for after a step has been made."""
        beta_loss = np.abs((self.beta_target - self.state_log[self.beta_idx, -1]))
        exp_term = np.exp(self.rew_coefs[1] * (self.stab_log[-1] - 0.5))
        dis_loss = self.rew_coefs[0] * (exp_term / (1 + exp_term))
        rew_info = {
                'BetaLoss': beta_loss,
                'TearPenalty': dis_loss,
                'Tearability': self.stab_log[-1],
        }
        if self.pos_reward:
            rew = 3 - beta_loss
        else:
            rew = -beta_loss - dis_loss
        return rew, rew_info

class SISOTargetFusionEnv(BaseFusionEnv):
    def __init__(
            self,
            signal_list,
            dyn_model,
            dis_model,
            init_dir,
            horizon,
            dyn_time_scale=(200,200),
            dis_time_scale=(100,250),
            init_at_start_only=False,
            num_cached_starts=None,
            preloaded_inits=None,
            normalize_observations=False,
            standardize_observations=True,
            beta_target=1.5,
            rew_coefs=(9, 10),
            pos_reward=False,
    ):
        """
        Args:
            signal_list: (list) of signals in order desired
            dyn_model: Model for the dynamics.
            dis_model: Model for the disruption.
            inits: List of tuples where first item is np array of init states
                and second is np array of actions.
            horizon: Length of time to run for.
            dyn_time_scale: Time scales for dynamics model where the first
                number is amount of history for input and second is the
                number of ms in the future to predict.
            dis_time_scale: Time scales for disruption model where the first
                number is amount of history for input and second is the
                number of ms in the future to predict.
            beta_target: Target beta to reach.
            rew_coefs: Coefficients for reward in form of (a, b) where the
                reward is (-beta_L2 - a * sigmoid(b * (disrupt_prob - 0.5))).
        """
        # Have normalized action space but this will really be between
        # 0 and 2500000.
        self.beta_target = beta_target
        self.rew_coefs = rew_coefs
        self.pos_reward = pos_reward
        self.action_space = spaces.Box(
                low=-1,
                high=1,
                shape=(1,),
                dtype=np.float32,
        )
        # Observation space is the last recorded signal + slope between this
        # and last prediction.
        low, high = SIGNAL_BOUNDS['efsbetan']
        self.observation_space = spaces.Box(
                low=np.asarray([low]),
                high=np.asarray([high]),
                dtype=np.float32,
        )
        super(SISOTargetFusionEnv, self).__init__(
                signal_list,
                dyn_model,
                dis_model,
                init_dir,
                horizon,
                dyn_time_scale=dyn_time_scale,
                dis_time_scale=dis_time_scale,
                init_at_start_only=init_at_start_only,
                num_cached_starts=num_cached_starts,
                preloaded_inits=preloaded_inits,
                normalize_observations=normalize_observations,
                standardize_observations=standardize_observations,
        )


    def reset(self):
        """Reset and get the first state.."""
        super(SISOTargetFusionEnv, self).reset()
        return np.asarray([self.state_log[self.beta_idx, -1]])


    def step(self, action):
        """Take step in the environment."""
        # Unnormalize action.
        action = np.clip(action, -1, 1)
        action = float(2500000 * (action + 1) / 2)
        # Create constant actions for each of the beams.
        true_act = action * np.ones(shape=(
                self.act_log.shape[0],
                self.dyn_time_scale[1]),
        )
        s, r, d, info = super(SISOTargetFusionEnv, self).step(true_act)
        # Transform observation into last observations + slopes.
        nxt_state = np.asarray([s[self.beta_idx, -1]])
        return nxt_state, r, d, info

    def _get_reward(self):
        """Calculate the current reward for after a step has been made."""
        beta_loss = np.abs((self.beta_target - self.state_log[self.beta_idx, -1]))
        exp_term = np.exp(self.rew_coefs[1] * (self.stab_log[-1] - 0.5))
        dis_loss = self.rew_coefs[0] * (exp_term / (1 + exp_term))
        rew_info = {
                'BetaLoss': beta_loss,
                'TearPenalty': dis_loss,
                'Tearability': self.stab_log[-1],
        }
        if self.pos_reward:
            rew = 3 - beta_loss
        else:
            rew = -beta_loss - dis_loss
        return rew, rew_info

class SmallTargetFusionEnv(BaseFusionEnv):
    """Environment where observation is average beta_n and average probability
    of disrupting."""
    def __init__(
            self,
            signal_list,
            dyn_model,
            dis_model,
            init_dir,
            horizon,
            dyn_time_scale=(200,200),
            dis_time_scale=(100,250),
            init_at_start_only=False,
            num_cached_starts=None,
            preloaded_inits=None,
            normalize_observations=False,
            standardize_observations=True,
            beta_target=1.5,
            rew_coefs=(9, 10),
    ):
        """
        Args:
            signal_list: (list) of signals in order desired
            dyn_model: Model for the dynamics.
            dis_model: Model for the disruption.
            inits: List of tuples where first item is np array of init states
                and second is np array of actions.
            horizon: Length of time to run for.
            dyn_time_scale: Time scales for dynamics model where the first
                number is amount of history for input and second is the
                number of ms in the future to predict.
            dis_time_scale: Time scales for disruption model where the first
                number is amount of history for input and second is the
                number of ms in the future to predict.
            beta_target: Target beta to reach.
            rew_coefs: Coefficients for reward in form of (a, b) where the
                reward is (-beta_L2 - a * sigmoid(b * (disrupt_prob - 0.5))).
        """
        # Have normalized action space but this will really be between
        # 0 and 2500000.
        self.beta_target = beta_target
        self.rew_coefs = rew_coefs
        self.action_space = spaces.Box(
                low=-1,
                high=1,
                shape=(1,),
                dtype=np.float32,
        )
        # Observation space is the last recorded signal + slope between this
        # and last prediction.
        low, high = SIGNAL_BOUNDS['efsbetan']
        self.observation_space = spaces.Box(
                low=np.asarray([low, low, 0]),
                high=np.asarray([high, high, 1]),
                dtype=np.float32,
        )
        super(SmallTargetFusionEnv, self).__init__(
                signal_list,
                dyn_model,
                dis_model,
                init_dir,
                horizon,
                dyn_time_scale=dyn_time_scale,
                dis_time_scale=dis_time_scale,
                init_at_start_only=init_at_start_only,
                num_cached_starts=num_cached_starts,
                preloaded_inits=preloaded_inits,
                normalize_observations=normalize_observations,
                standardize_observations=standardize_observations,
        )


    def reset(self):
        """Reset and get the first state.."""
        super(SmallTargetFusionEnv, self).reset()
        return np.asarray([
            np.mean(self.state_log[self.beta_idx, :self.state_size]),
            self.state_log[self.beta_idx, -1] - self.state_log[self.beta_idx, 0],
            np.mean(self.stab_log[-self.state_size:]),
        ])


    def step(self, action):
        """Take step in the environment."""
        # Unnormalize action.
        action = np.clip(action, -1, 1)
        action = float(2500000 * (action + 1) / 2)
        # Create constant actions for each of the beams.
        true_act = action * np.ones(shape=(
                self.act_log.shape[0],
                self.dyn_time_scale[1]),
        )
        s, r, d, info = super(SmallTargetFusionEnv, self).step(true_act)
        # Transform observation into last observations + slopes.
        nxt_state = np.asarray([
            np.mean(s[self.beta_idx]),
            s[self.beta_idx, -1] - s[self.beta_idx, 0],
            np.mean(self.stab_log[-self.state_size:]),
        ])
        raw_obs = info['raw_observation']
        info['raw_observation'] = np.asarray([
            np.mean(raw_obs[self.beta_idx]),
            raw_obs[self.beta_idx, -1] - raw_obs[self.beta_idx, 0],
            np.mean(self.stab_log[-self.state_size:]),
        ])
        return nxt_state, r, d, info

    def _get_reward(self):
        """Calculate the current reward for after a step has been made."""
        beta_loss = np.abs((self.beta_target - self.state_log[self.beta_idx, -1]))
        exp_term = np.exp(self.rew_coefs[1] * (self.stab_log[-1] - 0.5))
        dis_loss = self.rew_coefs[0] * (exp_term / (1 + exp_term))
        rew_info = {
                'BetaLoss': beta_loss,
                'TearPenalty': dis_loss,
                'Tearability': self.stab_log[-1],
        }
        return -beta_loss - dis_loss, rew_info


class PIDFusionEnv(SISOTargetFusionEnv):
    """Environment where observation is average beta_N and the PID components."""
    def __init__(
            self,
            signal_list,
            dyn_model,
            dis_model,
            init_dir,
            horizon,
            dyn_time_scale=(200,200),
            dis_time_scale=(100,250),
            init_at_start_only=False,
            num_cached_starts=None,
            preloaded_inits=None,
            normalize_observations=False,
            standardize_observations=True,
            beta_target=2,
            rew_coefs=(0, 10),
            state_space='b,p,i,d',
            pos_reward=False,
    ):
        """
        Args:
            signal_list: (list) of signals in order desired
            dyn_model: Model for the dynamics.
            dis_model: Model for the disruption.
            inits: List of tuples where first item is np array of init states
                and second is np array of actions.
            horizon: Length of time to run for.
            dyn_time_scale: Time scales for dynamics model where the first
                number is amount of history for input and second is the
                number of ms in the future to predict.
            dis_time_scale: Time scales for disruption model where the first
                number is amount of history for input and second is the
                number of ms in the future to predict.
            beta_target: Target beta to reach.
            rew_coefs: Coefficients for reward in form of (a, b) where the
                reward is (-beta_L2 - a * sigmoid(b * (disrupt_prob - 0.5))).
        """
        self.state_space = np.array([
            'b' in state_space,
            'p' in state_space,
            'i' in state_space,
            'd' in state_space,
        ]).flatten()
        self.pid = PID(1, 1, 1,
                       setpoint=beta_target)
        super(PIDFusionEnv, self).__init__(
                signal_list,
                dyn_model,
                dis_model,
                init_dir,
                horizon,
                dyn_time_scale=dyn_time_scale,
                dis_time_scale=dis_time_scale,
                init_at_start_only=init_at_start_only,
                num_cached_starts=num_cached_starts,
                preloaded_inits=preloaded_inits,
                normalize_observations=normalize_observations,
                standardize_observations=standardize_observations,
                beta_target=beta_target,
                rew_coefs=rew_coefs,
                pos_reward=pos_reward,
        )
        low, high = SIGNAL_BOUNDS['efsbetan']
        obs_dim = np.sum(self.state_space)
        self.observation_space = spaces.Box(
                low=np.asarray([low for _ in range(obs_dim)]),
                high=np.asarray([high for _ in range(obs_dim)]),
                dtype=np.float32,
        )


    def reset(self):
        """Reset and get the first state.."""
        self.pid.reset()
        obs = super(PIDFusionEnv, self).reset()
        obs = self._append_pid(obs)
        return obs[self.state_space]

    def set_target(self, target):
        self.pid = PID(1, 1, 1,
                       setpoint=target)
        self.beta_target = target

    def step(self, action):
        """Take step in the environment."""
        nxt, rew, done, info = super(PIDFusionEnv, self).step(action)
        nxt = self._append_pid(nxt)
        nxt = nxt[self.state_space]
        return nxt, rew, done, info

    def _append_pid(self, obs):
        self.pid(float(obs), dt=self.dyn_time_scale[1] / 1000)
        p, i, d = self.pid.components
        return np.append(obs, np.array([p, i, d]))


class MDPPIDFusionEnv(PIDFusionEnv):
    """Environment where observation is average beta_N and the PID components."""

    def set_hardcoded_signals(self, path):
        """Load in that stuff here."""
        self.hardcoded_signals = np.load(path)

    def step(self, action):
        # Rewrite history so all other signals are the same.
        for sigidx, sig in enumerate(self.hardcoded_signals):
            if sigidx == 3:
                continue
            self.state_log[sigidx, -len(sig):] = sig

        return super().step(action)


class FullPIDFusionEnv(SingleActTargetFusionEnv):
    """Environment where observation is average beta_N and the PID components."""
    def __init__(
            self,
            signal_list,
            dyn_model,
            dis_model,
            init_dir,
            horizon,
            dyn_time_scale=(200,200),
            dis_time_scale=(100,250),
            init_at_start_only=False,
            num_cached_starts=None,
            preloaded_inits=None,
            normalize_observations=False,
            standardize_observations=True,
            beta_target=2,
            rew_coefs=(0, 10),
            state_space='b,p,i,d',
            pos_reward=False,
    ):
        """
        Args:
            signal_list: (list) of signals in order desired
            dyn_model: Model for the dynamics.
            dis_model: Model for the disruption.
            inits: List of tuples where first item is np array of init states
                and second is np array of actions.
            horizon: Length of time to run for.
            dyn_time_scale: Time scales for dynamics model where the first
                number is amount of history for input and second is the
                number of ms in the future to predict.
            dis_time_scale: Time scales for disruption model where the first
                number is amount of history for input and second is the
                number of ms in the future to predict.
            beta_target: Target beta to reach.
            rew_coefs: Coefficients for reward in form of (a, b) where the
                reward is (-beta_L2 - a * sigmoid(b * (disrupt_prob - 0.5))).
        """
        self.pid = PID(1, 1, 1,
                       setpoint=beta_target)
        super(FullPIDFusionEnv, self).__init__(
                signal_list,
                dyn_model,
                dis_model,
                init_dir,
                horizon,
                dyn_time_scale=dyn_time_scale,
                dis_time_scale=dis_time_scale,
                init_at_start_only=init_at_start_only,
                num_cached_starts=num_cached_starts,
                preloaded_inits=preloaded_inits,
                normalize_observations=normalize_observations,
                standardize_observations=standardize_observations,
                beta_target=beta_target,
                rew_coefs=rew_coefs,
                pos_reward=pos_reward,
        )
        low, high = SIGNAL_BOUNDS['efsbetan']
        self.observation_space = spaces.Box(
                low=np.asarray([low for _ in range(13)]),
                high=np.asarray([high for _ in range(13)]),
                dtype=np.float32,
        )


    def reset(self):
        """Reset and get the first state.."""
        self.pid.reset()
        super(FullPIDFusionEnv, self).reset()
        return self._get_obs()

    def set_target(self, target):
        self.pid = PID(1, 1, 1,
                       setpoint=target)
        self.beta_target = target

    def step(self, action):
        """Take step in the environment."""
        _, rew, done, info = super(FullPIDFusionEnv, self).step(action)
        nxt = self._get_obs()
        return nxt, rew, done, info

    def _get_obs(self):
        obs = np.mean(self.state_log[:, -self.state_size:], axis=1)
        self.pid(float(obs[self.beta_idx]), dt=self.dyn_time_scale[1] / 1000)
        p, i, d = self.pid.components
        return np.append(obs, np.array([p, i, d]))
