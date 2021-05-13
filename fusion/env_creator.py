import numpy as np
import os
import pickle as pkl

from fusion.fusion_env import BaseFusionEnv, SingleActTargetFusionEnv,\
        SISOTargetFusionEnv, SmallTargetFusionEnv, PIDFusionEnv,\
        MDPPIDFusionEnv
from fusion.disrupt_predictor import CatBoostDisruptionPredictor,\
    load_cb_from_files
from fusion.state_predictor_frame import make_signal_namespace,\
        CatBoostStatePredictor

def create_base_env(
        state_model_dir,
        disrupt_model_dir,
        shot_dir,
        dyn_time_scale=(200, 200),
        **kwargs
):
    headers, dis_model, dyn_model, all_shots = load_in_all_info(
            state_model_dir,
            disrupt_model_dir,
            shot_dir,
            dyn_time_scale=dyn_time_scale,
    )
    return BaseFusionEnv(
            headers[:10],
            dyn_model,
            dis_model,
            shot_dir,
            dyn_time_scale=dyn_time_scale,
            **kwargs
    )

def create_single_act_target_env(
        state_model_dir,
        disrupt_model_dir,
        shot_dir,
        dyn_time_scale=(200, 200),
        **kwargs
):
    headers, dis_model, dyn_model, all_shots = load_in_all_info(
            state_model_dir,
            disrupt_model_dir,
            shot_dir,
            dyn_time_scale=dyn_time_scale
    )
    return SingleActTargetFusionEnv(
            headers[:10],
            dyn_model,
            dis_model,
            shot_dir,
            dyn_time_scale=dyn_time_scale,
            **kwargs
    )

def create_siso_target_env(
        state_model_dir,
        disrupt_model_dir,
        shot_dir,
        dyn_time_scale=(200, 200),
        **kwargs
):
    headers, dis_model, dyn_model, all_shots = load_in_all_info(
            state_model_dir,
            disrupt_model_dir,
            shot_dir,
            dyn_time_scale=dyn_time_scale
    )
    return SISOTargetFusionEnv(
            headers[:10],
            dyn_model,
            dis_model,
            shot_dir,
            dyn_time_scale=dyn_time_scale,
            **kwargs
    )

def create_pid_env(
        state_model_dir,
        disrupt_model_dir,
        shot_dir,
        dyn_time_scale=(200, 200),
        **kwargs
):
    headers, dis_model, dyn_model, all_shots = load_in_all_info(
            state_model_dir,
            disrupt_model_dir,
            shot_dir,
            dyn_time_scale=dyn_time_scale
    )
    return PIDFusionEnv(
            headers[:10],
            dyn_model,
            dis_model,
            shot_dir,
            dyn_time_scale=dyn_time_scale,
            **kwargs
    )

def create_mdp_pid_env(
        state_model_dir,
        disrupt_model_dir,
        shot_dir,
        dyn_time_scale=(200, 200),
        **kwargs
):
    headers, dis_model, dyn_model, all_shots = load_in_all_info(
            state_model_dir,
            disrupt_model_dir,
            shot_dir,
            dyn_time_scale=dyn_time_scale
    )
    env = MDPPIDFusionEnv(
            headers[:10],
            dyn_model,
            dis_model,
            shot_dir,
            dyn_time_scale=dyn_time_scale,
            **kwargs
    )
    env.set_hardcoded_signals('hardcoded_state.npy')
    return env

def create_small_target_env(
        state_model_dir,
        disrupt_model_dir,
        shot_dir,
        dyn_time_scale=(200, 200),
        **kwargs
):
    headers, dis_model, dyn_model, all_shots = load_in_all_info(
            state_model_dir,
            disrupt_model_dir,
            shot_dir,
            dyn_time_scale=dyn_time_scale
    )
    return SmallTargetFusionEnv(
            headers[:10],
            dyn_model,
            dis_model,
            shot_dir,
            dyn_time_scale=dyn_time_scale,
            **kwargs
    )

def load_in_all_info(
        state_model_dir,
        disrupt_model_dir,
        shot_dir,
        dyn_time_scale=(200, 200),

):
    # Load in the start state shots.
    all_shots = []
    for fname in os.listdir(shot_dir):
        fpath = os.path.join(shot_dir, fname)
        if 'npy' in fname:
            all_shots.append(np.load(fpath))
        else:
            with open(fpath, 'rb') as f:
                headers = pkl.load(f)
    # Load in the disruptor model.
    dis_model = load_cb_from_files(
        # TODO: Possibly change model file name.
        os.path.join(disrupt_model_dir, 'model.cbm'),
        os.path.join(disrupt_model_dir, 'dranges.pkl'),
        os.path.join(disrupt_model_dir, 'headers.pkl'),
        headers[:10],
    )
    # Load in the dynamics model.
    num_actions = 8
    hist_len, pred_len = dyn_time_scale
    signal_namespace = make_signal_namespace(headers[:10], state_model_dir)
    dyn_model = CatBoostStatePredictor(
            input_signal_order=headers,
            state_signals_used=headers,
            action_signals_used=headers[10:],
            signal_namespace=signal_namespace,
            num_actions=num_actions,
            history_len=hist_len,
            pred_len=pred_len,
    )
    return headers, dis_model, dyn_model, all_shots
