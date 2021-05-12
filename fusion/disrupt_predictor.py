"""
Gym component for predicting disruptions.

Author: Ian Char
Date: 5/8/2020
"""

import numpy as np
import pickle as pkl
from catboost import CatBoostClassifier

class DisruptionPrediction(object):

    def predict(self, state):
        raise NotImplementedError('Abstract Method')

class CatBoostDisruptionPredictor(object):

    def __init__(self,
                 cb_model,
                 data_min_maxs,
                 cb_columns,
                 data_in_columns,
                 dt=1,
    ):
        self._cb_model = cb_model
        self._data_min_maxs = [np.asarray(data_min_maxs[i][0])
                               for i in range(2)]
        self._data_spread = self._data_min_maxs[1] - self._data_min_maxs[0]
        self._dt = dt
        self._data_in_columns = data_in_columns
        self._feature_cols, self._mapping = self._preprocess_columns(
                cb_columns,
                data_in_columns,
        )

    def predict(self, state):
        """Given the state, return probability of disruption in 250 ms."""
        # Preprocess the state.
        model_in = self._get_features(state)
        # Pass state through the model and return.
        return self._cb_model.predict(
                model_in,
                prediction_type='Probability',
        )[1]

    def multi_predict(self, states):
        """Given the state, return probability of disruption in 250 ms."""
        # Preprocess the state.
        model_in = np.vstack([self._get_features(s) for s in states])
        # Pass state through the model and return.
        return self._cb_model.predict(
                model_in,
                prediction_type='Probability',
        )[:, 1]

    def _preprocess_columns(self, cb_columns, data_in_columns):
        feature_cols = []
        for sig in data_in_columns:
            for d_type in ['avg', 'var', 'slope']:
                for j in range(1, 4):
                    for i in range(1, j + 1):
                        feature_cols.append(
                                '%s-%s-%d/%d' % (sig, d_type, i, j)
                        )
        permutation_mapping = [feature_cols.index(cb) for cb in cb_columns]
        return feature_cols, permutation_mapping

    def _get_features(self, state):
        """Get the features needed for the Catboost Model."""
        # Figure out where halves and thirds are.
        state_dim = state.shape[0]
        half_pivot = state_dim // 2
        third_pivots = [state_dim // 3, 2 * (state_dim // 3)]
        # Generate the features.
        features = []
        for f_idx, fname in enumerate(self._data_in_columns):
            features += self._get_subframe_features(state[:, f_idx])
            features += self._get_subframe_features(state[:half_pivot, f_idx])
            features += self._get_subframe_features(state[half_pivot:, f_idx])
            features += self._get_subframe_features(
                    state[:third_pivots[0], f_idx]
            )
            features += self._get_subframe_features(
                    state[third_pivots[0]:third_pivots[1], f_idx]
            )
            features += self._get_subframe_features(
                    state[third_pivots[1]:, f_idx]
            )
        features = np.asarray(features)
        features = features[self._mapping]
        features = (features - self._data_min_maxs[0]) / self._data_spread
        return features

    def _get_subframe_features(self, subframe):
        mean = np.mean(subframe)
        variance = np.var(subframe)
        subtimes = self._dt * np.arange(len(subframe))
        slope = np.polyfit(subtimes, subframe, 1)[0]
        return [mean, variance, slope]

def load_cb_from_files(cb_model_path,
                       data_min_max_path,
                       cb_column_path,
                       data_in_columns,
                       dt=0.001,
):
    cb_model = CatBoostClassifier(verbose=False,
                                  depth=15,
                                  l2_leaf_reg=3,
                                  learning_rate=0.3)
    cb_model.load_model(cb_model_path)
    with open(data_min_max_path, 'rb') as f:
        data_min_max = pkl.load(f)
    with open(cb_column_path, 'rb') as f:
        cb_columns = pkl.load(f)[2:]
    return CatBoostDisruptionPredictor(cb_model,
                                       data_min_max,
                                       cb_columns,
                                       data_in_columns,
                                       dt=dt)

if __name__ == '__main__':
    import pudb; pudb.set_trace()
    load_cb_from_files('notebooks/stability/models/catboost/model_5-8/model_5-8-2020.cbm',
                       'notebooks/stability/models/catboost/model_5-8/dranges.pkl',
                       'notebooks/stability/models/catboost/model_5-8/headers.pkl',
                       ['efsbetan', ''])
