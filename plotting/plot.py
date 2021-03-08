"""
Plotting functions.
"""
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plotting.baselines import BASELINES, get_raw_score


COLORS = ['red', 'green', 'orange', 'violet', 'blue']

def plot_from_df(
    df,
    task_name,
    top_baselines_to_plot=2,
    plot_baselines=[],
    save_path=None,
    show_plot=False,
):
    vless_task = task_name[:-3]
    sns.lineplot('Epoch', 'Returns/avg', hue='Algorithm', data=df)
    baselines = set(plot_baselines).union(
            set(get_top_k(vless_task, top_baselines_to_plot)))
    cidx = 0
    for b in baselines:
        score = get_raw_score(task_name, BASELINES[vless_task][b])
        plt.axhline(score, ls='--', label=b, color=COLORS[cidx % len(COLORS)])
        cidx += 1
    plt.legend()
    if save_path is not None:
        plt.savefig(save_path)
    if show_plot:
        plt.show()


def plot_from_stats_files(
        stats_dict,
        task_name,
        **kwargs
):
    df = None
    for algo_name, spaths in stats_dict.items():
        for spath in spaths:
            curr_df = load_stats_file(spath, algo_name)
            if df is None:
                df = curr_df
            else:
                df = df.append(curr_df)
    return plot_from_df(df, task_name, **kwargs)

def load_stats_file(stats_path, algo_name):
    df = pd.read_csv(stats_path)
    df.insert(0, 'Algorithm', [algo_name for _ in range(len(df))])
    return df

def get_top_k(task_name, k):
    scores = [v for v in BASELINES[task_name].values()]
    tops = np.argsort(scores)[::-1]
    return np.array([alg for alg in BASELINES[task_name].keys()])[tops[:k]]
