import ray
from ray import tune
from ray.tune.suggest.ax import AxSearch
from ray.tune.schedulers import ASHAScheduler
from functools import partial
from copy import deepcopy
from run import main, parse_arguments


def training_function(config, args):
    args = deepcopy(args)
    suffix_list = []
    for k, v in config.items():
        args.__setattr__(k, v)
        suffix_list.append(f"{k}: {v}")
    args.name = args.name + "_" + ",".join(suffix_list)
    args.__setattr__('hp_tune', True)
    main(args)


def hp_main(args):
    train_with_args = partial(training_function, args=args)
    ray.init(address="auto")
    asha_scheduler = ASHAScheduler(
        time_attr='training_iteration',
        metric='avg_return',
        mode='max',
        max_t=args.num_stitching_iters)
    ax_search = AxSearch(metric="avg_return")
    search_space = {
            'epsilon_planning': tune.uniform(0.05, 0.5),
            'planning_quantile': tune.uniform(0.4, 1.),
            'epsilon_neighbors': tune.uniform(1, 4),
            }
    analysis = tune.run(
            train_with_args,
            config=search_space,
            search_alg=ax_search,
            scheduler=asha_scheduler,
            num_samples=10,
            resources_per_trial={"cpu": 80})
    print("Best config: ", analysis.get_best_config(
                    metric="avg_return", mode="max"))
    df = analysis.results_df
    df.write_csv(f'hp_tuning_{args.name}')


if __name__ == '__main__':
    args = parse_arguments()
    hp_main(args)
