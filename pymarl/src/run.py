import datetime
import os
import pprint
import time
import threading
import torch as th
from types import SimpleNamespace as SN
from utils.logging import Logger
from utils.timehelper import time_left, time_str
from os.path import dirname, abspath

from learners import REGISTRY as le_REGISTRY
from runners import REGISTRY as r_REGISTRY

from runners import EpisodeRunner

from controllers import REGISTRY as mac_REGISTRY
from components.episode_buffer import ReplayBuffer
from components.episode_buffer import Prioritized_ReplayBuffer
from components.transforms import OneHot
from utils.torch_utils import to_cuda
from modules.agents.LRN_KNN import LRU_KNN
from components.episodic_memory_buffer import Episodic_memory_buffer

import numpy as np
import copy as cp
import random

def run(_run, _config, _log):

    # check args sanity
    _config = args_sanity_check(_config, _log)

    args = SN(**_config)
    args.device = "cuda" if args.use_cuda else "cpu"
    set_device = os.getenv('SET_DEVICE')
    if args.use_cuda and set_device != '-1':
        if set_device is None:
            args.device = "cuda"
        else:
            args.device = f"cuda:{set_device}"
    else:
        args.device = "cpu"

    # setup loggers
    logger = Logger(_log)

    _log.info("Experiment Parameters:")
    experiment_params = pprint.pformat(_config,
                                       indent=4,
                                       width=1)
    _log.info("\n\n" + experiment_params + "\n")

    # configure tensorboard logger
    unique_token = "{}__{}".format(args.name, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    args.unique_token = unique_token
    if args.use_tensorboard:
        tb_logs_direc = os.path.join(dirname(dirname(abspath(__file__))), "results", "tb_logs", args.env,
                                     args.env_args['map_name'])
        tb_exp_direc = os.path.join(tb_logs_direc, "{}").format(unique_token)
        logger.setup_tb(tb_exp_direc)
        tb_info_get = os.path.join("results", "tb_logs", args.env, args.env_args['map_name'], "{}").format(unique_token)
        _log.info("saving tb_logs to " + tb_info_get)

    # sacred is on by default
    logger.setup_sacred(_run)

    # Run and train
    run_sequential(args=args, logger=logger)

    # Clean up after finishing
    print("Exiting Main")

    print("Stopping all threads")
    for t in threading.enumerate():
        if t.name != "MainThread":
            print("Thread {} is alive! Is daemon: {}".format(t.name, t.daemon))
            t.join(timeout=1)
            print("Thread joined")

    print("Exiting script")

    # Making sure framework really exits
    # os._exit(os.EX_OK)


def evaluate_sequential(args, runner):

    for _ in range(args.test_nepisode):
        runner.run(test_mode=True)

    if args.save_replay:
        runner.save_replay()

    runner.close_env()


def save_one_buffer(args, save_buffer, env_name, from_start=False):
    x_env_name = env_name
    if from_start:
        x_env_name += '_from_start/'
    path_name = '../../buffer/' + x_env_name + '/buffer_' + str(args.save_buffer_id) + '/'
    if os.path.exists(path_name):
        random_name = '../../buffer/' + x_env_name + '/buffer_' + str(random.randint(10, 1000)) + '/'
        os.rename(path_name, random_name)
    if not os.path.exists(path_name):
        os.makedirs(path_name)
    save_buffer.save(path_name)


def run_sequential(args, logger):

    # Init runner so we can get env info
    runner : EpisodeRunner = r_REGISTRY[args.runner](args=args, logger=logger)

    # Set up schemes and groups here
    env_info = runner.get_env_info()
    args.episode_limit = env_info["episode_limit"]
    args.n_agents = env_info["n_agents"]
    args.n_actions = env_info["n_actions"]
    args.state_shape = env_info["state_shape"]
    args.unit_dim = env_info["unit_dim"]

    args.obs_shape = env_info["obs_shape"]
    args.subgoal_shape = env_info["subgoal_shape"]
    args.Goal_shape = env_info["Goal_shape"]

    # Default/Base scheme
    scheme = {
        "state": {"vshape": env_info["state_shape"]},  # 92
        "obs": {"vshape": env_info["obs_shape"], "group": "agents"}, # 46
        "Goal": {"vshape": env_info["Goal_shape"], "group": "agents"},
        "subgoal": {"vshape": env_info["subgoal_shape"], "group": "agents"}, # 23
        "actions": {"vshape": (1,), "group": "agents", "dtype": th.long}, # 2
        "avail_actions": {"vshape": (env_info["n_actions"],), "group": "agents", "dtype": th.int},
        "reward": {"vshape": (1,)},
        "low_reward": {"vshape": (env_info["n_agents"],)},
        "terminated": {"vshape": (1,), "dtype": th.uint8},
    }
    groups = {
        "agents": args.n_agents
    }
    preprocess = {
        "actions": ("actions_onehot", [OneHot(out_dim=args.n_actions)])
    }

    env_name = args.env
    # is_prioritized_buffer: true
    if args.is_prioritized_buffer:
        buffer = Prioritized_ReplayBuffer(scheme, groups, args.buffer_size, env_info["episode_limit"] + 1,
                                        args.prioritized_buffer_alpha,
                                        preprocess=preprocess,
                                        device="cpu" if args.buffer_cpu_only else args.device)
        high_buffer = Prioritized_ReplayBuffer(scheme, groups, args.buffer_size, env_info["episode_limit"] + 1,
                                        args.prioritized_buffer_alpha,
                                        preprocess=preprocess,
                                        device="cpu" if args.buffer_cpu_only else args.device)
    
    # is_save_buffer: false
    if args.is_save_buffer:
        save_buffer = ReplayBuffer(scheme, groups, args.save_buffer_size, env_info["episode_limit"] + 1,
                                   args.burn_in_period,
                                   preprocess=preprocess,
                                   device="cpu" if args.buffer_cpu_only else args.device)

    # gridworld: is_batch_rl: false
    if args.is_batch_rl:
        assert (args.is_save_buffer == False)
        x_env_name = env_name
        if args.is_from_start:
            x_env_name += '_from_start/'
        path_name = '../../buffer/' + x_env_name + '/buffer_' + str(args.load_buffer_id) + '/'
        assert (os.path.exists(path_name) == True)
        buffer.load(path_name)

    if getattr(args, "use_emdqn", False):
        ec_buffer=Episodic_memory_buffer(args,scheme)

    # Setup multiagent controller here
    # high level: hlevel_mac-->hlevel_controller-->HLevelMac
    mac = mac_REGISTRY[args.mac](buffer.scheme, groups, args)

    # low level: llevel_mac-->level_controller
    action_mac = mac_REGISTRY[args.action_mac](buffer.scheme, groups, args)

    # Give runner the scheme
    # runner = episode
    runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac, action_mac=action_mac)
    # Learner
    learner = le_REGISTRY[args.learner](mac, buffer.scheme, logger, args, groups=groups)

    action_learner = le_REGISTRY[args.actionlearner](action_mac, buffer.scheme, logger, args, groups=groups)

    if hasattr(args, "save_buffer") and args.save_buffer:
        learner.buffer = buffer

    if args.use_cuda:
        learner.cuda()

    
    # start training
    episode = 0
    last_test_T = -args.test_interval - 1
    last_log_T = 0
    model_save_time = 0

    start_time = time.time()
    last_time = start_time
    Arr = Sum = 0
    _p, _sum = 0, 0

    logger.console_logger.info("Beginning training for {} timesteps".format(args.t_max))

    while runner.t_env <= args.t_max:

        # is_bach_rl: false
        if not args.is_batch_rl:
            # Run for a whole episode at a time
            # batch 带上 goal
            episode_batch, _p, _sum = runner.run(test_mode=False)
            Arr += _p
            Sum += _sum
            if getattr(args, "use_emdqn", False):
                ec_buffer.update_ec(episode_batch)
            buffer.insert_episode_batch(episode_batch)
            high_buffer.insert_episode_batch(episode_batch)

            # is_save_buffer: false
            if args.is_save_buffer:
                save_buffer.insert_episode_batch(episode_batch)
                if save_buffer.is_from_start and save_buffer.episodes_in_buffer == save_buffer.buffer_size:
                    save_buffer.is_from_start = False
                    save_one_buffer(args, save_buffer, env_name, from_start=True)
                    break
                if save_buffer.buffer_index % args.save_buffer_interval == 0:
                    print('current episodes_in_buffer: ', save_buffer.episodes_in_buffer)

        # num_circle: 1
        if runner.t_env >= args.train_delay_steps:
            for _ in range(args.num_circle):
                if buffer.can_sample(args.batch_size):
                    if args.is_prioritized_buffer:
                        sample_indices, episode_sample = buffer.sample(args.batch_size)
                        # print('sample_indiced: ', sample_indices)
                    else:
                        episode_sample = buffer.sample(args.batch_size)

                    # is_batch_rl: false
                    if args.is_batch_rl:
                        runner.t_env += int(th.sum(episode_sample['filled']).cpu().clone().detach().numpy()) // args.batch_size

                    # Truncate batch to only filled timesteps
                    max_ep_t = episode_sample.max_t_filled()
                    episode_sample = episode_sample[:, :max_ep_t]

                    if episode_sample.device != args.device:
                        episode_sample.to(args.device)

                    # gridworld : use_emdqn: false
                    if args.is_prioritized_buffer:
                        if getattr(args, "use_emdqn", False):
                            td_error = learner.train(episode_sample, runner.t_env, episode, ec_buffer=ec_buffer)
                            ltd_error = action_learner.train(episode_sample, runner.t_env, episode, ec_buffer=ec_buffer)
                        else:
                            td_error = learner.train(episode_sample, runner.t_env, episode)
                            ltd_error = action_learner.train(episode_sample, runner.t_env, episode)
                            buffer.update_priority(sample_indices, td_error)
                    else:
                        if getattr(args, "use_emdqn", False):
                            td_error = learner.train(episode_sample, runner.t_env, episode, ec_buffer=ec_buffer)
                            ltd_error = action_learner.train(episode_sample, runner.t_env, episode, ec_buffer=ec_buffer)
                        else:
                            learner.train(episode_sample, runner.t_env, episode)
                            ltd_error = action_learner.train(episode_sample, runner.t_env, episode)
                    # print(episode_sample['goals'][-1,-1])

            # Execute test runs once in a while
            n_test_runs = max(1, args.test_nepisode // runner.batch_size)
            if (runner.t_env - last_test_T) / args.test_interval >= 1.0 :

                # print(f'Arrive Goal: {Arr}/{Sum}')
                # Arr = Sum = 0
                logger.console_logger.info("t_env: {} / {}".format(runner.t_env, args.t_max))
                logger.console_logger.info("Estimated time left: {}. Time passed: {}".format(
                    time_left(last_time, last_test_T, runner.t_env, args.t_max), time_str(time.time() - start_time)))
                last_time = time.time()
                logger.log_stat("num_circle", args.num_circle, runner.t_env)
                # print('TEST n runs', n_test_runs)
                last_test_T = runner.t_env
                for _ in range(n_test_runs):
                    episode_sample, _p, _sum = runner.run(test_mode=True)
                    g1r = th.max(episode_sample["subgoal"][0, :, 0, :11], dim=-1)[1]
                    g1c = th.max(episode_sample["subgoal"][0, :, 0, 11:], dim=-1)[1]
                    g2r = th.max(episode_sample["subgoal"][0, :, 1, :11], dim=-1)[1]
                    g2c = th.max(episode_sample["subgoal"][0, :, 1, 11:], dim=-1)[1]
                    print(f'low_reward: {episode_sample["low_reward"]}')
                    print(f'first obs {episode_sample["obs"][0,0,:,:23]}')
                    print(f'last obs {episode_sample["obs"][0,-1,:,:23]}')
                    # print(f'goals {th.cat([g1r, g1c, g2r, g2c], dim=-1).reshape(-1, 4)}')
                    print(f'actions {episode_sample["actions"]}')
                    # print(f'ext reward {episode_sample["reward"]}')
                    # print(f'terminated {episode_sample["terminated"][0]}')
                    if args.mac == "offline_mac":
                        max_ep_t = episode_sample.max_t_filled()
                        episode_sample = episode_sample[:, :max_ep_t]

                        if episode_sample.device != args.device:
                            episode_sample.to(args.device)
                        learner.train(episode_sample, runner.t_env, episode, show_v=True)
                        action_learner.train(episode_sample, runner.t_env, episode, show_v=True)


            if args.save_model and (runner.t_env - model_save_time >= args.save_model_interval or model_save_time == 0):
                model_save_time = runner.t_env
                save_path = os.path.join(args.local_results_path, "models", args.unique_token, str(runner.t_env))
                actionmodel_path = os.path.join(args.local_results_path, "models", args.unique_token, str(runner.t_env), 'action')
                #"results/models/{}".format(unique_token)
                os.makedirs(save_path, exist_ok=True)
                if args.double_q:
                    os.makedirs(save_path + '_x', exist_ok=True)

                if args.learner == 'curiosity_learner' or args.learner == 'curiosity_learner_new' or args.learner == 'qplex_curiosity_learner'\
                        or args.learner == 'qplex_curiosity_rnd_learner' or args.learner =='qplex_rnd_history_curiosity_learner':
                    os.makedirs(save_path + '/mac/', exist_ok=True)
                    os.makedirs(save_path + '/extrinsic_mac/', exist_ok=True)
                    os.makedirs(save_path + '/predict_mac/', exist_ok=True)
                    if args.learner == 'curiosity_learner_new' or args.learner == 'qplex_curiosity_rnd_learner'or args.learner =='qplex_rnd_history_curiosity_learner':
                        os.makedirs(save_path + '/rnd_predict_mac/', exist_ok=True)
                        os.makedirs(save_path + '/rnd_target_mac/', exist_ok=True)

                if args.learner == 'rnd_learner' or args.learner == 'rnd_learner2' or args.learner =='qplex_rnd_learner'\
                        or args.learner =='qplex_rnd_history_learner' or args.learner =='qplex_rnd_emdqn_learner' :
                    os.makedirs(save_path + '/mac/', exist_ok=True)
                    os.makedirs(save_path + '/rnd_predict_mac/', exist_ok=True)
                    os.makedirs(save_path + '/rnd_target_mac/', exist_ok=True)
                if args.learner == 'qplex_curiosity_single_learner' or "qplex_curiosity_single_fast_learner":
                    os.makedirs(save_path + '/mac/', exist_ok=True)
                    os.makedirs(save_path + '/predict_mac/', exist_ok=True)
                    os.makedirs(save_path + '/soft_update_target_mac/', exist_ok=True)


                logger.console_logger.info("Saving models to {}".format(save_path))

                # learner should handle saving/loading -- delegate actor save/load to mac,
                # use appropriate filenames to do critics, optimizer states
                learner.save_models(save_path)
                action_learner.save_models(actionmodel_path)

            episode += args.batch_size_run * args.num_circle

            if (runner.t_env - last_log_T) >= args.log_interval:
                logger.log_stat("episode", episode, runner.t_env)
                logger.print_recent_stats()
                last_log_T = runner.t_env

    if args.is_save_buffer and save_buffer.is_from_start:
        save_buffer.is_from_start = False
        save_one_buffer(args, save_buffer, env_name, from_start=True)

    runner.close_env()
    logger.console_logger.info("Finished Training")


def args_sanity_check(config, _log):

    # set CUDA flags
    # config["use_cuda"] = True # Use cuda whenever possible!
    if config["use_cuda"] and not th.cuda.is_available():
        config["use_cuda"] = False
        _log.warning("CUDA flag use_cuda was switched OFF automatically because no CUDA devices are available!")

    if config["test_nepisode"] < config["batch_size_run"]:
        config["test_nepisode"] = config["batch_size_run"]
    else:
        config["test_nepisode"] = (config["test_nepisode"]//config["batch_size_run"]) * config["batch_size_run"]

    return config
