from marl_models.base_model import MARLModel
from environment.env import Env
from marl_models.utils import get_model, load_step_count
from train import train_on_policy, train_off_policy, train_random
from test import test_model
from utils.logger import Logger
from utils.plot_logs import generate_plots
import config
import torch
import numpy as np
import argparse
import warnings
import os
from datetime import datetime


def start_training(args: argparse.Namespace):
    timestamp: str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    print(f"\n🚀 Training started at {timestamp} for {args.num_episodes} episodes\n")

    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)
    env: Env = Env()
    model_name: str = config.MODEL.lower() #关键！从config.py读取
    model: MARLModel = get_model(model_name)

    # Setup logging directory
    model_log_dir = f"train_logs/{model_name}"
    if not os.path.exists(model_log_dir):
        os.makedirs(model_log_dir)

    logger: Logger = Logger(model_log_dir, timestamp)
    resume_training: bool = args.resume_path is not None
    #根据模型类型选择训练函数
    if resume_training:
        if args.config_path is None:
            raise ValueError("If --resume_path is provided, --config_path must also be provided.")
        else:
            logger.load_configs(args.config_path)  # Resume training with old config
    else:  # Fresh training
        if args.config_path is not None:
            warnings.warn("--config_path is ignored during training unless --resume_path is also provided.")
        logger.log_configs()

    total_step_count: int = 0  # for off policy models
    if resume_training:
        model.load(args.resume_path)
        total_step_count = load_step_count(args.resume_path)
        print(f"📥 Models loaded successfully from {args.resume_path}")
        print(f"📂 Resumed training from: {args.resume_path}\n")

    if model_name in ["maddpg", "attention_maddpg", "matd3", "attention_matd3", "masac", "attention_masac"]:
        train_off_policy(env, model, logger, args.num_episodes, total_step_count)
    elif model_name in ["mappo", "attention_mappo"]:
        train_on_policy(env, model, logger, args.num_episodes)
    else:  # "random"
        train_random(env, model, logger, args.num_episodes)

    # ========== 在这里添加能耗统计 ==========
    # 假设环境对象 env 中有 uavs 列表
    total_fly = sum(u.total_fly_energy for u in env.uavs)
    total_wpt = sum(u.total_wpt_tx_energy for u in env.uavs)
    total_compute = sum(u.total_compute_energy for u in env.uavs)
    total_energy = total_fly + total_wpt + total_compute

    print("\n========== Energy Breakdown ==========")
    print(f"Flight+Hover  : {total_fly:.2e} J ({total_fly / total_energy * 100:.1f}%)")
    print(f"WPT Transmit  : {total_wpt:.2e} J ({total_wpt / total_energy * 100:.1f}%)")
    print(f"Compute       : {total_compute:.2e} J ({total_compute / total_energy * 100:.1f}%)")
    print(f"Total         : {total_energy:.2e} J")
    print("=======================================")


    print("✅ Training Completed!\n")
    print("📊 Generating plots...")

    model_plot_dir = f"train_plots/{model_name}"
    #生成图表
    generate_plots(f"{model_log_dir}/log_data_{timestamp}.json", f"{model_plot_dir}/", "train", timestamp)


def start_testing(args: argparse.Namespace):
    timestamp: str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    print(f"\n🚀 Testing started at {timestamp} for {args.num_episodes} episodes\n")

    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)
    env: Env = Env()
    model_name: str = config.MODEL.lower()
    model: MARLModel = get_model(model_name)

    # Setup logging directory
    model_log_dir = f"test_logs/{model_name}"
    if not os.path.exists(model_log_dir):
        os.makedirs(model_log_dir)

    logger: Logger = Logger(model_log_dir, timestamp)
    logger.load_configs(args.config_path)

    model.load(args.model_path)
    print(f"📥 Models loaded successfully from {args.model_path}")

    test_model(env, model, logger, args.num_episodes)

    print("✅ Testing Completed!\n")
    print("📊 Generating plots...")
    generate_plots(f"test_logs/log_data_{timestamp}.json", "test_plots/", "test", timestamp, smoothing_window=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode", required=True)
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument("--num_episodes", type=int, required=True)
    train_parser = subparsers.add_parser("train", parents=[parent_parser])
    train_parser.add_argument("--resume_path", type=str, default=None)
    train_parser.add_argument("--config_path", type=str, default=None)

    test_parser = subparsers.add_parser("test", parents=[parent_parser])
    test_parser.add_argument("--model_path", type=str, required=True)
    test_parser.add_argument("--config_path", type=str, required=True)

    args = parser.parse_args()
    if args.mode == "train":
        start_training(args)
    elif args.mode == "test":
        start_testing(args)
    print("🎉 All done!")
