import torch
import numpy as np
import argparse
import os

from torch.backends import cudnn
from utils.utils import *
from solver import Solver


SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def str2bool(v):
    return v.lower() in "true"


def main(config):
    cudnn.benchmark = True
    if not os.path.exists(config.model_save_path):
        mkdir(config.model_save_path)
    solver = Solver(vars(config))

    if config.mode == "train":
        solver.train()
    elif config.mode == "test":
        solver.test()

    return solver


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--win_size", type=int, default=100)
    parser.add_argument("--input_c", type=int, default=38)
    parser.add_argument("--output_c", type=int, default=38)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--pretrained_model", type=str, default=None)
    parser.add_argument("--dataset", type=str, default="SMD")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "test"])
    parser.add_argument("--data_path", type=str, default="./datasets/data/SMD")
    parser.add_argument("--model_save_path", type=str, default="./datasets/checkpoints")
    parser.add_argument("--anormly_ratio", type=float, default=0.5)

    config = parser.parse_args()

    config.dataset = "MSL"

    if config.dataset == "SMD":
        config.anormly_ratio = 0.5
        config.num_epochs = 10
        config.input_c = 38
        config.output_c = 38
        config.data_path = "./datasets/data/SMD"
    elif config.dataset == "MSL":
        config.anormly_ratio = 1
        config.num_epochs = 3
        config.input_c = 55
        config.output_c = 55
        config.data_path = "./datasets/data/MSL"

    args = vars(config)
    print("------------ Options -------------")
    for k, v in sorted(args.items()):
        print("%s: %s" % (str(k), str(v)))
    print("-------------- End ----------------")

    # main(config)

    config.mode = "test"
    config.pretrained_model = 20
    main(config)
