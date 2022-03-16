import torch.nn as nn
from IPython import embed
from utils.parser import args
from gan_training import gan_init
from generator_training import generator_init


if __name__ == "__main__":
    config = {
        "lr": args.lr,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        'log_interval': 5
    }

    if args.loss == "l1":
        config["loss"] = nn.L1Loss()
    else:
        config["loss"] = nn.MSELoss()

    if args.net == "gan":
        gan_init(config)
    else:
        generator_init(config)


