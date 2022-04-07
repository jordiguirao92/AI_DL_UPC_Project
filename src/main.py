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
        "log_interval": 1,
        "d_weight": args.d_weight,
        "generator_last": args.generator_last,
        "generator_norm": args.generator_norm,
        "discriminator_last": args.discriminator_last,
        "discriminator_norm": args.discriminator_norm,
        "discriminator_activation": args.discriminator_activation
    }

    if args.generator_last == "sigmoid":
        config["generator_last"] = nn.Sigmoid()
    elif args.generator_last == "tanh":
        config["generator_last"] = nn.Tanh()
        
    if args.loss == "l1":
        config["loss"] = nn.L1Loss()
    else:
        config["loss"] = nn.MSELoss()

    if args.net == "gan":
        gan_init(config)
    else:
        generator_init(config)


