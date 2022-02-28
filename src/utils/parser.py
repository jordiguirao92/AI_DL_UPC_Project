import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--training", help="generator or discriminator training", type=str, default="generator")
parser.add_argument("--lr", help="learning rate", type=float, default=0.0001)
parser.add_argument("--batch_size", help="batch size", type=int, default=4)
parser.add_argument("--epochs", help="training number of epochs", type=int, default=10)
parser.add_argument("--loss", help="l1 or mse", type=str, default="l1")

args = parser.parse_args()