import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--net", help="generator or gan", type=str, default="gan")
parser.add_argument("--lr", help="learning rate", type=float, default=0.0001)
parser.add_argument("--batch_size", help="batch size", type=int, default=4)
parser.add_argument("--epochs", help="training number of epochs", type=int, default=11)
parser.add_argument("--loss", help="l1 or mse", type=str, default="l1")
parser.add_argument("--d_weight", help="value of discriminator loss", type=float, default=40)
parser.add_argument("--generator_last", help="sigmoid or tanh", type=str, default="sigmoid")
parser.add_argument("--generator_norm", help="batch, instance, spectral", type=str, default="batch")
parser.add_argument("--discriminator_last", help="sigmoid or tanh", type=str, default="sigmoid")
parser.add_argument("--discriminator_norm", help="batch, instance, spectral", type=str, default="batch")
parser.add_argument("--discriminator_activation", help="leakyRelu, relu", type=str, default="leakyRelu")
parser.add_argument("--discriminator_size", help="14, 15 or 1", type=int, default=14)

args = parser.parse_args()