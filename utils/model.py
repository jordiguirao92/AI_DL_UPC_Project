import torch

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_model(model, path):
    torch.save(model.state_dict(), path)
