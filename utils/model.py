import torch

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def save_model(model, path):
    torch.save(model.state_dict(), path)

def accuracy(labels, outputs):
    preds = outputs.argmax(-1)
    acc = (preds == labels.view_as(preds)).float().detach().numpy().mean()
    return acc
    
def binary_accuracy(labels, outputs):
    preds = outputs.round()
    acc = (preds == labels.view_as(preds)).float().detach().numpy().mean()
    return acc

def binary_accuracy_with_logits(labels, outputs):
    preds = torch.sigmoid(outputs).round()
    acc = (preds == labels.view_as(preds)).float().detach().numpy().mean()
    return acc