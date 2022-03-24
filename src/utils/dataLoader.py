from torchvision import transforms
from torch.utils.data import DataLoader
from dataset.dataset import NoiseDataset

def get_data_loaders(batch_size):
    #transform = transforms.Compose([transforms.Resize(256), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]) 
    transform_train = transforms.Compose([transforms.ToPILImage(),transforms.RandomCrop(256), transforms.Resize(256), transforms.ToTensor()])
    transform_eval = transforms.Compose([transforms.ToPILImage(),transforms.CenterCrop(256), transforms.Resize(256), transforms.ToTensor()])
    train_set = NoiseDataset(path_to_images='../dataset/images', mode='training', transform=transform_train)
    eval_set = NoiseDataset(path_to_images='../dataset/images', mode='validation', transform=transform_eval)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    eval_loader = DataLoader(eval_set, batch_size=batch_size, shuffle=False)

    return train_loader, eval_loader