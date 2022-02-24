from torchvision import transforms
from torch.utils.data import DataLoader
from dataset.dataset import NoiseDataset

def get_data_loaders(batch_size):
    #transform = transforms.Compose([transforms.Resize(256), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]) 
    transform = transforms.Compose([transforms.CenterCrop(256), transforms.Resize(256), transforms.ToTensor()])
    train_set = NoiseDataset(path_to_images='./dataset/images', mode='training', transform=transform)
    test_set = NoiseDataset(path_to_images='./dataset/images', mode='testing', transform=transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader