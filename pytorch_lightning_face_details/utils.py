from pathlib import Path

import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

from pytorch_lightning_face_details.custom_dataset import FacesDataset


def split_dataset(dataset, test_split=0.25, train_sample_number=None, val_sample_number=None, seed=0):
    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=test_split, random_state=seed)

    if train_sample_number:
        train_sample_split = 1.0 - float(train_sample_number) / len(train_idx)
        train_idx_sample = train_test_split(train_idx, test_size=train_sample_split, random_state=seed)[0]
        train_dataset = Subset(dataset, train_idx_sample)
    else:
        train_dataset = Subset(dataset, train_idx)

    if val_sample_number:
        val_sample_split = 1.0 - float(train_sample_number) / len(train_idx)
        val_idx_sample = train_test_split(train_idx, test_size=val_sample_split, random_state=seed)[0]
        val_dataset = Subset(dataset, val_idx_sample)
    else:
        val_dataset = Subset(dataset, val_idx)
    return train_dataset, val_dataset


def get_dataloaders(return_classes=False):
    root_dir = Path("/home/sam/Downloads/face_details_dataset")
    batch_size = 32

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomRotation(30),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    train_sample_number = 800
    val_sample_number = 200
    rtl_dataset = FacesDataset(str(root_dir), transform=transform)
    labels = rtl_dataset.classes
    train_dataset, val_dataset = split_dataset(rtl_dataset,
                                               train_sample_number=train_sample_number,
                                               val_sample_number=val_sample_number
                                               )
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
    if return_classes:
        return train_dataloader, val_dataloader, labels
    return train_dataloader, val_dataloader


def load_model(model_path):
    model_path = Path(model_path)
    assert model_path.is_file()
    checkpoint = torch.load(model_path)
    model = checkpoint.get("model")
    model.load_state_dict(checkpoint.get("state_dict"))
    return model


if __name__ == "__main__":
    t = get_dataloaders()
    one = 1
