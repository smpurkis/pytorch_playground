from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
from torchvision import transforms


class FacesDetailsDataset(Dataset):
    def __init__(self, base_path, transform=None):
        self.base_path = Path(base_path)
        self.csv_path = Path(base_path, "person.csv")
        self.images_path = Path(base_path, "front", "front")
        self.target_labels = {"weight": "continuous", "hair": "categoric", "sex": "categoric", "height": "continuous",
                              "race": "categoric", "eyes": "categoric"}
        self.image_ids, self.data = self.extract_csv_data(self.csv_path)
        self.encoded_y = self.encode_data(self.data)
        self.transform = transform

    def __len__(self):
        return len(self.encoded_y)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        img_path = Path(self.images_path, self.image_ids[index] + ".jpg")
        assert img_path.exists()
        image = Image.open(str(img_path))
        landmarks = self.encoded_y[index]

        if self.transform:
            if image.mode != "RGB":
                image = image.convert("RGB")
            image = self.transform(image)
        sample = [image, landmarks]

        return sample

    def extract_csv_data(self, csv_path):
        assert csv_path.is_file()
        data = pd.read_csv(str(csv_path))
        labels = list(data.columns.values)
        target_indices = [labels.index(target) for target in self.target_labels.keys()]
        ids = data.values[:, 0]
        labels = data.values[:, target_indices[0]:target_indices[-1] + 1]
        return ids, labels

    def encode_data(self, data):
        encoded_x = None
        self.classes = []
        for i in range(0, data.shape[1]):
            target_label = list(self.target_labels.keys())[i]
            data_type = self.target_labels.get(target_label)
            label_encoder = LabelEncoder()
            column = data[:, i]
            if data_type == "continuous":
                column, label_encoder = self.categorize_data(column)
                column = column.reshape(data.shape[0], 1)
            elif data_type == "categoric":
                types_list = [type(i) for i in column]
                t = np.argmax([types_list.count(type_) for type_ in set(types_list)])
                main_type = tuple(set(types_list))[t]
                column = column.astype(main_type)
                column = label_encoder.fit_transform(column)
                label_encoder = list(label_encoder.classes_)
                column = column.reshape(data.shape[0], 1)
            self.classes.append({"name": target_label, "classes": label_encoder})
            if encoded_x is None:
                encoded_x = column
            else:
                encoded_x = np.concatenate((encoded_x, column), axis=1)
        print("X shape: : ", encoded_x.shape)
        encoded_x = encoded_x.astype(np.float32)
        encoded_x = np.nan_to_num(encoded_x, nan=-1)
        return encoded_x

    def decode_data(self, output):
        if not self.classes:
            raise Exception("Please load label classes")
        output = torch.round(output).int()
        result = {self.classes[index].get("name"): self.classes[index].get("classes")[class_number]
                  for index, class_number in enumerate(output)}
        return result

    def categorize_data(self, data, number_of_categories=10):
        split = pd.qcut(data, number_of_categories, duplicates="drop")
        label_encoder = [str(elem) for elem in split.categories]
        data = split.codes
        return data, label_encoder


if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomRotation(30),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    root_dir = Path("/home/sam/Downloads/face_details_dataset")
    rtl_dataset = FacesDetailsDataset(str(root_dir), transform=transform)
    output = torch.Tensor([4., 3., 1., 1., 6., 1.])
    result = rtl_dataset.decode_data(output)
    one = 1
