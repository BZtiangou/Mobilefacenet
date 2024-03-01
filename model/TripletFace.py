import random
import os
from PIL import Image
from torch.utils.data import Dataset
class TripletFaceDataset(Dataset):
    def __init__(self, image_folder,persons,transform=None):
        self.image_folder = image_folder
        self.transform = transform
        self.persons = persons

    def __getitem__(self, _):
        anchor_person = random.choice(self.persons)
        positive_person = anchor_person
        negative_person = random.choice(self.persons)
        while negative_person == anchor_person:
            negative_person = random.choice(self.persons)

        anchor_img_path = random.choice(os.listdir(
            os.path.join(self.image_folder, anchor_person)))
        positive_img_path = random.choice(os.listdir(
            os.path.join(self.image_folder, positive_person)))
        negative_img_path = random.choice(os.listdir(
            os.path.join(self.image_folder, negative_person)))

        anchor_img = Image.open(os.path.join(
            self.image_folder, anchor_person, anchor_img_path))
        positive_img = Image.open(os.path.join(
            self.image_folder, positive_person, positive_img_path))
        negative_img = Image.open(os.path.join(
            self.image_folder, negative_person, negative_img_path))

        # 进行数据处理
        transform = self.transform

        anchor_img = transform(anchor_img)
        positive_img = transform(positive_img)
        negative_img = transform(negative_img)

        return (anchor_img, positive_img, negative_img),(anchor_person, positive_person, negative_person)

    def __len__(self):
        return len(self.persons) * 5  # 假设每个人有5张图片