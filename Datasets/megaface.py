import os
import cv2
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

def image_loader(image_path):
    try:
        image = cv2.imread(image_path)
        if len(image.shape) == 2:
            image = np.stack([image]*3, 2)
        return image
    except IOError:
        print('fail to load image:' + image_path)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5))
])

class MegaFace(Dataset):
    def __init__(self, facescrub_dir, megaface_dir):
        test_image_file_list = []
        print('Scanning files under facescrub and megaface...')
        for root, dirs, files in os.walk(facescrub_dir):
            for e in files:
                filename = os.path.join(root, e)
                ext = os.path.splitext(filename)[1].lower()
                if ext in ('.png', '.bmp', '.jpg', '.jpeg'):
                    test_image_file_list.append(filename)
        for root, dirs, files in os.walk(megaface_dir):
            for e in files:
                filename = os.path.join(root, e)
                ext = os.path.splitext(filename)[1].lower()
                if ext in ('.png', '.bmp', '.jpg', '.jpeg'):
                    test_image_file_list.append(filename)

        self.image_list = test_image_file_list

    def __getitem__(self, index):
        img_path = self.image_list[index]
        img = image_loader(img_path)

        #random flip with ratio of 0.5
        if np.random.choice(2) == 1:
            img = cv2.flip(img, 1)

        img = transform(img)

        return img, img_path

    def __len__(self):
        return len(self.image_list)


if __name__ == '__main__':
    facescrub = '/data/face_datasets/test_datasets/face_recognition/MegaFace/facescrub_align_112/'
    megaface = '/data/face_datasets/test_datasets/face_recognition/MegaFace/megaface_align_112/'

    megaface_dataset = MegaFace(facescrub, megaface)
    megaface_dataloader = DataLoader(megaface_dataset, batch_size=128, shuffle=False, num_workers=4, drop_last=False)
    print(len(megaface_dataset))
    print(len(megaface_dataloader))
    for data in megaface_dataloader:
        print(len(data))