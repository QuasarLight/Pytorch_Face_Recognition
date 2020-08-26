from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np
import cv2
import os

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

class LFW(Dataset):
    def __init__(self, dataset_path, file_list):
        self.dataset_path = dataset_path
        self.file_list = file_list
        self.left_images = []
        self.right_images = []
        self.folds = []
        self.labels = []

        with open(file_list) as f:
            pairs = f.read().splitlines()[1:]
        for i, p in enumerate(pairs):
            p = p.split('\t')
            if len(p) == 3:
                left_image = p[0] + '/' + p[0] + '_' + '{:04}.jpg'.format(int(p[1]))
                right_image = p[0] + '/' + p[0] + '_' + '{:04}.jpg'.format(int(p[2]))
                fold = i // 600
                label = 1
            elif len(p) == 4:
                left_image = p[0] + '/' + p[0] + '_' + '{:04}.jpg'.format(int(p[1]))
                right_image = p[2] + '/' + p[2] + '_' + '{:04}.jpg'.format(int(p[3]))
                fold = i // 600
                label = -1
            self.left_images.append(left_image)
            self.right_images.append(right_image)
            self.folds.append(fold)
            self.labels.append(label)

    def __getitem__(self, index):

        image_left = image_loader(os.path.join(self.dataset_path, self.left_images[index]))
        image_right = image_loader(os.path.join(self.dataset_path, self.right_images[index]))
        image_list = [image_left, cv2.flip(image_left, 1), image_right, cv2.flip(image_right, 1)]

        for i in range(len(image_list)):
            image_list[i] = transform(image_list[i])

        return image_list

    def __len__(self):
        return len(self.left_images)


if __name__ == '__main__':
    dataset_path = '/home/CaiMao/Face_Pytorch-master/dataset/lfw-112x112/lfw-112x112'
    file_list = '/home/CaiMao/Face_Pytorch-master/dataset/lfw-112x112/pairs.txt'

    lfw_dataset = LFW(dataset_path, file_list)
    lfw_dataloader = DataLoader(lfw_dataset, batch_size=128, shuffle=False, num_workers=4, drop_last=False)
    print(len(lfw_dataset))
    print(len(lfw_dataloader))
    for data in lfw_dataloader:
        print(len(data))