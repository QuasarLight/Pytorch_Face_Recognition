from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np
import cv2

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

class CASIA_WebFace(Dataset):
    def __init__(self, dataset_path, file_list):
        self.dataset_path = dataset_path

        img_list = []
        label_list = []
        with open(file_list) as f:
            img_label_list = f.read().splitlines()
            for img_label in img_label_list:
                img_path, label = img_label.split('  ')
                img_list.append(img_path)
                label_list.append(int(label))

        self.img_list = img_list
        self.label_list = label_list
        self.num_images = len(self.img_list)
        self.num_classes = len(np.unique(self.label_list))
        print('dataset size: ', 'num_images/num_classes ', self.num_images, '/', self.num_classes)

    def __getitem__(self, index):
        image_path = self.img_list[index]
        label = self.label_list[index]

        # load image
        image = image_loader(image_path)

        # random flip with ratio of 0.5
        if np.random.choice(2) == 1:
            image = cv2.flip(image, 1)

        # transform numpy.ndarray to tensor and normalize it
        image = transform(image)

        return image, label

    def __len__(self):
        return self.num_images

if __name__ == '__main__':
    dataset_path = '/home/CaiMao/Face_Pytorch-master/dataset/webface-112x112/casia-112x112'
    file_list = '/home/CaiMao/Face_Pytorch-master/dataset/webface-112x112/casia-112x112.list'

    dataset = CASIA_WebFace(dataset_path, file_list)
    trainloader = DataLoader(dataset, batch_size = 256, shuffle = True, num_workers = 32, drop_last = False)

    print(len(trainloader))
    for data in trainloader:
        image_batch = data[0]
        label = data[1]
        print(type(data[0]),type(data[1]))
        # print(image_batch.shape)
        # print(len(label))

