import os
import torch
import struct
import argparse
import numpy as np
from Datasets import MegaFace
from torch.nn import DataParallel
from torch.utils.data import DataLoader
from Backbones.Backbone import MobileFacenet, CBAMResNet

cv_type_to_dtype = {5: np.dtype('float32'), 6: np.dtype('float64')}
dtype_to_cv_type = {v: k for k, v in cv_type_to_dtype.items()}

def write_mat(filename, m):
    """Write mat m to file f"""
    if len(m.shape) == 1:
        rows = m.shape[0]
        cols = 1
    else:
        rows, cols = m.shape
    header = struct.pack('iiii', rows, cols, cols * 4, dtype_to_cv_type[m.dtype])

    with open(filename, 'wb') as outfile:
        outfile.write(header)
        outfile.write(m.data)

def read_mat(filename):
    """
    Reads an OpenCV mat from the given file opened in binary mode
    """
    with open(filename, 'rb') as fin:
        rows, cols, stride, type_ = struct.unpack('iiii', fin.read(4 * 4))
        mat = np.fromstring(str(fin.read(rows * stride)), dtype=cv_type_to_dtype[type_])
        return mat.reshape(rows, cols)

def extract_feature(model_path, backbone_net, face_scrub_path, megaface_path, batch_size=1024, gpus='0', do_norm=False):
    # gpu init
    multi_gpus = False
    if len(gpus.split(',')) > 1:
        multi_gpus = True
    os.environ['CUDA_VISIBLE_DEVICES'] = gpus
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # backbone
    backbones = {'MobileFaceNet': MobileFacenet(),
                 'ResNet50_IR': CBAMResNet(50, feature_dim=args.feature_dim, mode='ir'),
                 'SEResNet50_IR': CBAMResNet(50, feature_dim=args.feature_dim, mode='ir_se'),
                 'ResNet100_IR': CBAMResNet(100, feature_dim=args.feature_dim, mode='ir'),
                 'SEResNet100_IR': CBAMResNet(100, feature_dim=args.feature_dim, mode='ir_se')}
    if backbone_net in backbones:
        net = backbones[backbone_net]
    else:
        print(backbone_net + ' is not available!')

    # load parameter
    net.load_state_dict(torch.load(model_path))

    if multi_gpus == True:
        net = DataParallel(net).to(device)
    else:
        net = net.to(device)
    net.eval()

    # dataset and dataloader
    megaface_dataset = MegaFace(face_scrub_path, megaface_path)
    megaface_dataloader = DataLoader(megaface_dataset, batch_size=batch_size, shuffle=False, num_workers=12, drop_last=False)

    for data in megaface_dataloader:
        img, img_path= data[0].to(device), data[1]
        with torch.no_grad():
            output = net(img).data.cpu().numpy()

        if do_norm is False:
            for i in range(len(img_path)):
                abs_path = img_path[i] + '.feat'
                write_mat(abs_path, output[i])
            print('extract 1 batch...without feature normalization')
        else:
            for i in range(len(img_path)):
                abs_path = img_path[i] + '.feat'
                feat = output[i]
                feat = feat / np.sqrt((np.dot(feat, feat)))
                write_mat(abs_path, feat)
            print('extract 1 batch...with feature normalization')
    print('all images have been processed!')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test model on MegaFace')
    parser.add_argument('--model_path', type=str, default='../Trained_Models/CASIA_WebFace_MobileFace_2020-08-12 16:24:48/Iter_53400_net.pth', help='The path of trained model')
    parser.add_argument('--backbone', type=str, default='MobileFaceNet', help='MobileFaceNet, ResNet50_IR, SEResNet50_IR, ResNet100_IR, SEResNet100_IR')
    parser.add_argument('--facescrub_dir', type=str, default='/data/face_datasets/test_datasets/face_recognition/MegaFace/facescrub_align_112/', help='facescrub data')
    parser.add_argument('--megaface_dir', type=str, default='/data/face_datasets/test_datasets/face_recognition/MegaFace/megaface_align_112/', help='megaface data')
    parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
    parser.add_argument('--feature_dim', type=int, default=128, help='feature dimension')
    parser.add_argument('--gpus', type=str, default='0,1', help='gpu list')
    parser.add_argument("--do_norm", type=int, default=1, help="1 if normalize feature, 0 do nothing(Default case)")
    args = parser.parse_args()

    extract_feature(args.model_path, args.backbone, args.facescrub_dir, args.megaface_dir, args.batch_size, args.gpus, args.do_norm)