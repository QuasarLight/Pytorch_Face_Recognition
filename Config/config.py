import argparse

parser = argparse.ArgumentParser(description='Pytorch For Deep Face Recognition')

# parameter adjustment mode
parser.add_argument('--para_adj_mode', type=bool, default=False,help='parameter adjustment mode')

# visualizer
parser.add_argument('--use_visdom', type=bool, default=False,help='whether to use visdom')

# device parameters
parser.add_argument('--use_amp', type=bool, default=True,help='whether to use automatic mixed precision (AMP)')
parser.add_argument('--use_multi_gpus', type=bool, default=True,help='whether to use multiple GPU devices')
parser.add_argument('--gpus', type=list, default=[0, 1],help='appoint GPU devices')

# dataset parameters
parser.add_argument('--train_dataset', type=str, default='MS_Celeb_1M', help='CASIA_WebFace, MS_Celeb_1M')
parser.add_argument('--webface_dataset_path', type=str, default='/home/CaiMao/Face_Pytorch-master/dataset/webface-112x112/casia-112x112', help='webface dataset path')
parser.add_argument('--webface_file_list', type=str, default='/home/CaiMao/Face_Pytorch-master/dataset/webface-112x112/casia-112x112.list', help='webface files list')
parser.add_argument('--ms1m_dataset_path', type=str, default='/home/CaiMao/MS1M_112x112/MS1M_112x112', help='ms1m dataset path')
parser.add_argument('--ms1m_file_list', type=str, default='/home/CaiMao/MS1M_112x112/MS1M-112x112.txt', help='ms1m files list')
parser.add_argument('--lfw_dataset_path', type=str, default='/home/CaiMao/Face_Pytorch-master/dataset/lfw-112x112/lfw-112x112', help='lfw dataset path')
parser.add_argument('--lfw_file_list', type=str, default='/home/CaiMao/Face_Pytorch-master/dataset/lfw-112x112/pairs.txt', help='lfw pair file list')
parser.add_argument('--cfp_dataset_path', type=str, default='/data/face_datasets/train_datasets/MS1M_112x112/cfp_fp', help='cfp-fp dataset path')
parser.add_argument('--cfp_file_list', type=str, default='/data/face_datasets/train_datasets/MS1M_112x112/cfp_fp_pair.txt', help='cfp-fp pair file list')
parser.add_argument('--agedb_dataset_path', type=str, default='/data/face_datasets/train_datasets/MS1M_112x112/agedb_30', help='agedb-30 dataset path')
parser.add_argument('--agedb_file_list', type=str, default='/data/face_datasets/train_datasets/MS1M_112x112/agedb_30_pair.txt', help='agedb-30 pair file list')

# training parameters
parser.add_argument('--initial_lr', type=float, default=0.1, help='initial learning rate')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
parser.add_argument('--batch_size', type=int, default=256, help='batch size')
parser.add_argument('--total_epoch', type=int, default=25, help='total epochs')
parser.add_argument('--backbone', type=str, default='ResNet50_IR', help='MobileFaceNet, ResNet50_IR, SEResNet50_IR, ResNet100_IR, SEResNet100_IR')
parser.add_argument('--margin', type=str, default='ArcFace', help='ArcFace, CosFace, Softmax')
parser.add_argument('--feature_dim', type=int, default=512, help='feature dimension, 128 or 512, if backbone is MobileFaceNet,this option must be 128')
parser.add_argument('--scale_size', type=float, default=32.0, help='scale size')

# testing parameters
parser.add_argument('--test_freq', type=int, default=1000, help='the frequency of testing model')
parser.add_argument('--test_on_megaface', type=bool, default=True, help='whether to test model on megaface at the end of the iteration')

# saving parameters
parser.add_argument('--save_freq', type=int, default=1000, help='the frequency of saving model')
parser.add_argument('--save_dir', type=str, default='./Trained_Models', help='model save dir')

# resume model parameters
parser.add_argument('--resume', type=bool, default=False, help='resume model')
parser.add_argument('--resume_backbone_path', type=str, default='Trained_Models/CASIA_WebFace_MobileFace_2020-08-12 16:24:48/Iter_53400_net.pth', help='resume backbone path')
parser.add_argument('--resume_margin_path', type=str, default='Trained_Models/CASIA_WebFace_MobileFace_2020-08-12 16:24:48/Iter_53400_margin.pth', help='resume margin path')

args = parser.parse_args()