import os
import torch
import scipy.io
import numpy as np
from Config import args
from Datasets import AgeDB30
from torch.nn import DataParallel
from torch.utils.data import DataLoader
from Backbones.Backbone import MobileFacenet, CBAMResNet

def getAccuracy(scores, flags, threshold):
    p = np.sum(scores[flags == 1] > threshold)
    n = np.sum(scores[flags == -1] < threshold)
    return 1.0 * (p + n) / len(scores)

def getThreshold(scores, flags, thrNum):
    accuracys = np.zeros((2 * thrNum + 1, 1))
    thresholds = np.arange(-thrNum, thrNum + 1) * 1.0 / thrNum
    for i in range(2 * thrNum + 1):
        accuracys[i] = getAccuracy(scores, flags, thresholds[i])
    max_index = np.squeeze(accuracys == np.max(accuracys))
    bestThreshold = np.mean(thresholds[max_index])
    return bestThreshold

def evaluation_10_fold(feature_path='./result/cur_epoch_agedb_result.mat'):
    ACCs = np.zeros(10)
    result = scipy.io.loadmat(feature_path)
    for i in range(10):
        fold = result['fold']
        flags = result['flag']
        featureLs = result['fl']
        featureRs = result['fr']

        valFold = fold != i
        testFold = fold == i
        flags = np.squeeze(flags)

        mu = np.mean(np.concatenate((featureLs[valFold[0], :], featureRs[valFold[0], :]), 0), 0)
        mu = np.expand_dims(mu, 0)
        featureLs = featureLs - mu
        featureRs = featureRs - mu
        featureLs = featureLs / np.expand_dims(np.sqrt(np.sum(np.power(featureLs, 2), 1)), 1)
        featureRs = featureRs / np.expand_dims(np.sqrt(np.sum(np.power(featureRs, 2), 1)), 1)

        scores = np.sum(np.multiply(featureLs, featureRs), 1)
        threshold = getThreshold(scores[valFold[0]], flags[valFold[0]], 10000)
        ACCs[i] = getAccuracy(scores[testFold[0]], flags[testFold[0]], threshold)

    return ACCs

def loadModel(data_root, file_list, backbone_net, gpus='0', model_para_path=None):
    # gpu init
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, args.gpus))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # backbone
    backbones = {'MobileFaceNet':  MobileFacenet(),
                 'ResNet50_IR':    CBAMResNet(50, feature_dim=args.feature_dim, mode='ir'),
                 'SEResNet50_IR':  CBAMResNet(50, feature_dim=args.feature_dim, mode='ir_se'),
                 'ResNet100_IR':   CBAMResNet(100, feature_dim=args.feature_dim, mode='ir'),
                 'SEResNet100_IR': CBAMResNet(100, feature_dim=args.feature_dim, mode='ir_se')}
    if backbone_net in backbones:
        net = backbones[backbone_net]
    else:
        print(backbone_net + ' is not available!')

    # load parameter
    net.load_state_dict(torch.load(model_para_path))

    if args.use_multi_gpus == True:
        net = DataParallel(net).to(device)
    else:
        net = net.to(device)

    # dataset and dataloader
    agedb_dataset = AgeDB30(data_root, file_list)
    agedb_loader = DataLoader(agedb_dataset, batch_size=128, shuffle=False, num_workers=4, drop_last=False)

    return net.eval(), device, agedb_dataset, agedb_loader

def getFeatureFromTorch(feature_save_dir, net, device, data_set, data_loader):
    featureLs = None
    featureRs = None
    count = 0
    for data in data_loader:
        for i in range(len(data)):
            data[i] = data[i].to(device)
        count += data[0].size(0)
        #print('extracing deep features from the face pair {}...'.format(count))
        with torch.no_grad():
            res = [net(d).data.cpu().numpy() for d in data]
        featureL = np.concatenate((res[0], res[1]), 1)
        featureR = np.concatenate((res[2], res[3]), 1)
        # print(featureL.shape, featureR.shape)
        if featureLs is None:
            featureLs = featureL
        else:
            featureLs = np.concatenate((featureLs, featureL), 0)
        if featureRs is None:
            featureRs = featureR
        else:
            featureRs = np.concatenate((featureRs, featureR), 0)
        # print(featureLs.shape, featureRs.shape)

    result = {'fl': featureLs, 'fr': featureRs, 'fold': data_set.folds, 'flag': data_set.labels}
    scipy.io.savemat(feature_save_dir, result)

if __name__ == '__main__':
    model_para_path = 'Trained_Models/CASIA_WebFace_MobileFace_2020-08-12 16:24:48/Iter_53400_net.pth'
    net, device, agedb_dataset, agedb_loader = loadModel(args.agedb_dataset_path, args.agedb_file_list, args.backbone, args.gpus, model_para_path)
    getFeatureFromTorch('Test_Data/cur_agedb_result.mat', net, device, agedb_dataset, agedb_loader)
    ACCs = evaluation_10_fold('Test_Data/cur_agedb_result.mat')
    for i in range(len(ACCs)):
        print('{}    {:.2f}'.format(i + 1, ACCs[i] * 100))
    print('--------')
    print('AVE    {:.4f}'.format(np.mean(ACCs) * 100))

