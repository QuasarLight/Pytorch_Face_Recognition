import os
import time
import torch
import numpy as np
import torch.optim as optim
from apex import amp
from Config import args
from Datasets import LFW, CFP_FP, AgeDB30
from Datasets import CASIA_WebFace, MS_Celeb_1M
from Utils import Visualizer
from Utils import init_logger
from Utils import ChangeTimeFormat
from torch.nn import DataParallel
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from Backbones.Backbone import MobileFacenet, CBAMResNet
from Backbones.Margin import ArcMarginProduct, CosineMarginProduct, InnerProduct
from LFW_Evaluation import evaluation_10_fold, getFeatureFromTorch

# device initialization
os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, args.gpus))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('The current device is:', device)

# model_save_dir and log initialization
if not args.para_adj_mode:
    save_dir = os.path.join(args.save_dir, args.train_dataset + '_' + args.backbone + '_' + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    if os.path.exists(save_dir):
        raise NameError('model dir exists!')
    os.makedirs(save_dir)
    logger = init_logger(save_dir)
    _print = logger.info
else:
    _print = print

# visualizer initialization
if args.use_visdom == True:
    vis = Visualizer(env=args.model_prefix + '_' + args.backbone)

# train_dataset and train_dataloader
if args.train_dataset is 'CASIA_WebFace':
    train_dataset = CASIA_WebFace(args.webface_dataset_path, args.webface_file_list)
elif args.train_dataset is 'MS_Celeb_1M':
    train_dataset = MS_Celeb_1M(args.ms1m_dataset_path, args.ms1m_file_list)
train_dataloader = DataLoader(train_dataset, args.batch_size, shuffle = True, num_workers = 32, drop_last = False, pin_memory = True)
# test_dataset and test_dataloader
lfw_dataset = LFW(args.lfw_dataset_path, args.lfw_file_list)
lfw_dataloader = DataLoader(lfw_dataset, batch_size = 128, shuffle = False, num_workers = 32, drop_last = False)
cfp_dataset = CFP_FP(args.cfp_dataset_path, args.cfp_file_list)
cfp_dataloader = DataLoader(cfp_dataset, batch_size = 128, shuffle = False, num_workers = 32, drop_last = False)
agedb_dataset = AgeDB30(args.agedb_dataset_path, args.agedb_file_list)
agedb_dataloader = DataLoader(agedb_dataset, batch_size = 128, shuffle = False, num_workers = 32, drop_last = False)

# select backbone and margin
# backbone
backbones = {'MobileFaceNet': MobileFacenet(),
            'ResNet50_IR':    CBAMResNet(50, feature_dim=args.feature_dim, mode='ir'),
            'SEResNet50_IR':  CBAMResNet(50, feature_dim=args.feature_dim, mode='ir_se'),
            'ResNet100_IR':   CBAMResNet(100, feature_dim=args.feature_dim, mode='ir'),
            'SEResNet100_IR': CBAMResNet(100, feature_dim=args.feature_dim, mode='ir_se')}
if args.backbone in backbones:
    net = backbones[args.backbone]
else:
    _print(args.backbone + ' is not available!')
# margin
margins = {'ArcFace': ArcMarginProduct(args.feature_dim, train_dataset.num_classes, s=args.scale_size),
           'CosFace': CosineMarginProduct(args.feature_dim, train_dataset.num_classes, s=args.scale_size),
           'Softmax': InnerProduct(args.feature_dim, train_dataset.num_classes)}
if args.margin in margins:
    margin = margins[args.margin]
else:
    _print(args.margin + 'is not available!')
# resume model
if args.resume == True:
    _print('resume the model from:  '+args.resume_backbone_path+'\n\t\t\t\t\t\t'+args.resume_margin_path)
    net.load_state_dict(torch.load(args.resume_backbone_path))
    margin.load_state_dict(torch.load(args.resume_margin_path))
# put tensor on device
net = net.to(device)
margin = margin.to(device)

# loss function
loss_fn = torch.nn.CrossEntropyLoss()
loss_fn =loss_fn.to(device)

# optimizer
optimizer = optim.SGD([
    {'params': net.parameters()},
    {'params': margin.parameters()},
],lr = args.initial_lr, momentum = 0.9, nesterov = True, weight_decay = args.weight_decay)

# learning rate scheduler
scheduler = lr_scheduler.MultiStepLR(optimizer, milestones = [13, 21], gamma = 0.1)

# use amp
if args.use_amp == True:
    [net, margin], optimizer_ft = amp.initialize([net, margin], optimizer, opt_level="O1")

# use multiple GPU devices
if args.use_multi_gpus == True:
    net = DataParallel(net).to(device)
    margin = DataParallel(margin).to(device)

# best test accuracy and corresponding iteration times
best_lfw_accuracy = 0.0
best_lfw_iters = 0
best_agedb_accuracy = 0.0
best_agedb_iters = 0
best_cfp_accuracy = 0.0
best_cfp_iters = 0

# training network
current_iters = 0
total_iters = args.total_epoch*len(train_dataloader)
since_time = time.time()
for epoch in range(1, args.total_epoch + 1):
    # trian model
    _print('Training Epoch: '+str(epoch)+'/'+str(args.total_epoch))
    net.train()
    for train_data in train_dataloader:
        # get images and labels from dataloader
        images, labels = train_data[0], train_data[1]
        images = images.to(device)
        labels = labels.to(device)
        # forward propagation
        embeddings = net(images)
        output = margin(embeddings, labels)
        # calculate loss
        loss = loss_fn(output, labels)
        # back propagation
        optimizer.zero_grad()
        if args.use_amp == True:
            if not torch.isnan(loss):
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                optimizer.step()
        else:
            loss.backward()
            optimizer.step()
        current_iters += 1

        # print train information
        if current_iters % 100 == 0:
            # calculate train accuracy
            _, prediction = torch.max(output.data, 1)
            num_correct_classified = float((np.array(prediction.cpu()) == np.array(labels.data.cpu)).sum())
            train_accuracy = num_correct_classified/args.batch_size
            # calculate remaining training time
            current_time = time.time()
            remaining_time = (total_iters - current_iters)*(current_time - since_time)/100
            remaining_time = ChangeTimeFormat(remaining_time)
            since_time = time.time()
            # draw softmax loss curve and train accuracy curve
            if args.use_visdom == True:
                vis.plot_curves({'softmax loss': loss.item()}, x=current_iters, title='train loss',
                                xlabel='iterations', ylabel='train loss')
                vis.plot_curves({'train accuracy': train_accuracy}, x=current_iters, title='train accuracy',
                                xlabel='iterations', ylabel='train accuracy')
            # print train information, including current epoch, current iterations times, loss, train accuracy, learning rate and remaining training time
            _print('Iters:'+str(epoch)+'/'+str(current_iters)+' loss: %.4f' % (loss.item())+' train accuracy:'+
                   str(train_accuracy)+' learning rate:'+str(optimizer.param_groups[0]['lr'])+' remaining time:'+
                   str(remaining_time))

        #test model
        if current_iters % args.test_freq == 0:
            net.eval()
            # test model on lfw
            getFeatureFromTorch('./Test_Data/cur_lfw_result.mat', net, device, lfw_dataset, lfw_dataloader)
            lfw_accuracy = evaluation_10_fold('./Test_Data/cur_lfw_result.mat')
            lfw_accuracy = np.mean(lfw_accuracy)
            _print('LFW Average Accuracy: {:.4f}%'.format(np.mean(lfw_accuracy) * 100))
            if best_lfw_accuracy <= lfw_accuracy * 100:
                best_lfw_accuracy = lfw_accuracy * 100
                best_lfw_iters = current_iters
            # test model on AgeDB30
            getFeatureFromTorch('./Test_Data/cur_agedb_result.mat', net, device, agedb_dataset, agedb_dataloader)
            agedb_accuracy = evaluation_10_fold('./Test_Data/cur_agedb_result.mat')
            agedb_accuracy = np.mean(agedb_accuracy)
            _print('AgeDB-30 Average Accuracy: {:.4f}%'.format(np.mean(agedb_accuracy) * 100))
            if best_agedb_accuracy <= best_agedb_accuracy * 100:
                best_agedb_accuracy = agedb_accuracy * 100
                best_agedb_iters = current_iters
            # test model on CFP-FP
            getFeatureFromTorch('./Test_Data/cur_cfp_result.mat', net, device, cfp_dataset, cfp_dataloader)
            cfp_accuracy = evaluation_10_fold('./Test_Data/cur_cfp_result.mat')
            cfp_accuracy = np.mean(cfp_accuracy)
            _print('CFP-FP Average Accuracy: {:.4f}%'.format(np.mean(cfp_accuracy) * 100))
            if best_cfp_accuracy <= cfp_accuracy * 100:
                best_cfp_accuracy = cfp_accuracy * 100
                best_cfp_iters = current_iters
            # draw test accuracy curve
            if args.use_visdom == True:
                vis.plot_curves({'lfw_accuracy': lfw_accuracy,'agedb_accuracy':agedb_accuracy,'cfp_accuracy':cfp_accuracy}, x=current_iters, title='test accuracy', xlabel='iterations',
                                ylabel='test accuracy')
            # print current best test accuracy
            _print('Current Best Test Accuracy: LFW: {:.4f}% in iters: {}, AgeDB-30: {:.4f}% in iters: {} and CFP-FP: {:.4f}% in iters: {}'.format(
                   best_lfw_accuracy, best_lfw_iters, best_agedb_accuracy, best_agedb_iters, best_cfp_accuracy, best_cfp_iters))
            net.train()

        #save model
        if epoch == args.total_epoch and current_iters % args.save_freq == 0:
            if not args.para_adj_mode:
                _print('Saving model: {}'.format(current_iters))
                if not os.path.exists(save_dir):
                    os.mkdir(save_dir)
                torch.save(net.module.state_dict(), os.path.join(save_dir, 'Iter_%d_net.pth' % current_iters))
                torch.save(margin.module.state_dict(), os.path.join(save_dir, 'Iter_%d_margin.pth' % current_iters))

    # adjust learning rate
    scheduler.step()

# test model on megaface
if args.test_on_megaface == True:
    pass

