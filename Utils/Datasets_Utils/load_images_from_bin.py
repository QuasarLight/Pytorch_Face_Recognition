import os
import cv2
import pickle
import mxnet as mx
from tqdm import tqdm

# For MS1M dataset, insightface provide a mxnet .rec file, just install a mxnet-cpu for extract images

def load_mx_rec(rec_path):
    save_path = os.path.join(rec_path, 'MS1M_112x112')
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    imgrec = mx.recordio.MXIndexedRecordIO(os.path.join(rec_path, 'train.idx'), os.path.join(rec_path, 'train.rec'), 'r')
    img_info = imgrec.read_idx(0)
    header,_ = mx.recordio.unpack(img_info)
    max_idx = int(header.label[0])
    for idx in tqdm(range(1,max_idx)):
        img_info = imgrec.read_idx(idx)
        header, img = mx.recordio.unpack_img(img_info)
        label = int(header.label)
        label_path = os.path.join(save_path, str(label).zfill(6))
        if not os.path.exists(label_path):
            os.makedirs(label_path)
        cv2.imwrite(os.path.join(label_path, str(idx).zfill(8) + '.jpg'), img)

def load_image_from_bin(bin_path, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    file = open(os.path.join(save_dir, '../', 'agedb_30_pair.txt'), 'w')
    bins, issame_list = pickle.load(open(bin_path, 'rb'), encoding='bytes')
    for idx in tqdm(range(len(bins))):
        _bin = bins[idx]
        img = mx.image.imdecode(_bin).asnumpy()
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(save_dir, str(idx+1).zfill(5)+'.jpg'), img)
        if idx % 2 == 0:
            label = 1 if issame_list[idx//2] == True else -1
            file.write(str(idx+1).zfill(5) + '.jpg' + ' ' + str(idx+2).zfill(5) +'.jpg' + ' ' + str(label) + '\n')

if __name__ == '__main__':
    bin_path = '/data/face_datasets/train_datasets/MS1M_112x112/agedb_30.bin'
    save_dir = '/data/face_datasets/train_datasets/MS1M_112x112/agedb_30'
    load_image_from_bin(bin_path, save_dir)
    # rec_path = '/data/face_datasets/train_datasets/MS1M_112x112'
    # load_mx_rec(rec_path)
