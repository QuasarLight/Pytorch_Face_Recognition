# Pytorch_Face_Recognition  
Pytorch implementation of mainstream face recognition algorithms(ArcFace, CosFace).
## 1. Introduction  
(1) Pytorch implementation of ArcFace and CosFace.  
(2) Cleaned datasets are provided, including WebFace, MS-Celeb-1M, LFW, AgeDB-30, CFP-FP and MegaFace.  
(3) Pretrained models are provided. See **3. Results and Pretrained Models** for further details  
(4) Automatic Mixed Precision(AMP) Training is supported to accelerate training process.  
(5) Visdom is supported to visualize the changes of loss and accuracy during training process.
## 2. Usage 
(1) **Environment Preparation**   
 > Python 3.7  
 > Pytorch 1.4  
 > torchvision 0.5  
 > cudatoolkit 10.0  
 > apex 0.1 (optional)    
 > visdom 0.1.8.9 (optional)  
 
(2) Download this project to your machine.
  
(3) **Datasets Preparation**  
* Training Datasets :   
    > CASIA-WebFace (453580/10575) [BaiduNetDisk](https://pan.baidu.com/s/1V0679yun1JYYxRmzqiNjdw) *Extraction code* :r9a4   
    MS-Celeb-1M (3923399/86876)  [BaiduNetDisk](https://pan.baidu.com/s/1n7G3TCKZBaGizQYMoStdsw) *Extraction code* :04al  
* Test Datasets :  
    > LFW [BaiduNetDisk](https://pan.baidu.com/s/1NiWIj552t-yHD0KPBCWz2g) *Extraction code* :o93a   
    CFP-FP [BaiduNetDisk](https://pan.baidu.com/s/1-uJTJsJrWXgSS4PvvF3b8g) *Extraction code* :h7jo    
    AgeDB-30 [BaiduNetDisk](https://pan.baidu.com/s/1Qitk-M4h8wp9T2j2uceDgQ) *Extraction code* :dj7g     
    MegaFace [BaiduNetDisk](https://pan.baidu.com/s/1fHgY83E3jobskr-pGA8KYw) *Extraction code* :pfl2
    
(4) For training datasets, use Utils/Datasets_Utils/generate_dataset_list.py to generate dataset files list.
   
(5) Set training hyperparameters like *batch size*, *backbone*, *initial learning rate* in Config/config.py
  
(6) Run Train.py to start training process and training information will be saved in the log file.

(7) Use LFW_Evaluation.py, AgeDB-30_Evaluation.py, CFP-FP_Evaluation.py and MegaFace_Evaluation to run individual evaluation.
  
**Tips** :   
(1) In config.py, you can choose to open parameter adjustment mode. In this mode, training information and model will 
not be saved. It is convenient when you adjust the training hyperparameters.  

(2) When you want to visualize the training process, you can turn the option 'use_visdom' in config.py to True. Before use
it, make sure that you have installed the visdom and opened the server.  

(3) You can use Automatic Mixed Precision(AMP) to accelerate training process. It allows you to use the bigger batch size and
AMP can also avoid gradient explosion problem. To use it, just set option 'use_amp' in config.py as True. Remember to make sure
that you have installed the apex and your GPU has TensorCore before use AMP.  

(4) If you have multiple GPU devices and want to run parallel training, just set option 'use_multi_gpus' in config.py as True.

## 3. Results and Pretrained Models
###(1) LFW, AgeDB-30 and CFP-FP Evaluation Results
Training Dataset|Backbone     |Model Size|Loss   |LFW    |AgeDB-30|CFP-FP |Pretrained Models                |
:--------------:|:-----------:|:--------:|:-----:|:------|:------:|:-----:|:-------------------------------:|
CASIA-WebFace   |MobileFaceNet|4MB       |ArcFace|99.3333|92.5833 |94.0143|[BaiduNetDisk](https://pan.baidu.com/s/1wU7F8w-jYgJpjbZGFaJJtA) Extraction code:e3qm|
CASIA-WebFace   |ResNet50-IR  |170MB     |ArcFace|99.4667|93.9333 |95.5571|[BaiduNetDisk](https://pan.baidu.com/s/1H6vgckjqqAer9Rp2pHU_cQ) Extraction code:byqs|
CASIA-WebFace   |SEResNet50-IR|171MB     |ArcFace|99.3833|93.9333 |95.5857|[BaiduNetDisk](https://pan.baidu.com/s/19YoVDVB_N6MPR6VGI6tyQg) Extraction code:c355|
CASIA-WebFace   |ResNet100-IR |256MB     |ArcFace|99.5833|94.3500 |96.0429|[BaiduNetDisk](https://pan.baidu.com/s/14NoOJjKZar9JUp6fjruB_A) Extraction code:kqsi|
###(2) MegaFace Rank 1 Identifiaction Accuracy and Verfication TPR@FPR=1e-6 Results
Training Dataset|Backbone     |Model Size|Loss   |Identification Rank1 Acc|Verfication TPR@FPR=1e-6|
:--------------:|:-----------:|:--------:|:-----:|:----------------------:|:----------------------:|
CASIA-WebFace   |MobileFaceNet|4MB       |ArcFace|   68.46                |   83.49                |
CASIA-WebFace   |ResNet50-IR  |170MB     |ArcFace|   74.50                |   89.89                |
CASIA-WebFace   |SEResNet50-IR|171MB     |ArcFace|   74.72                |   89.41                |
CASIA-WebFace   |ResNet100-IR |256MB     |ArcFace|   74.39                |   90.86                | 
###(3) The experimental condition :  
CPU :　　　Intel(R) Xeon(R) Silver 4114 CPU @ 2.20GHz 10 cores 20 threads × 2  
Memory : 　128GB  
GPU :　　　RTX2080ti × 2  
###(4) Training hyperparameters :  
CASIA-WebFace:　Batch size: 256  
　　　　　　　　　Initial learning rate: 0.05  
　　　　　　　　　Total epoch = 36  
　　　　　　　　　Learning rate scheduler = [22, 31]  
　　　　　　　　　S = 32  

MS-Celeb-1M: 　　 Batch size: 256  
　　　　　　　　　Initial learning rate: 0.05  
 　　　　　　　　　Total epoch = 25  
 　　　　　　　　　Learning rate scheduler = [13, 21]  
 　　　　　　　　　S = 32
## 4. References
[wujiyang/Face_Pytorch](https://github.com/wujiyang/Face_Pytorch)  
[deepinsight/insightface](https://github.com/deepinsight/insightface)   
[Xiaoccer/MobileFaceNet_Pytorch](https://github.com/Xiaoccer/MobileFaceNet_Pytorch)  
[TreB1eN/InsightFace_Pytorch](https://github.com/TreB1eN/InsightFace_Pytorch)  
## 5. If this project is useful to you, please give me a star, love you !

