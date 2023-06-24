# Hands-on-EEG
## Graduation project
Handle EEG signal for classication using ANN  
（这是本项目的公开分支，在使用cuda上的一些实践可切换至其他分支）
### 提示（hint）
本项目使用EMOTIV Plex采集脑电信号，并使用EMOTIV PRO软件进行导出，导出格式为CSV格式，
设计神经网络模型，对脑电信号进行分类，最终实现脑电信号的分类。  
下面是本项目的处理流程与文件结构说明：  
raw csv file -> [open/NN/cnn_old/code/process.ipynb](1)(包含文件名修饰和冗余文件删除，数据切片，文件清点，数据集分割) -> [open\NN\dataset\eegdata\raw]()  
|

-> [open/NN/rnn/code/RNN.ipynb]()/[open/NN/lstm/code/LSTM.ipynb]()/[open/NN/transformer_old/code/transformer3000.ipynb]() (包含适应对应模型的数据形状的处理，数据集分割)   
|

-> [open/NN/dataset/128_s100_slice.ipynb]() (使用滑动窗口切片原始数据) --> [open/NN/dataset/128_s100](128数据点长数据集) --> [open/NN/cnn/code/cnn_3_conv_128_s100.ipynb]() (对应模型训练)  
-> [open/NN/transformer/code/transformer_32_128_s100.ipynb](对应模型训练)  

为确保调试过程的方便，本文文件路径均为绝对路径，如需使用，请自行修改。（大部分路径已做修改，数据集路径如有需要可定位至NN\dataset确定相关路径）  

### 文件夹说明
- [open/NN/cnn_old/code/]()：早期CNN模型代码（勿用，仅作参考）
- [open/NN/dataset/]()：数据集
- [open/NN/lstm/code/]()：LSTM模型代码
- [open/NN/rnn/code/]()：RNN模型代码
- [open/NN/transformer_old/code/]()：早期Transformer模型代码（勿用，仅作参考）
- [open/NN/cnn/code/]()：CNN模型代码
- [open/NN/transformer/code/]()：Transformer模型代码
- [open/NN/dataset]() ：数据集
- [open/origin_wave_display/]()：原始波形图绘制
- [open/performance]() ：模型性能测试
- [open/preprocess/]()：数据预处理
- [open/pywavelet/]()：小波包变换
- [open/spectrogram/]()：时频图绘制
- [pic/]()：绘制的图片
- [normal/model/]()：训练的一些模型
- [open/export/]()：导出的模型为ONNX
- [open/app/]()：CAM实现及数据可视化

出于对实际受试人的保护，数据集不公开。可向相关人求取。

