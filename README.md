
# 2022年兴智杯人工智能大赛国产开发框架工程化应用赛方案

[比赛官网](http://1.203.115.183:81/#/trackDetail?id=24)给出了具体的比赛细则，我们智控所智能检测团队最终取得了[排行榜](http://www.aiinnovation.com.cn/#/aiaeDetail?id=558)的第二名。

## 环境要求

- PaddlePaddle 2.2
- OS 64位操作系统
- Python 3(3.5.1+/3.6/3.7/3.8/3.9)，64位版本
- pip/pip3(9.0.1+)，64位版本
- CUDA >= 10.2
- cuDNN >= 7.6

## 安装说明

### 1. 安装PaddlePaddle

```
# CUDA10.2
python -m pip install paddlepaddle-gpu==2.2.2 -i https://mirror.baidu.com/pypi/simple

# CPU
python -m pip install paddlepaddle==2.2.2 -i https://mirror.baidu.com/pypi/simple
```
- 更多CUDA版本或环境快速安装，请参考[PaddlePaddle快速安装文档](https://www.paddlepaddle.org.cn/install/quick)
- 更多安装方式例如conda或源码编译安装方法，请参考[PaddlePaddle安装文档](https://www.paddlepaddle.org.cn/documentation/docs/zh/install/index_cn.html)

请确保您的PaddlePaddle安装成功并且版本不低于需求版本。使用以下命令进行验证。

```
# 在您的Python解释器中确认PaddlePaddle安装成功
>>> import paddle
>>> paddle.utils.run_check()

# 确认PaddlePaddle版本
python -c "import paddle; print(paddle.__version__)"
```
**注意**
1. 如果您希望在多卡环境下使用PaddleDetection，请首先安装NCCL

### 2. 安装PaddleDetection

```
git clone https://github.com/jiayisong/GearDefectDetector.git
# 安装其他依赖
cd GearDefectDetector
pip install -r requirements.txt
# 编译安装paddledet
python setup.py install
```

**注意**
1. 若您使用的是Windows系统，由于原版cocoapi不支持Windows，`pycocotools`依赖可能安装失败，可采用第三方实现版本，该版本仅支持Python3

    ```pip install git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI```

2. 若您使用的是Python <= 3.6的版本，安装`pycocotools`可能会报错`distutils.errors.DistutilsError: Could not find suitable distribution for Requirement.parse('cython>=0.27.3')`, 您可通过先安装`cython`如`pip install cython`解决该问题


安装后确认测试通过：

```
python ppdet/modeling/tests/test_architectures.py
```

测试通过后会提示如下信息：

```
.......
----------------------------------------------------------------------
Ran 7 tests in 12.816s
OK
```
## 数据准备
[训练数据](http://www.aiinnovation.com.cn/#/aiaeDetail?id=558)由比赛方给出。
### 拆分数据集为训练集和验证集
按照9:1的比例，划分训练集和验证集。
使用飞桨全流程开发工具[PaddleX](https://gitee.com/paddlepaddle/PaddleX)提供的一键切分数据集解决方案。
得到的标注文件为[train.json](dataset/齿轮异常检测/train.json)和[val.json](dataset/齿轮异常检测/val.json)

## 训练模型
### 两阶段训练策略
首先根据自己数据集的存放路径配置好配置文件的数据集路径，本项目所使用的配置文件为[configs/testB](configs/testB)文件夹下的所有配置文件
1. 强数据增强训练
```
python3 tools/train.py --config configs/testB/ppyoloe_adamw_flip3_clip.yml  --use_vdl=True --vdl_log_dir=./testB/ppyoloe_adamw_flip3_clip/ --eval --amp
python3 tools/train.py --config configs/testB/ppyoloe_adamw_flip3_clip_m.yml  --use_vdl=True --vdl_log_dir=./testB/ppyoloe_adamw_flip3_clip_m/ --eval --amp
python3 tools/train.py --config configs/testB/ppyoloe_adamw_flip3_clip_s.yml  --use_vdl=True --vdl_log_dir=./testB/ppyoloe_adamw_flip3_clip_s/ --eval --amp
python3 tools/train.py --config configs/testB/ppyoloe_adamw_flip3_clip_x.yml  --use_vdl=True --vdl_log_dir=./testB/ppyoloe_adamw_flip3_clip_x/ --eval --amp
```
本项目单模型默认均使用翻转增强，训练结果如下

训练好的模型权重都在如下文件中：
通过网盘分享的文件：output.zip
链接: https://pan.baidu.com/s/1o1j047EmVm_Z1_ArxAarBQ?pwd=6h34 提取码: 6h34

| 模型 |  配置文件        |  验证集mAP                |模型权重     |
| :-------------: | :-------------: | :----------: | :------------: |
| ppyoloe-s |   [config](configs/testB/ppyoloe_adamw_flip3_clip_s.yml)   | 81.1  | [weight](output/ppyoloe_adamw_flip3_clip_s/best_model.pdparams)|
| ppyoloe-m |   [config](configs/testB/ppyoloe_adamw_flip3_clip_m.yml)   | 81.5    |[weight](output/ppyoloe_adamw_flip3_clip_m/best_model.pdparams)|
| ppyoloe-l |   [config](configs/testB/ppyoloe_adamw_flip3_clip.yml)   | 80.8   |[weight](output/ppyoloe_adamw_flip3_clip/best_model.pdparams) |
| ppyoloe-x |   [config](configs/testB/ppyoloe_adamw_flip3_clip_x.yml)   |81.1    |[weight](output/ppyoloe_adamw_flip3_clip_x/best_model.pdparams)|
注：ppyoloe-l模型翻转融合方式为nms，未使用WBF，其余模型均使用WBF翻转融合
2. 无数据增强多尺度微调
```
python3 tools/train.py --config configs/testB/ppyoloe_adamw_flip3_clip2.yml  --use_vdl=True --vdl_log_dir=./testB/ppyoloe_adamw_flip3_clip2/ --eval --amp
python3 tools/train.py --config configs/testB/ppyoloe_adamw_flip3_clip_m2.yml  --use_vdl=True --vdl_log_dir=./testB/ppyoloe_adamw_flip3_clip_m2/ --eval --amp
python3 tools/train.py --config configs/testB/ppyoloe_adamw_flip3_clip_s2.yml  --use_vdl=True --vdl_log_dir=./testB/ppyoloe_adamw_flip3_clip_s2/ --eval --amp
python3 tools/train.py --config configs/testB/ppyoloe_adamw_flip3_clip_x2.yml  --use_vdl=True --vdl_log_dir=./testB/ppyoloe_adamw_flip3_clip_x2/ --eval --amp
python3 tools/train.py --config configs/testB/ppyoloe_adamw_flip3_clip2_55.yml  --use_vdl=True --vdl_log_dir=./testB/ppyoloe_adamw_flip3_clip2_55/ --eval --amp
python3 tools/train.py --config configs/testB/ppyoloe_adamw_flip3_clip_m2_55.yml  --use_vdl=True --vdl_log_dir=./testB/ppyoloe_adamw_flip3_clip_m2_55/ --eval --amp
python3 tools/train.py --config configs/testB/ppyoloe_adamw_flip3_clip_s2_55.yml  --use_vdl=True --vdl_log_dir=./testB/ppyoloe_adamw_flip3_clip_s2_55/ --eval --amp
python3 tools/train.py --config configs/testB/ppyoloe_adamw_flip3_clip_x2_55.yml  --use_vdl=True --vdl_log_dir=./testB/ppyoloe_adamw_flip3_clip_x2_55/ --eval --amp
python3 tools/train.py --config configs/testB/ppyoloe_adamw_flip3_clip2_75.yml  --use_vdl=True --vdl_log_dir=./testB/ppyoloe_adamw_flip3_clip2_75/ --eval --amp
python3 tools/train.py --config configs/testB/ppyoloe_adamw_flip3_clip_m2_75.yml  --use_vdl=True --vdl_log_dir=./testB/ppyoloe_adamw_flip3_clip_m2_75/ --eval --amp
python3 tools/train.py --config configs/testB/ppyoloe_adamw_flip3_clip_s2_75.yml  --use_vdl=True --vdl_log_dir=./testB/ppyoloe_adamw_flip3_clip_s2_75/ --eval --amp
python3 tools/train.py --config configs/testB/ppyoloe_adamw_flip3_clip_x2_75.yml  --use_vdl=True --vdl_log_dir=./testB/ppyoloe_adamw_flip3_clip_x2_75/ --eval --amp
```

训练结果

| 模型序号 | 模型 | 微调尺寸 |  配置文件        |  验证集mAP          |  测试集B榜mAP                    |模型权重     |
| :-------------:| :-------------: | :-------------: | :----------: | :------------: | :------------: |:------------: |
|1| ppyoloe-s |0.65|   [s](configs/testB/ppyoloe_adamw_flip3_clip_s2.yml)   | 81.1 | 78.2   | [weight](output/ppyoloe_adamw_flip3_clip_s2/best_model.pdparams)|
|2| ppyoloe-m |0.65|   [m](configs/testB/ppyoloe_adamw_flip3_clip_m2.yml)   |81.5 | 78.1   | [weight](output/ppyoloe_adamw_flip3_clip_m2/best_model.pdparams)|
|3| ppyoloe-l |0.65|   [l](configs/testB/ppyoloe_adamw_flip3_clip2.yml)   | 81.6 | 78.7|   [weight](output/ppyoloe_adamw_flip3_clip2/best_model.pdparams)|
|4| ppyoloe-x |0.65|   [x](configs/testB/ppyoloe_adamw_flip3_clip_x2.yml)   | 81.6| 78.6   | [weight](output/ppyoloe_adamw_flip3_clip_x2/best_model.pdparams)|
|5| ppyoloe-s |0.55|   [s](configs/testB/ppyoloe_adamw_flip3_clip_s2_55.yml)   |80.6 | -   | [weight](output/ppyoloe_adamw_flip3_clip_s2_55/best_model.pdparams)|
|6| ppyoloe-m |0.55|   [m](configs/testB/ppyoloe_adamw_flip3_clip_m2_55.yml)   |80.8 | -   | [weight](output/ppyoloe_adamw_flip3_clip_m2_55/best_model.pdparams)|
|7| ppyoloe-l |0.55|   [l](configs/testB/ppyoloe_adamw_flip3_clip2_55.yml)   | 81.1 | 78.5  | [weight](output/ppyoloe_adamw_flip3_clip2_55/best_model.pdparams)|
|8| ppyoloe-x |0.55|   [x](configs/testB/ppyoloe_adamw_flip3_clip_x2_55.yml)   | 80.9| -   | [weight](output/ppyoloe_adamw_flip3_clip_x2_55/best_model.pdparams)|
|9| ppyoloe-s |0.75|   [s](configs/testB/ppyoloe_adamw_flip3_clip_s2_75.yml)   | 81.0 | -   | [weight](output/ppyoloe_adamw_flip3_clip_s2_75/best_model.pdparams)|
|10| ppyoloe-m |0.75|   [m](configs/testB/ppyoloe_adamw_flip3_clip_m2_75.yml)   |81.4 | -   | [weight](output/ppyoloe_adamw_flip3_clip_m2_75/best_model.pdparams)|
|11| ppyoloe-l |0.75|   [l](configs/testB/ppyoloe_adamw_flip3_clip2_75.yml)   | 81.8 | 78.4   | [weight](output/ppyoloe_adamw_flip3_clip2_75/best_model.pdparams)|
|12| ppyoloe-x |0.75|   [x](configs/testB/ppyoloe_adamw_flip3_clip_x2_75.yml)   | 81.6| - |   [weight](output/ppyoloe_adamw_flip3_clip_x2_75/best_model.pdparams)|

注：ppyoloe-l模型的日志文件中的翻转融合方式为nms，未使用WBF

所有模型的训练日志都在模型权重的相同文件夹下
## 模型推理
先使用不同尺寸微调后的模型进行多尺度翻转增强推理，融合方法使用WBF(运行前需要配置好[tools/infer_submit_multimodel.py](tools/infer_submit_multimodel.py)第57行的测试集路径)
```
python3 tools/infer_submit_multimodel.py
```
再把每个模型的结果进行集成，融合方法使用WBF(运行前需要配置好[tools/wbf_fuse.py](tools/wbf_fuse.py)第7行的测试集路径)
```
python3 tools/wbf_fuse.py
```
得到最终的输出文件位于output/SMLX_556575/upload.json

我们还尝试了不同融合方式，结果如下

| 融合模型的序号   |验证集mAP|  测试集B榜mAP      |
| :-------------:| :-------------: | :-------------: | 
|  3，7，11    |  82.2|     79.0          |
|  1，2，3    |    82.3|    79.1          |
|  1，2，3，4    |   82.4|     79.2          |
|  1-12（最终方案）    |  82.5|      79.2          |

