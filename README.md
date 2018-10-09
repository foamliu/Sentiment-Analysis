# 细粒度用户评论情感分析

在线评论的细粒度情感分析对于深刻理解商家和用户、挖掘用户情感等方面有至关重要的价值，并且在互联网行业有极其广泛的应用，主要用于个性化推荐、智能搜索、产品反馈、业务安全等。

## 依赖
- Python 3.5
- PyTorch 0.4

## 数据集
使用 AI Challenger 2018 的细粒度用户评论情感分析数据集，共包含6大类20个细粒度要素的情感倾向。

### 数据说明

数据集分为训练、验证、测试A与测试B四部分。数据集中的评价对象按照粒度不同划分为两个层次，层次一为粗粒度的评价对象，例如评论文本中涉及的服务、位置等要素；层次二为细粒度的情感对象，例如“服务”属性中的“服务人员态度”、“排队等候时间”等细粒度要素。评价对象的具体划分如下表所示。

![image](https://github.com/foamliu/Sentiment-Analysis/raw/master/images/PingJiaDuiXiang.JPG)

每个细粒度要素的情感倾向有四种状态：正向、中性、负向、未提及。使用[1,0,-1,-2]四个值对情感倾向进行描述，情感倾向值及其含义对照表如下所示：

![image](https://github.com/foamliu/Sentiment-Analysis/raw/master/images/QingGanQingXiang.JPG)

数据标注示例如下：

![image](https://github.com/foamliu/Sentiment-Analysis/raw/master/images/ShuJuShiLi.JPG)

请到[官网链接](https://challenger.ai/dataset/fsaouord2018)下载数据集。



## 用法

### 数据预处理
提取训练和验证样本：
```bash
$ python pre-process.py
```

### 训练
```bash
$ python train.py
```

要想可视化训练过程，在终端中运行：
```bash
$ tensorboard --logdir path_to_current_dir/logs
```

### Demo
下载 [预训练模型](https://github.com/foamliu/Sentiment-Analysis/releases/download/v1.0/model.85-0.7657.hdf5) 放在 models 目录然后执行:

```bash
$ python demo.py
```