## 1. Python 运行配置

```
conda env create -f env.yml
```

## 2. data 存储相关的数据、预处理后的数据、预处理代码

### 目录：

- images  存储规则分割后的图像（此处example以512*512为例）
- labels 对应的标签
- list 训练图像、测试图像的列表（格式见示例）
- get_data.py 预处理代码

```
# 生成标签数据（分别为边缘标签、面域标签、方向注意力标签）
$ cd data
$ python get_data.py -m e -s labels -t edge
$ python get_data.py -m r -s labels -t region
$ python get_data.py -m d -s labels -t direction
```

执行以上代码后，生成目录：

- edge 
- region
- direction

## 3. model 运行模型

1. 在config.yaml中设置路径：（以下为默认路径，具体按需更改）

```
IMAGE_PATH: "../data/images/"
LABEL_PATH: "../data/labels/"
EDGE_PATH: "../data/edge/"
REGION_PATH: "../data/region/"
DIR_PATH: "../data/direction/"
TRAIN_LIST: "../data/list/train_list.txt"
TEST_LIST: "../data/list/test_list.txt"
```

2. 进入Python的虚拟环境中；

3. 训练模型：

```
$ cd model
$ python main.py 
```

4. 测试模型：(参数"200"表示使用训练到第200个epoch结果的权重)，可视化结果输出在model目录下的logs_ours/pred_threshold中

```
$ cd model
$ python pred.py 200
```

