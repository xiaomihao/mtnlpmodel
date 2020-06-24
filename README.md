# mtnlpmodel

基于 TensorFlow 的多任务NLP算法（目前包含 `命名实体识别(NER)` 和 `子功能点分类（Sub-func Classification）`，更多算法正在持续添加中）。

## 特色
* 通用的序列标注：能够解决通用的序列标注问题：分词、词性标注和实体识别仅仅是特例。
* Tag schema free: 你可以选择你想用的任何 Tagset。依赖于 [tokenizer_tools](https://github.com/howl-anderson/tokenizer_tools) 提供的编码、解码功能
* 基于 TensorFlow Keras: 便于功能扩展和快速验证。

## 功能
```
* 文本分类： 我要听七里香。=> ( label：听音乐 )
* 实体识别： 我要听七里香。=> ( 我要听**七里香**<歌名>。)
```
## 内容结构
### 代码结构
```
mtnlpmodel
├── server/                       // server:inference->用于模型推理; evaluation->用于模型评估     
├── train.py                      // 模型训练入口
├── core.py                       // 模型构造，包括从零训练和finetuning
└── utils/                        // 与模型训练相关的内容
    ├── deliverablemodel_util.py  // 保存模型相关组件
    ├── loss_func_util.py         // 一些损失函数
    ├── lrset_util.py             // 学习率修改组件
    ├── input_process_util.py     // 输入数据预处理组件
    ├── model_util.py             // 一些layer和模块
    ├── optimizer_util.py         // 处理模型保存、统计等任务的组件
    └── triplet_loss_util         // triplet_loss相关组件，目前未启用

```
### 安装
```
pip install mtnlpmodel
```
### Train
* train.py：融合模型训练入口（包括：1.正常训练 'random_initial'；
                                 2.fine-tuning 'load weights'）。
```
# WORKDIR为configure.yaml所在的路径
python -m mtnlpmodel.train.py  #启动多输入模型训练
```
##### 正常训练
```
vi configure.yaml 

# finetune=false 
   # 以当前configure配置从头训练模型，模型参数随机初始化
   # 从头开始训练模型
```
##### fine-tuning
```
vi configure.yaml

# finetune=true 
   # 加载预训练模型以configure配置微调模型，模型参数由预训练模型填充；
   # 在train.py中可根据实际权重参数存放位置修改model_weights_path
   # train.py中recommend_freeze_list的层为冻结层，不参与训练更新，可自行配置。
```
###### 模型选择
```
vi configure.yaml 

 # model_choice: "VIRTUAL_EMBEDDING"  # 选择虚拟embedding融合结构
 # model_choice: "CLS2NER_INPUT"      # 选择虚拟关键字融合结构
 # model_choice: "OTHER"              # 选择其他（任务独立结构）

 # Arcloss: true                      # 选择Arcsoftmax loss作为分类loss
 # Arcloss: false                     # 选择softmax loss作为分类loss
```
### 可视化训练
```
tensorboard --logdir=./results/summary_log_dir
```
### 模型推理和评估
* Inference 推理
```
# 修改 server/inference/configure.json 配置inference所需模型、数据的路径
python server/inference/inference.py
```
* Evaluation 评估
```
# 修改 server/evaluation/configure.json 配置evaluation所需模型、数据的路径
python server/evaluation/evaluation.py
```

### 推广
* tokenizer_tools
* deliverable_model
* corpus_flow
```
> tokenizer_tools 强大的语料文本处理工具，本项目中语料数据预处理环节大量调用该工具
> deliverable_model 强大的模型封装工具，包含各种预处理和后处理，本项目推理和评估大量使用该工具
> corpus_flow 强大的语料增强工具，实现了语料增强
```

### TODO
* 数据增强
* 功能增加