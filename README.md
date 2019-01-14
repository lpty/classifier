## 引言
一些常见的神经网络的复现

## 历史
#### 2019-01-14
* bert

    基于Google的BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding进行fine-tuning，构造二分类模型
    
#### 2019-01-10
* transformer

    基于论文 Attention Is All You Need 实现的transformer分类模型

#### 2018-12-27
* attention

    实现论文 Neural Machine Translation by Jointly Learning to Align and Translate中模型

#### 2018-12-17
* dcnn

    实现论文A Convolutional Neural Network for Modelling Sentences中模型。

#### 2018-12-06
* textrnn

    简单实现了Bi-LSTM+softmax,似乎效果并不是特别好。更多相关的模型可以参考：
    Recurrent Neural Network for Text Classification with Multi-Task Learning
    
#### 2018-11-27
* textcnn

    来源于Convolutional Neural Networks for Sentence Classification。
    目前实现的是CNN-rand模型。
    
#### 2018-11-14
* fasttext

    来源于Bag of Tricks for Efficient Text Classification这篇文章。
    目前代码中的模型只是简单的单层神经网络，具体如word级别的n-gram及char级别的n-gram，目前并没有加进去，
    当然增加的方式也很简单，在预处理的地方进行n-grams切分，后续转化为句向量时再做一些reduce_mean之类的操作即可。
    
