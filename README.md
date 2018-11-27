## 引言
一些常见的神经网络的复现

## 历史
#### 2018-11-27
* textcnn

    来源于Convolutional Neural Networks for Sentence Classification。
    目前实现的是CNN-rand模型。
    
#### 2018-11-14
* fasttext

    来源于Bag of Tricks for Efficient Text Classification这篇文章。
    目前代码中的模型只是简单的单层神经网络，具体如word级别的n-gram及char级别的n-gram，目前并没有加进去，
    当然增加的方式也很简单，在预处理的地方进行n-grams切分，后续转化为句向量时再做一些reduce_mean之类的操作即可。
    
