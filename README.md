# 咽喉反流诊断
## 难点
这个问题的难点在于数据集过小，稍微训练就很容易导致模型过拟合。

## 思路
对于这个问题有两个解决方案，第一种是采用迁移学习的方法，将别人训练好的模型进行Fine-tune；第二种是进行数据扩充，利用旋转、对称等方法获得更多的图片。特别的，对于数据集过小的问题，2017.8 有一篇关于如何将数据进行扩充的paper介绍了Radial transform的方法用于数据扩充。链接：  
<https://arxiv.org/abs/1708.04347v1>

## 迁移学习
迁移学习的方法的直接思路是把已学训练好的模型参数迁移到新的模型来帮助新模型训练。考虑到图像识别问题存在相关性的，因为大多数神经元都用于提取图像的特征，所以通过迁移学习可以将已经学到的模型参数分享给新模型，帮助我们不用像大多数网络那样从零学习。并且，由于利用了已有模型，大大加强了模型的鲁棒性，使得小数据训练后的模型泛化能力更好，不容易过拟合。  
这里专门使用Fine-tune的方法，将卷积层处理后输出的特征拿出来，然后直接用全连接层分类，仅训练最后的全连接层，获得针对这个问题的模型。一方面自己训练肯定没有办法比大公司、实验室用强大计算机训练的模型更好，另一方面只针对几十张图片训练模型很难提取到的广泛特征。这样子就相当于是在用上百万张图片训练模型。不过这个做法的难点就是要找到下载下来模型的softmax层，不仅要了解模型结构，更为关键的是使用Tensorflow要拿到这一层的命名，准确将这一层的输出保存下来。

## 数据扩充
将图片进行旋转和翻转可让我们的训练图片数量翻倍，从而获得更大的训练集。这个操作很简单不用赘述。需要特别介绍的是利用Radial transform扩充数据集的方法。
具体做法就是任意找图中一点作为轴心，将原图上的每一点用极坐标系表示，然后分别以r、theta 作为新图的横、纵坐标，绘制得到新图片。
<div align=center><img src="https://pic3.zhimg.com/80/v2-a0f5bc32fd5a4647f658f79467bb0796_hd.jpg"/></div>  
算法思路如下：
<div align=center><img width="400" height="250" src="https://pic4.zhimg.com/80/v2-c12c62cd54841966d95bbd45d0311d7d_hd.jpg"/></div>  
原图和新图对比：
<p align="center">
  <img src="https://github.com/Wi-sc/Pharynx-and-larynx-diagnosis/raw/master/figure_2.png" width="300" height="250"/>
  <img src="https://github.com/Wi-sc/Pharynx-and-larynx-diagnosis/raw/master/figure_1.png" width="300" height="250"/>
</p>
这样，想法设法将图片变多获得相对而言大的数据集，对模型的识别能力会有质的提升。


## 结果
仅采用Fine-tune的方法效果很难提升，先后尝试了Inception-v3、Inception-v4、Densnet模型都在70%-80%左右，数据集过小的弊端非常显著。  
最后我将两种方法结合起来，既扩大数据集，又使用比较知名的模型作为预训练模型，可以很容易得到不错的结果。将每个图片经过旋转（顺时针旋转-30度，-25度，……30度）、翻转、Radial Transform（图像关键特征集中在喉管附近，因此取其中心作为轴心）操作得到新的14张图片。这样97张图片就扩充成了1455张。然后利用Inception-v4作为pre-trained model获得（1\*1536）维的特征，之后训练分类层，最终二分类（有病、无病）结果可以达到91.7%，三分类(无病、阴性、阳性)结果可以达到95.9%。  
至于为什么三分类结果反而会比二分类结果好，我猜是因为模型在二分类情况下遇到特征不明显的图片只能随机分，而三分类则给了模型“不能仅通过图片确定”的选项，因此分类结果反而更好。当然具体原因还需要研究。
