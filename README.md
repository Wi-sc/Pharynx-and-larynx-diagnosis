# Pharynx-and-larynx-diagnosis
这个问题的难点在于数据集过小，稍微训练就很容易导致模型过拟合。
对于这个问题有两个解决方案，第一种是采用迁移学习的方法，将别人训练好的模型进行Fine-tuning；第二种是进行数据扩充，利用旋转、对称等方法获得更多的图片。特别的，对于数据集过小的问题，2017.8 有一篇关于如何将数据进行扩充的paper介绍了Radial transform的方法用于数据扩充。链接如下：
https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1708.04347v1
