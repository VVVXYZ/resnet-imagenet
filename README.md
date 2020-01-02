# resnet-imagenet

## 1.数据
<br>
测试集图片数据和模型： https://pan.baidu.com/s/1o54JP3ZFbk_54g0_MOvCwg
测试集图片由ImageNet测试集每个类选50张图片，并且resize到256*256

## 2.问题
<br>
用官网预训练的模型识别图片，当is_training为True(即batch_norm层为训练模式)的时候准确率82%，为False的时候(即batch_norm层非训练模式)的时候准确率约等于0%。测试的收到is_training是否应该设为False？
