初步实现了Joint entity and relation extraction based on a hybrid neural network论文问中的网络结构。
tensorflow网络图存放在data中

不足:RC卷积部分一次只能卷积一个句子，不能实现batch size
     RC卷积部分没有做到end2end，需要在计算图外面自己定义规则函数