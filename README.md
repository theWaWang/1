本代码是基于MNIST手写数字图片集构建的两层神经网络分类器
它包含以下三个部分
1.训练：
(1)激活函数：隐层激活函数使用leaky_relu（负半轴系数设为0.01），输出分类函数使用softmax函数
(2)正、反向传播，训练集及验证集loss以及准确率的计算（这里使用训练数据集前45000个数据作为训练集，后15000个数据作为验证集）
(3)学习率下降策略:余弦退火策略，这里规定训练的总epoch数为16，然后分别在第3，7个epoch时进行重启，并且在每个学习率下降阶段设定最大学习率和最小学习率。通过重启学习率跳过鞍点，并且通过第7-16个epoch使其训练充分，由于可以设置最小学习率，故可不通过早退出而防止过拟合
(4)L2正则化：设置lamda/45000为正则化强度
(5)优化器SGD:可以在MLP类中设置batch的大小，然后在每个epoch中，训练iteration轮的随机batch个样本
(6)保存模型：保留模型为权重矩阵w1、w2,偏置矩阵b1、b2，以及以数组形式存储参数隐藏层层数、最大学习率、最小学习率、lamda的矩阵config
2.参数查找：学习率，隐藏层大小，正则化强度
使用Random search parameter.py文件可以进行上述四个（学习率有最大最小两个）进行参数查找，其中basenum为设定的参数查找模型个数，参数查找范围在train_the_model函数中自行设定，查找方式是每一轮在每个参数设定范围内随机选取后生成模型，计算测试精确度，待全部训练完成后，保留测试精确率最高的模型参数，并画出它的训练loss，验证loss以及验证精确度。
3.测试：导入模型，用经过参数查找后的模型进行测试，输出分类精度
导入模型使用保存模型的形式，读入config以及四个矩阵。
参数查找后的模型在调用Random search parameter.py文件后会被保留在./result文件夹中，可以通过查看config文件内容，在Optional parameters.py文件中输入参数进行导入，也可以将Random search parameter.py中训练部分注释掉，然后通过下面编写的模型导入函数进行导入，导入后自动计算并打印分类精度。
4.参数可视化，在每个.py文件导入模型下方，都有将w1与w2降维为三通道图片从而可视化的函数及操作步骤，画出的第一副图为w1（如需要，自行保存），第二幅图为w2.
