基于MNIST手写数字图片集的两层神经网络分类器  

一、环境  
代码为python3版本，需要安装struct、numpy以及matplotlib三个库。  

二、网络架构  
 它包含以下四个部分：    
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
  
三、文件说明  
上传文件中，Random search parameter.py以及Optional parameters.py均包含网络训练，储存模型，导入模型以及可视化的功能。二者的区别在于Random search parameter.py可以进行参数查找（可自行设置查找范围），储存保留查找后最优的模型，而Optional parameters.py则是通过（可自行设定的）参数进行单个模型的训练，不含有自动参数查找的功能。  
mnist压缩包中存放mnist数据集，result压缩包中储存了固定隐藏层数为128，（batch、lrmin、lrmax）参数查找的一些结果和模型，文件夹名为00，代表其在测试集准确率为98.00%，文件夹为31，代表其在测试集准确率为98.31%.(若需加载这些模型，需要更改MLP以及load_the_model函数，使其读取参数只有batch、lrmin、lrmax，其余参数自行设置) 。result—true压缩包包含实验报告中的所有模型。     

四、训练步骤  
  1. 训练开始时，需将.py文件与mnist文件夹放至同一目录下。  
  2. 若使用Random search parameter.py进行训练，需先将注释线以下的导入模型以及可视化部分注释掉，然后只需更改main函数下的basenum的值，程序就可以训练生成basenum个随机模型（若需修改参数范围，需要到train_the_model函数中进行更改），输出各个模型的测试精度以及最优模型测试精度，并在全部训练结束后绘制最优测试准确率的模型loss以及acc曲线，然后将模型存至同目录下的result文件夹中。  
  若使用Optional parameters.pyy进行训练，需先将注释线以下的导入模型以及可视化部分注释掉，然后main函数下的config数组的值（0：隐藏神经元数量，1：最大学习率，2：最小学习率，3：lamda），程序就可以训练生成对应参数的模型，（输出该模型的模型测试精度，在代码中被注释掉了），并在训练结束后绘制测试准确率的模型loss以及acc曲线，然后将模型存至同目录下的result文件夹中。  
  3. 导入模型需将main函数下训练，储存模型以及绘制曲线的函数注释掉，即将读取数据下，注释线上的部分注释掉。然后注释线下的部分不注释，路径填写存放参数矩阵.txt的上层文件夹，进行运行，将输出测试准确率，以及w1、w2的降维参数可视化图。  

注：若需修改神经网络的其他参数，或更改参数查找的其他参数，需自行修改MLP类以及train_the_model函数内容。  
如其中batch大小的设定，也存在不同的设定，本方法目前得到的最好模型为  
隐层神经元数量：128 
epoch: 16  
Ti: [2,4,10]  
batch：32   
lrmin：0.0193  
lrmax:0.5423  
lamda:0  
acc:98.33%  
（但由于训练时每个训练单元使用随机batch个数据进行训练，故参数相同时，训练模型不同，其准确率可能也存在差别）

（不使用SGD，早停止：）
隐层神经元数量：128  
epoch: 10  
Ti: [10]  
batch：1   
lrmin：0  
lrmax:0.02  
lamda:1  
acc:98.31%  


代码中参数查找中各参数范围并不是最佳，如正则化强度参数lamda，代码中范围为：[1,100),而事实上50-60已经是很大的值了，故范围确定有待进一步探讨
