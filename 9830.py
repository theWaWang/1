import struct
import numpy as np
import matplotlib.pyplot as plt
# 读取原始数据并进行预处理
def data_fetch_preprocessing():
    train_image = open('mnist/train-images.idx3-ubyte', 'rb')
    test_image = open('mnist/t10k-images.idx3-ubyte', 'rb')
    train_label = open('mnist/train-labels.idx1-ubyte', 'rb')
    test_label = open('mnist/t10k-labels.idx1-ubyte', 'rb')

    magic, n = struct.unpack('>II',train_label.read(8))# 读取文件的前8字节
    # 原始数据的标签
    y_train_label = np.array(np.fromfile(train_label,dtype=np.uint8), ndmin=1)# 读取60000个标签数据
    y_train = np.ones((10, 60000)) * 0.01
    for i in range(60000):
        y_train[y_train_label[i]][i] = 0.99# 将结果转化为10维的列向量[0.01,0.99,...,0.01]

    # 测试数据的标签
    magic_t, n_t = struct.unpack('>II',
                                 test_label.read(8))
    y_test = np.fromfile(test_label,
                         dtype=np.uint8).reshape(10000, 1)
    magic, num, rows, cols = struct.unpack('>IIII', train_image.read(16))
    x_train = np.fromfile(train_image, dtype=np.uint8).reshape(len(y_train_label), 784).T

    magic_2, num_2, rows_2, cols_2 = struct.unpack('>IIII', test_image.read(16))
    x_test = np.fromfile(test_image, dtype=np.uint8).reshape(len(y_test), 784).T
    x_train = x_train / 255 * 0.99 + 0.01
    x_test = x_test / 255 * 0.99 + 0.01

    # 关闭打开的文件
    train_image.close()
    train_label.close()
    test_image.close()
    test_label.close()

    return x_train, y_train, x_test, y_test

x_train, y_train, x_test, y_test=data_fetch_preprocessing()

# for i in range(100):
#     if y_train[4,i]==0.99:
#         data=x_train[:,i].reshape(28,28)
#         plt.imshow(data,cmap='Greys',interpolation=None)
#         plt.show()


class MLP():
    def __init__(self,Wmatrix):
        self.lamda=1
        self.learnrate=0.2
        self.epoch=10
        self.batch=1
        self.iteration=int(45000/self.batch)
        self.innum=Wmatrix[0]
        self.hdnum=Wmatrix[1]
        self.outnum=Wmatrix[2]
        self.w1=np.random.randn(self.hdnum,self.innum)*0.01
        self.w2=np.random.randn(self.outnum,self.hdnum)*0.01
        self.b1=np.zeros((self.hdnum,1))
        self.b2=np.zeros((self.outnum,1))
    def leaky_relu(self,x):
        return np.where(x<0,0.01*x,x)
    def dleaky_relu(self,x):
        return np.where(x<0,0.01,1)
    # def tanh(self,x):
    #     return (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
    # def dtanh(self,x):
    #     return 1-self.tanh(x)**2
    def softmax(self,x):
        exps = np.exp(x-np.max(x))
        return exps / np.sum(exps)
    def loss(self,label,prelabel):
        return np.sum(-label*np.log(prelabel+0.0001))
        
    def forward(self,inputdata):
        hidein=np.dot(self.w1,inputdata)+self.b1
        hideout=self.leaky_relu(hidein)
        outin=np.dot(self.w2,hideout)+self.b2
        result=self.softmax(outin)
        return hidein,hideout,result
    
    def bankwards(self,inputdata,hidein,hideout,result,label):
        whz=[]
        wlh=[]
        for i in range(10):
            whz.append([])
            wlh.append([])
            for j in range(10):
                if(i==j):
                    whz[-1].append(result[i]*(1-result[i]))
                else:
                    whz[-1].append(-result[i]*result[j])
            wlh[-1].append(-label[i][0]/(result[i]+0.0001))
        whz=np.array(whz).reshape((10,10))
        wlh=np.array(wlh).reshape((10,1))
        outloss=np.dot(whz,wlh)
        w2loss=np.dot(outloss,hideout.T)
        b2loss=outloss
        hideloss=np.dot(self.w2.T,outloss)*self.dleaky_relu(hidein)
        w1loss=np.dot(hideloss,inputdata.T)
        b1loss=hideloss
        return w1loss,w2loss,b1loss,b2loss
    def train(self,inputdata,label):
        xlinput=inputdata[:,:45000]
        xllabel=label[:,:45000]
        yzinput=inputdata[:,45000:]
        yzlabel=label[:,45000:]
        lr = self.compute_learnrate(0,0.02)
        costlist = []
        acclist = []
        x=[]
        for item in range(self.epoch):
            print('第%d轮次开始执行' % item)
            # w10=self.w1
            # w20=self.w2
            # b10=self.b1
            # b20=self.b2
            for i in range(self.iteration):
                self.learnrate=lr[item*self.iteration+i]
                w2l=np.zeros((self.outnum,self.hdnum))
                w1l=np.zeros((self.hdnum,self.innum))
                b1l=np.zeros((self.hdnum,1))
                b2l=np.zeros((self.outnum,1))
                for j in range(self.batch):
                    # 前向传播
                    z1, h1, h2 = self.forward(xlinput[:, i*self.batch+j].reshape(-1, 1))
                    # 反向传播
                    w1loss,w2loss,b1loss,b2loss=self.bankwards(xlinput[:, i*self.batch+j].reshape(-1, 1),z1, h1, h2,xllabel[:, i*self.batch+j].reshape(-1, 1))
                    w2l += w2loss
                    w1l += w1loss
                    b1l += b1loss
                    b2l += b2loss
                self.w2 -= self.learnrate * (w2l/self.batch + self.lamda/45000*self.w2)
                self.b2 -= self.learnrate * b2l/self.batch
                self.w1 -= self.learnrate * (w1l/self.batch + self.lamda/45000*self.w1)
                self.b1 -= self.learnrate * b1l/self.batch
            cost,acc = self.calculate(yzinput, yzlabel)
            costlist.append(cost)
            acclist.append(acc)
            x.append(item)
            ## 早停止策略
            if(len(acclist)>1 and acc<acclist[-2]):
                # self.w1=w10
                # self.w2=w20
                # self.b1=b10
                # self.b2=b20
                break
        plt.plot(x, costlist)
        plt.show()
        plt.plot(x, acclist)
        plt.show()
        
        
    def calculate(self, inputdata, label):
        precision = 0
        losscost = 0
        for i in range(15000):
            z1, h1, h2 = self.forward(inputdata[:, i].reshape(-1, 1))
            if np.argmax(h2) == np.argmax(label[:,i]):
                precision += 1
            losscost += self.loss(label[:, i].reshape(-1, 1),h2)
        cost = losscost/15000+self.lamda/30000*(np.sum(np.square(self.w1))+np.sum(np.square(self.w2)))
        acc = 100 * precision / 15000
        print("loss: %f" % (cost) )
        print("准确率：%f" % (acc) + "%")
        return cost, acc
    
    ### 学习率余弦退火策略
    def compute_eta_t(self, eta_min, eta_max, T_cur, Ti):
        pi = np.pi
        eta_t = eta_min + 0.5 * (eta_max - eta_min) * (np.cos(pi * T_cur / Ti) + 1)
        return eta_t
    def compute_learnrate(self, eta_min, eta_max):
        Ti = [10]# [2, 3, 5, 10]
        n_batches = self.iteration
        eta_ts = []
        for ti in Ti:
            T_cur = np.arange(0, ti, 1 / n_batches)
            for t_cur in T_cur:
                eta_ts.append(self.compute_eta_t(eta_min, eta_max, t_cur, ti))
        return eta_ts
    
    
    def predict(self, inputdata, label):
        precision = 0
        for i in range(10000):
            z1, h1, h2 = self.forward(inputdata[:, i].reshape(-1, 1))
            # print('模型预测值为:{0},\n实际值为{1}'.format(np.argmax(h2), label[i][0]))
            if np.argmax(h2) == label[i]:
                precision += 1
            # else:
            #     print('模型预测值为:{0},\n实际值为{1}'.format(np.argmax(h2), label[i][0]))
        print("准确率：%f" % (100 * precision / 10000) + "%")

    # 向量训练的预测结果
    def predict_vector(self, inputdata, label):
        z1, h1, h2 = self.forward(inputdata)
        precision=0
        for item in range(10000):
            if np.argmax(h2[:,item])==label[item][0]:
                precision+=1
        print('准确率：{0}%'.format(precision*100/10000))




if __name__ == '__main__':
    
    # 输入层数据维度784，隐藏层100，输出层10
    dl = MLP([784,128,10])
    x_train, y_train, x_test, y_test = data_fetch_preprocessing()
    # print(y_train[:, 1].reshape(-1, 1))
    # 循环训练方法
    dl.train(x_train, y_train)

    # 预测模型
    dl.predict(x_test, y_test)