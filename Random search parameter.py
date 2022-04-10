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
        y_train[y_train_label[i]][i] = 0.91# 将结果转化为10维的列向量[0.01,0.99,...,0.01]

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

# for i in range(100):
#     if y_train[4,i]==0.99:
#         data=x_train[:,i].reshape(28,28)
#         plt.imshow(data,cmap='Greys',interpolation=None)
#         plt.show()


class MLP():
    def __init__(self,Wmatrix,lamda,lratemin,lratemax):
        self.lamda=lamda
        self.learnrate=0.5
        self.epoch=16
        self.batch=100
        self.lratemax=lratemax
        self.lratemin=lratemin
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
        # idx1 = np.random.randint(low=0,high=60000,size=(45000,),dtype='int')
        # # idx1 = np.array([i for i in range(45000)])
        # # idx2 = np.random.randint(low=0,high=60000,size=(15000,),dtype='int')
        # idx2 = np.array([i for i in range(45000,60000)])
        xlinput=inputdata[:,:45000]
        xllabel=label[:,:45000]
        yzinput=inputdata[:,45000:]
        yzlabel=label[:,45000:]
        lr = self.compute_learnrate(self.lratemin,self.lratemax)
        costlist0=[]
        costlist = []
        acclist = []
        x=[]
        for item in range(self.epoch):
            # print('第%d轮次开始执行' % item)
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
                idx3 = np.random.randint(low=0,high=45000,size=(self.batch,),dtype='int')
                for j in range(self.batch):
                    # 前向传播 i*self.batch+j
                    z1, h1, h2 = self.forward(xlinput[:, idx3[j]].reshape(-1, 1))
                    # 反向传播
                    w1loss,w2loss,b1loss,b2loss=self.bankwards(xlinput[:, idx3[j]].reshape(-1, 1),z1, h1, h2,xllabel[:, idx3[j]].reshape(-1, 1))
                    w2l += w2loss
                    w1l += w1loss
                    b1l += b1loss
                    b2l += b2loss
                self.w2 -= self.learnrate * (w2l/self.batch + self.lamda/45000*self.w2)
                self.b2 -= self.learnrate * b2l/self.batch
                self.w1 -= self.learnrate * (w1l/self.batch + self.lamda/45000*self.w1)
                self.b1 -= self.learnrate * b1l/self.batch
            cost0,acc0 = self.calculate(xlinput, xllabel, 45000)
            cost,acc = self.calculate(yzinput, yzlabel, 15000)
            costlist0.append(cost0)
            costlist.append(cost)
            acclist.append(acc)
            x.append(item)
            ## 早停止策略
            # if(len(acclist)>9 and acc<acclist[-2]):
            #     # self.w1=w10
            #     # self.w2=w20
            #     # self.b1=b10
            #     # self.b2=b20
            #     break
        return self.w1,self.w2,self.b1,self.b2,x,costlist0,costlist,acclist
        
        
    def calculate(self, inputdata, label, num):
        precision = 0
        losscost = 0
        for i in range(num):
            z1, h1, h2 = self.forward(inputdata[:, i].reshape(-1, 1))
            if np.argmax(h2) == np.argmax(label[:, i]):
                precision += 1
            losscost += self.loss(label[:, i].reshape(-1, 1),h2)
        cost = losscost/num+self.lamda/(2*num)*(np.sum(np.square(self.w1))+np.sum(np.square(self.w2)))
        acc = 100 * precision / num
        # print("loss: %f" % (cost) )
        # print("准确率：%f" % (acc) + "%")
        return cost, acc
    
    ### 学习率余弦退火策略
    def compute_eta_t(self, eta_min, eta_max, T_cur, Ti):
        pi = np.pi
        eta_t = eta_min + 0.5 * (eta_max - eta_min) * (np.cos(pi * T_cur / Ti) + 1)
        return eta_t
    def compute_learnrate(self, eta_min, eta_max):
        Ti = [2, 4, 10]#[self.epoch]# [2, 3, 5, 10]
        n_batches = self.iteration
        eta_ts = []
        for ti in Ti:
            T_cur = np.arange(0, ti, 1 / n_batches)
            for t_cur in T_cur:
                eta_ts.append(self.compute_eta_t(eta_min, eta_max, t_cur, ti))
        return eta_ts

    # 向量训练的预测结果
    def predict(self, inputdata, label):
        z1, h1, h2 = self.forward(inputdata)
        precision=0
        for item in range(10000):
            if np.argmax(h2[:,item])==label[item][0]:
                precision+=1
        # print('准确率：{0}%'.format(precision*100/10000))
        return precision/10000

def randomst(num,sectionlw,sectionhg):
    return np.around(sectionlw+(sectionhg-sectionlw)*np.random.rand(num),4)
def randomit(num,sectionlw,sectionhg):
    return np.random.randint(low=sectionlw,high=sectionhg,size=(num,),dtype='int')
def train_the_model(x_train, y_train, basenum):
    acc=0
    accmax=0
    for i in range(basenum):
        hidelayer=int(randomit(1,9,14)[0]**2)
        layermatrix=[784,hidelayer,10]
        rw1=np.zeros((layermatrix[1],784))
        rw2=np.zeros((10,layermatrix[1]))
        rb1=np.zeros((layermatrix[1],1))
        rb2=np.zeros((10,1))
        lratemax=randomst(1,0.1,1)[0]
        lratemin=randomst(1,0.01,0.1)[0]
        lamda=randomit(1,1,100)[0]
        print('当前训练模型数：{0}'.format(i+1))
        print('选取参数为：hidelayer:{0},lratemin:{1},lratemax:{2},lamda:{3}'.format(hidelayer,lratemin,lratemax,lamda))
        dl = MLP(layermatrix, lamda,lratemin,lratemax)
        w1,w2,b1,b2,x,costlist0,costlist,acclist=dl.train(x_train, y_train)
        acc=dl.predict(x_test, y_test)
        print('准确率：{0}%'.format(100*acc))
        if(acc>=accmax):
            rw1,rw2,rb1,rb2,rx=w1,w2,b1,b2,x
            rcostlist0=costlist0
            rcostlist=costlist
            racclist=acclist
            accmax=max(acc,accmax)
            maxnum=[hidelayer,lratemin,lratemax,lamda]
        print('最优选取参数为：hidelayer:{0},lratemin:{1},lratemax:{2},lamda:{3}'.format(maxnum[0],maxnum[1],maxnum[2],maxnum[3]))
        print('最优准确率：{0}%'.format(100*accmax))
    return rw1,rw2,rb1,rb2,rx,rcostlist0,rcostlist,racclist,maxnum

def save_the_model(w1,w2,b1,b2,config):
    np.savetxt('./result/w1.txt', w1, fmt="%.3f")
    np.savetxt('./result/w2.txt', w2, fmt="%.3f")
    np.savetxt('./result/b1.txt', b1, fmt="%.3f")
    np.savetxt('./result/b2.txt', b2, fmt="%.3f")
    np.savetxt('./result/config.txt', config, fmt="%.4f")
def draw_the_curve(x,costlist0,costlist,acclist):
    plt.figure(1)
    #第一行第一列图形
    ax1 = plt.subplot(2,2,1)
    #第一行第二列图形
    ax2 = plt.subplot(2,2,2)
    #第二行第二列图形
    ax3 = plt.subplot(2,2,4)

    plt.sca(ax1)
    plt.plot(x,costlist0,'r-.')
    plt.sca(ax2)
    plt.plot(x,costlist,'r-.')
    plt.sca(ax3)
    plt.plot(x,acclist,'g--')
    plt.show()
    
class PCA():
    # n_components:保留主成分的个数
    def __init__(self, n_components):
        self.n_components = n_components

    # X是初始数据，d行n列，n个d维向量，每列是一个向量
    def fit(self, X):
        n,self.d = X.shape
        assert self.n_components <= self.d
        assert self.n_components <= n

        mean = np.mean(X, axis=0)
        cov = np.cov(X-mean, rowvar=False)
        eigenvalue, featurevector = np.linalg.eig(cov)
        
        # 将特征值升序排列，index是原数组中的下标
        index = np.argsort(eigenvalue)
        # 取最大的n个特征值对应的下标
        n_index = index[-self.n_components:]
        # 取对应的列
        self.W = featurevector[:,n_index]
    
    def transform(self, X):
        n,d = X.shape
        # 数据维度必须一样
        assert d == self.d
        mean = np.mean(X, axis=0)
        X = X - mean
        return np.dot(X,self.W) 
    
    
def lode_the_model(path):
    config=np.loadtxt(path+'/config.txt')
    hidelayer=int(config[0])
    layermatrix=[784,hidelayer,10]
    lratemin=config[1]
    lratemax=config[2]
    lamda=config[3]
    dl = MLP(layermatrix, lamda,lratemin,lratemax)
    dl.w1=np.loadtxt(path+'/w1.txt')
    dl.w2=np.loadtxt(path+'/w2.txt')
    dl.b1=np.loadtxt(path+'/b1.txt').reshape((hidelayer,1))
    dl.b2=np.loadtxt(path+'/b2.txt').reshape((10,1))
    return dl

if __name__ == '__main__':
    x_train, y_train, x_test, y_test = data_fetch_preprocessing()
    basenum=1
    rw1,rw2,rb1,rb2,rx,rcostlist0,rcostlist,racclist,rconfig=train_the_model(x_train, y_train, basenum)
    save_the_model(rw1,rw2,rb1,rb2,rconfig)
    draw_the_curve(rx,rcostlist0,rcostlist,racclist)

    ##################################################################################
    path='./result'
    dl=lode_the_model(path)
    acc=dl.predict(x_test, y_test)
    print('准确率：{0}%'.format(100*acc))
    
    w1=dl.w1
    w2=dl.w2
    hdnum=dl.hdnum
    
    # w1降维至784*3 1张28*28三通道图片可视化w1
    data = w1.T
    pca = PCA(3)
    pca.fit(data)
    data_pca = pca.transform(data).reshape((28,28,3))
    # data_pca = 0.114*data_pca[:,:,0]+0.387*data_pca[:,:,1]+0.299*data_pca[:,:,2]
    maxnum=np.max(data_pca)
    minnum=np.min(data_pca)
    data_pca=(data_pca-minnum)/(maxnum-minnum)
    plt.imshow(data_pca,cmap=plt.get_cmap('gray'))
    plt.show()
    
    # w2降维至hdnum*3 1张size*size三通道图片可视化w2
    size=int(hdnum**0.5)
    data = w2.T
    pca = PCA(3)
    pca.fit(data)
    data_pca = pca.transform(data).reshape((size,size,3))
    # data_pca = 0.114*data_pca[:,:,0]+0.387*data_pca[:,:,1]+0.299*data_pca[:,:,2]
    maxnum=np.max(data_pca)
    minnum=np.min(data_pca)
    data_pca=(data_pca-minnum)/(maxnum-minnum)
    plt.imshow(data_pca,cmap=plt.get_cmap('gray'))
    plt.show()
    
    ## 下方注释内容为可视化隐藏层神经元数量为128时的参数
    # # 128*784->8*16*784->8*784 8张28*28图片可视化w1
    # wlist1=[]
    # for i in range(8):
    #     w11=np.zeros((1,784))
    #     for j in range(16):
    #         w11+=w1[i*10+j,:].reshape((1,784))
    #     wlist1.append(w11)
    
    # result=np.array(wlist1)
    # # result=np.maximum(result,-result)
    # # thenum=np.average(result)
    # # result[np.where(result<thenum)]=0
    # plt.figure(1)
    # ax1 = plt.subplot(2,4,1)
    # ax2 = plt.subplot(2,4,2)
    # ax3 = plt.subplot(2,4,3)
    # ax4 = plt.subplot(2,4,4)
    # ax5 = plt.subplot(2,4,5)
    # ax6 = plt.subplot(2,4,6)
    # ax7 = plt.subplot(2,4,7)
    # ax8 = plt.subplot(2,4,8)
    # plt.sca(ax1)
    # plt.imshow(result[0].reshape((28,28)))
    # plt.sca(ax2)
    # plt.imshow(result[1].reshape((28,28)))
    # plt.sca(ax3)
    # plt.imshow(result[2].reshape((28,28)))
    # plt.sca(ax4)
    # plt.imshow(result[3].reshape((28,28)))
    # plt.sca(ax5)
    # plt.imshow(result[4].reshape((28,28)))
    # plt.sca(ax6)
    # plt.imshow(result[5].reshape((28,28)))
    # plt.sca(ax7)
    # plt.imshow(result[6].reshape((28,28)))
    # plt.sca(ax8)
    # plt.imshow(result[7].reshape((28,28)))
    # plt.show()
    
    # # 10*128 10张8*16图片可视化w2
    # wlist2=[]
    # for i in range(10):
    #     wlist2.append(w2[i,:].reshape((1,128)))
    # result=np.array(wlist2)
    # plt.figure(1)
    # ax1 = plt.subplot(2,5,1)
    # ax2 = plt.subplot(2,5,2)
    # ax3 = plt.subplot(2,5,3)
    # ax4 = plt.subplot(2,5,4)
    # ax5 = plt.subplot(2,5,5)
    # ax6 = plt.subplot(2,5,6)
    # ax7 = plt.subplot(2,5,7)
    # ax8 = plt.subplot(2,5,8)
    # ax9 = plt.subplot(2,5,9)
    # ax10 = plt.subplot(2,5,10)
    # plt.sca(ax1)
    # plt.imshow(result[0].reshape((8,16)))
    # plt.sca(ax2)
    # plt.imshow(result[1].reshape((8,16)))
    # plt.sca(ax3)
    # plt.imshow(result[2].reshape((8,16)))
    # plt.sca(ax4)
    # plt.imshow(result[3].reshape((8,16)))
    # plt.sca(ax5)
    # plt.imshow(result[4].reshape((8,16)))
    # plt.sca(ax6)
    # plt.imshow(result[5].reshape((8,16)))
    # plt.sca(ax7)
    # plt.imshow(result[6].reshape((8,16)))
    # plt.sca(ax8)
    # plt.imshow(result[7].reshape((8,16)))
    # plt.sca(ax9)
    # plt.imshow(result[8].reshape((8,16)))
    # plt.sca(ax10)
    # plt.imshow(result[9].reshape((8,16)))
    # plt.show()
    
    # # 10*784 10张8*16图片可视化w2*w1
    # w3 = np.dot(w2,w1)
    # wlist3=[]
    # for i in range(10):
    #     wlist3.append(w3[i,:].reshape((1,784)))
    # result=np.array(wlist3)
    # plt.figure(1)
    # ax1 = plt.subplot(2,5,1)
    # ax2 = plt.subplot(2,5,2)
    # ax3 = plt.subplot(2,5,3)
    # ax4 = plt.subplot(2,5,4)
    # ax5 = plt.subplot(2,5,5)
    # ax6 = plt.subplot(2,5,6)
    # ax7 = plt.subplot(2,5,7)
    # ax8 = plt.subplot(2,5,8)
    # ax9 = plt.subplot(2,5,9)
    # ax10 = plt.subplot(2,5,10)
    # plt.sca(ax1)
    # plt.imshow(result[0].reshape((28,28)))
    # plt.sca(ax2)
    # plt.imshow(result[1].reshape((28,28)))
    # plt.sca(ax3)
    # plt.imshow(result[2].reshape((28,28)))
    # plt.sca(ax4)
    # plt.imshow(result[3].reshape((28,28)))
    # plt.sca(ax5)
    # plt.imshow(result[4].reshape((28,28)))
    # plt.sca(ax6)
    # plt.imshow(result[5].reshape((28,28)))
    # plt.sca(ax7)
    # plt.imshow(result[6].reshape((28,28)))
    # plt.sca(ax8)
    # plt.imshow(result[7].reshape((28,28)))
    # plt.sca(ax9)
    # plt.imshow(result[8].reshape((28,28)))
    # plt.sca(ax10)
    # plt.imshow(result[9].reshape((28,28)))
    # plt.show()