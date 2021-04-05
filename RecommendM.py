import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle
import time
from multiprocessing import Pool
import os

def splitTrainSetTestSet(odatas,frac):
    testset = odatas.sample(frac=frac, axis=0)
    trainset = odatas.drop(index=testset.index.values.tolist(), axis=0)
    return trainset,testset

def readDatas():
    path = 'rating.csv'
    odatas = pd.read_csv(path,usecols=[0,1,2])
    return odatas

class LFM():
    def __init__(self,dataset,factors, epochs, lr, lamda):
        self.dataset = dataset

        self.userList, self.itemList = self.__getListMap()
        self.factors=factors
        self.epochs=epochs
        self.lr=lr
        self.lamda=lamda

        self.p = pd.DataFrame(np.random.randn(len(self.userList), factors), index=self.userList)
        self.q = pd.DataFrame(np.random.randn(len(self.itemList), factors), index=self.itemList)
        self.bu = pd.DataFrame(np.random.randn(len(self.userList)), index=self.userList)
        self.bi = pd.DataFrame(np.random.randn(len(self.itemList)), index=self.itemList)

    def __prediction(self,pu, qi, bu, bi):
        return (np.dot(pu, qi.T) + bu + bi)[0]

    def __getError(self,r, pu, qi, bu, bi):
        return r - self.__prediction(pu, qi, bu, bi)

    def Divide(self,p1,p2,q1,q2):
        data1=[]
        data2=[]
        data3=[]
        data4=[]
        for d in self.dataset.values:
            u=d[0]
            i=d[1]
            r=d[2]
            if u in p1:
                if i in q1:
                    data1.append([u,i,r])
                if i in q2:
                    data3.append([u,i,r])
            if u in p2:
                if i in q1:
                    data2.append([u,i,r])
                if i in q2:
                    data4.append([u,i,r])
        return data1,data2,data3,data4

    def __getListMap(self):
        userSet, itemSet = set(), set()
        for d in self.dataset.values:
            userSet.add(int(d[0]))
            itemSet.add(int(d[1]))
        userList = list(userSet)
        itemList = list(itemSet)
        return userList, itemList

    def fit(self,testset):
        for e in tqdm(range(self.epochs)):
            for d in tqdm(self.dataset.values):
                u, i, r = d[0], d[1], d[2]
                error = self.__getError(r, self.p.loc[u], self.q.loc[i], self.bu.loc[u], self.bi.loc[i])
                self.p.loc[u] += self.lr * (error * self.q.loc[i] - self.lamda * self.p.loc[u])
                self.q.loc[i] += self.lr * (error * self.p.loc[u] - self.lamda * self.q.loc[i])
                self.bu.loc[u] += self.lr * (error - self.lamda * self.bu.loc[u])
                self.bi.loc[i] += self.lr * (error - self.lamda * self.bi.loc[i])
            self.testRMSE(testset)
    def child(self,data,p,q,bu,bi,N):
        start = time.time()
        print(str(N) + " " + str(os.getpid()))
        #print(str(N)+" "+str(os.getpid())+" start in " + start)
        halfu = int(len(self.userList)/2)
        halfi = int(len(self.itemList)/2)
        rp = pd.DataFrame(np.zeros((len(self.userList), self.factors)), index=self.userList)
        rq = pd.DataFrame(np.zeros((len(self.itemList), self.factors)), index=self.itemList)
        rbu = pd.DataFrame(np.zeros(len(self.userList)), index=self.userList)
        rbi = pd.DataFrame(np.zeros(len(self.itemList)), index=self.itemList)
        for d in data:
            u,i,r = d[0], d[1], d[2]
            error = self.__getError(r, p.loc[u], q.loc[i], bu.loc[u], bi.loc[i])
            p.loc[u] += self.lr * (error * q.loc[i] - self.lamda * p.loc[u])
            q.loc[i] += self.lr * (error * p.loc[u] - self.lamda * q.loc[i])
            bu.loc[u] += self.lr * (error - self.lamda * bu.loc[u])
            bi.loc[i] += self.lr * (error - self.lamda * bi.loc[i])
            if N == 1:
                rp[:halfu]=p
                rq[:halfi]=q
                rbu[:halfu]=bu
                rbi[:halfi]=bi
            if N == 2:
                rp[halfu:]=p
                rq[:halfi]=q
                rbu[halfu:]=bu
                rbi[:halfi]=bi
            if N == 3:
                rp[:halfu]=p
                rq[halfi:]=q
                rbu[:halfu]=bu
                rbi[halfi:]=bi
            if N == 4:
                rp[halfu:]=p
                rq[halfi:]=q
                rbu[halfu:]=bu
                rbi[halfi:]=bi
                
        return rp,rq,rbu,rbi
        end = time.time()
        print(end-start)
        
    
    def mfit(self,testset):
        halfu = int(len(self.userList)/2)
        p1 = pd.DataFrame(np.zeros((halfu, self.factors)), index=self.userList[:halfu])
        p2 = pd.DataFrame(np.zeros((len(self.userList)-halfu, self.factors)), index=self.userList[halfu:])
        p1=self.p[:halfu]
        p2=self.p[halfu:]
        bu1 = pd.DataFrame(np.zeros(halfu), index=self.userList[:halfu])
        bu2 = pd.DataFrame(np.zeros(len(self.userList)-halfu), index=self.userList[halfu:])
        bu1=self.bu[:halfu]
        bu2=self.bu[halfu:]
        
        halfi = int(len(self.itemList)/2)
        q1=pd.DataFrame(np.zeros((len(self.itemList[:halfi]), self.factors)), index=self.itemList[:halfi])
        q2=pd.DataFrame(np.zeros((len(self.itemList)-len(self.itemList[:halfi]), self.factors)), index=self.itemList[halfi:])
        q1=self.q[:halfi]
        q2=self.q[halfi:]
        bi1 = pd.DataFrame(np.zeros(len(self.itemList[:halfi])), index=self.itemList[:halfi])
        bi2 = pd.DataFrame(np.zeros(len(self.itemList)-len(self.itemList[:halfi])), index=self.itemList[halfi:])
        bi1=self.bi[:halfi]
        bi2=self.bi[halfi:]
        data1,data2,data3,data4=self.Divide(p1.index,p2.index,q1.index,q2.index)#划分数据集
        results=[]
        for e in tqdm(range(self.epochs)):#训练正式开始
            p1=self.p[:halfu]
            p2=self.p[halfu:]
            bu1=self.bu[:halfu]
            bu2=self.bu[halfu:]
            q1=self.q[:halfi]
            q2=self.q[halfi:]
            bi1=self.bi[:halfi]
            bi2=self.bi[halfi:]
            
            pool=Pool(2)#建立子进程池
            for a in range(2):
                if a == 0 :
                    sub1=pool.apply_async(self.child,args=(data1,p1,q1,bu1,bi1,1))
                    #rp1,rq1,rbu1,rbi1=self.child(data1,p1,q1,bu1,bi1,1)
                if a == 1 :
                    sub4=pool.apply_async(self.child,args=(data4,p2,q2,bu2,bi2,4))

                    #rp2,rq2,rbu2,rbi2=self.child(data4,p2,q2,bu2,bi2,4)

            rp1,rq1,rbu1,rbi1=sub1.get()
            rp2,rq2,rbu2,rbi2=sub4.get()
            
            self.p=rp1+rp2#在14两块后更新参数
            self.q=rq1+rq2
            self.bu=rbu1+rbu2
            self.bi=rbi1+rbi2
            
            p1=self.p[:halfu]
            p2=self.p[halfu:]
            bu1=self.bu[:halfu]
            bu2=self.bu[halfu:]
            q1=self.q[:halfi]
            q2=self.q[halfi:]
            bi1=self.bi[:halfi]
            bi2=self.bi[halfi:]
            for b in range(2):
                if b == 0:
                    sub2=pool.apply_async(self.child,args=(data2,p2,q1,bu2,bi1,2))
                if b == 1:
                    sub3=pool.apply_async(self.child,args=(data3,p1,q2,bu1,bi2,3))

            rp2,rq1,rbu2,rbi1=sub2.get()
            rp1,rq2,rbu1,rbi2=sub3.get()
            
            self.p=rp1+rp2#在23两块后更新参数
            self.q=rq1+rq2
            self.bu=rbu1+rbu2
            self.bi=rbi1+rbi2
            self.testRMSE(testset)
            
            pool.close()
            pool.join()
            
                

    def __RMSE(self,a, b):
        return(np.average((np.array(a) - np.array(b)) ** 2)) ** 0.5

    def testRMSE(self,testSet):
        y_true, y_hat = [], []
        for d in testSet.values:
            user = int(d[0])
            item = int(d[1])
            if user in self.userList and item in self.itemList:
                hat=self.__prediction(self.p.loc[user], self.q.loc[item], self.bu.loc[user], self.bi.loc[item])
                y_hat.append(hat)
                y_true.append(d[2])
        rmse = self.__RMSE(y_true,y_hat)
        with open("rmse.txt","a") as f:
            f.write(str(rmse))
            f.write("\n")
            f.close
        return rmse

    def save(self,path):
        with open(path,'wb+') as f:
            pickle.dump(self,f)

    @staticmethod
    def load(path):
        with open(path,'rb') as f:
            return pickle.load(f)

def play():
    start = time.time()
    factors=3 #隐因子数量
    epochs=20 #迭代次数
    lr=0.01 #学习率
    lamda=0.1 #正则项系数

    model_path='model/lfm.model'

    trainset, testSet = splitTrainSetTestSet(readDatas(),0.2)

    #lfm=LFM.load(model_path)
    lfm=LFM(trainset,factors, epochs, lr, lamda)
    #lfm.load(model_path)

    #lfm.fit(testSet)
    lfm.mfit(testSet)
    #rmse_test = lfm.testRMSE(testSet)
    #rmse_train = lfm.testRMSE(trainset)
    end = time.time()
    #print(rmse_train)
    #print(rmse_test)
    print(end-start)
    
if __name__ == '__main__':
    play()
