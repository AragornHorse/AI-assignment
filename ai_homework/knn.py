import numpy as np
import matplotlib.pyplot as plt

class2num={
    "Iris-setosa":0,
    "Iris-versicolor":1,
    "Iris-virginica":2
}

num2class={   # 方便以后加功能
    0:"Iris-setosa",
    1:"Iris-versicolor",
    2:"Iris-virginica"
}

def l2(n1:np.array,n2:np.array)->np.array:
    """
    :param n1:
    :param n2:
    :return: distance between n1,n2
    """
    return np.sum((n1-n2)**2,1)

def read_data(path=r"C:\Users\Administrator\Desktop\Iris.csv")->[np.array,np.array]:
    """
    :param path:
    :return: x,y
    """
    xs=[]
    ys=[]
    with open(path,'r') as f:
        for i in f:
            lst=i.split(",")
            x=lst[1:5]
            x=[float(j) for j in x]
            label=lst[-1].split("\n")[0]
            x=np.array(x)
            y=np.array(class2num[label])
            xs.append(x)
            ys.append(y)
    return np.array(xs),np.array(ys)


def shuff_data(xs,ys,seed=1)->[np.array,np.array]:
    """
    :param xs:
    :param ys:
    :param seed:
    :return: x,y
    """
    data=[[x,ys[i]] for i,x in enumerate(xs)]
    np.random.seed(seed)
    np.random.shuffle(data)
    xs,ys=[],[]
    for i in data:
        xs.append(i[0])
        ys.append(i[1])
    """
    用相同的seed打乱应该就行，没敢用
    """
    return np.array(xs),np.array(ys)

class dataset():
    def __init__(self,mode='train'):
        self.xs,self.ys=read_data()  # read from csv
        self.xs,self.ys=shuff_data(self.xs,self.ys)   # shuffle

        self.xs=self.xs[:100] if mode == 'train' else self.xs[100:]  # 训练集取前100 ， 测试集选后50
        self.ys = self.ys[:100] if mode == 'train' else self.ys[100:]

    def __getitem__(self, item):
        idx=item
        # idx=item % len(self.ys)
        return self.xs[idx],self.ys[idx]

    def __len__(self):
        return len(self.ys)

train_data=dataset()
test_data=dataset(mode='test')
train_x,train_y=train_data.xs,train_data.ys
test_x,test_y=test_data.xs,test_data.ys

def select_k(rg=60)->np.array:
    k_score=np.zeros([rg])
    for x,y in test_data:
        lst=np.argsort(l2(x,train_x))  # 按照距离排序
        for k in range(rg):
            rst_idx=lst[:k+1]   # 选最近的k个点
            rst=train_y[rst_idx]  # 最近的点的标签
            d0=np.sum((rst==0)+0)
            d1 = np.sum((rst == 1) + 0)
            d2 = np.sum((rst == 2) + 0)
            ds=np.array([d0,d1,d2])   # k个点中各标签的个数
            y_hat=np.argmax(ds)     # 数量最多的标签
            if y_hat==y:
                k_score[k]+=1    # 判断正确，个数+1
    return k_score / len(test_y)      # 准确率

def work(k:int)->list:
    r=[]
    for x,y in test_data:
        rst_idx=np.argsort(l2(x,train_x))[:k+1]
        rst=train_y[rst_idx]
        d0=np.sum((rst==0)+0)
        d1 = np.sum((rst == 1) + 0)
        d2 = np.sum((rst == 2) + 0)
        ds=np.array([d0,d1,d2])
        y_hat=np.argmax(ds)
        r.append([y_hat,y])
    return r

k_scores=select_k()
k,idx=np.max(k_scores),np.argmax(k_scores)



print("从1-60不同k下的正确率:")
print(k_scores)
print("最高正确率:",k)
print("最佳k:",idx+1)
print("在此k下的所有预测结果和真实标签:")
print(work(int(idx)))

plt.ylabel("accuracy")
plt.xlabel("k-value")
plt.plot(np.linspace(0,60,60),k_scores)
plt.show()








