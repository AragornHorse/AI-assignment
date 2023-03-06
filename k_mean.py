import numpy as np
import matplotlib.pyplot as plt

class2num={
    "Iris-setosa":0,
    "Iris-versicolor":1,
    "Iris-virginica":2
}

num2class={
    0:"Iris-setosa",
    1:"Iris-versicolor",
    2:"Iris-virginica"
}

def l2(n1,n2):
    """
    :param n1:
    :param n2:
    :return: distance between n1,n2
    """
    return np.sum((n1-n2)**2)

def read_data(path=r"C:\Users\Administrator\Desktop\Iris.csv"):
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


def shuff_data(xs,ys,seed=1):
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
    return np.array(xs),np.array(ys)


class datas():
    def __init__(self):
        self.xs,self.ys=read_data()
        self.xs,self.ys=shuff_data(self.xs,self.ys)  # x,y
        self.class_num=3   # 种类是3
        self.center=[np.random.random([4]),np.random.random([4]),np.random.random([4])]  # 随机给三个中心
        self.classes=[[],[],[]]   # 分配好的类

    def spilt_step(self,x):
        d0=l2(x,self.center[0])
        d1=l2(x,self.center[1])
        d2=l2(x,self.center[2])
        # # #  distance between x and 3 centers

        if d0<=d1 and d0<=d2:
            self.classes[0].append(x)
        elif d1<d0 and d1<d2:
            self.classes[1].append(x)
        elif d2<d0 and d2<d1:
            self.classes[2].append(x)
        # # 离谁近给哪类

    def split_all(self):
        """  xs 全部分类 """
        for x in self.xs:
            self.spilt_step(x)

    def optim_centers(self):

        class0=np.array(self.classes[0])
        class1=np.array(self.classes[1])
        class2=np.array(self.classes[2])

        c0=np.mean(class0,0) if len(class0>0) else np.array([0,0,0,0])
        c1 = np.mean(class1, 0) if len(class1>0) else np.array([0,0,0,0])
        c2 = np.mean(class2, 0) if len(class2>0) else np.array([0,0,0,0])
        # # 新的center是当前所有类的均值，如果类是空，就设为0

        self.center=[c0,c1,c2]

    def iter_k_mean(self,iter_num=8):
        for i in range(iter_num):
            self.classes=[[],[],[]]
            self.split_all()
            self.optim_centers()

    def __getitem__(self,idx):
        x=self.xs[idx]
        d0 = l2(x, self.center[0])
        d1 = l2(x, self.center[1])
        d2 = l2(x, self.center[2])
        if d0<=d1 and d0<=d2:
            self.classes[0].append(x)
            pred=num2class[0]
        elif d1<d0 and d1<d2:
            self.classes[1].append(x)
            pred=num2class[1]
        elif d2<d0 and d2<d1:
            self.classes[2].append(x)
            pred=num2class[2]
        return [x,num2class[self.ys[idx]],pred]

def correct_rate(c,prt=False):
    num = 0
    cor = 0
    for data in c:
        pred = data[-2]
        label = data[-1]
        if pred=="Iris-setosa":
            pred="Iris-versicolor"
        elif pred=="Iris-virginica":
            pred="Iris-setosa"
        else:
            pred="Iris-virginica"
        num += 1
        if prt:
            print(pred, label)
        if pred == label:
            cor += 1
    return cor / num

c=datas()
c.split_all()
scores=[]

for i in range(100):
    c.iter_k_mean(1)
    score=correct_rate(c)
    scores.append(score)

c.iter_k_mean(1)
print(correct_rate(c,True))
print(np.array(c.center))
print(scores)

plt.ylabel("accuracy")
plt.xlabel("iter-times")
plt.plot(np.linspace(0,100,100),np.array(scores))
plt.show()












