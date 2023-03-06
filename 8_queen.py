import numpy as np
from itertools import *
import time

def perutations(lst,num):
    """
    :param lst:
    :param num:
    :return: 0-8 的全排列
    """
    lsts=[]
    def sort_(lst1=[],lst2=lst):
        nonlocal lsts
        if len(lst2)==2:
            lsts.append(lst1+lst2)
            lsts.append(lst1+[lst2[1],lst2[0]])
        else:
            for idx,i in enumerate(lst2):
                sort_(lst1+[i],lst2[:idx]+lst2[idx+1:])
    sort_()
    return lsts

def init_map_idx(num=8):
    """
    :param num: 8
    :return: 0-8 的全排列
    """
    lst=list(range(0,num))
    a=list(permutations(lst,num))
    b=[list(i) for i in a]
    return np.array(b)

def init_map(idxs,num=8):
    """
    :param idxs: 0-8 的 全排列
    :param num: 8
    :return: 横竖不重合的所有可能
                0:none , 1: queen
    """
    mps=np.zeros([len(idxs),num,num])
    # print(mps.shape)
    for i,idx in enumerate(idxs):
        # print(i,idx)
        for k,id in enumerate(idx):
            mps[i,k,id]=1
    # print(mps)
    return mps

def generate_cross(site,num=8):
    """
    通过一个点的坐标生成矩阵，交叉经过此点的两条线上均为1，其余为0
    :param site:
    :return:
    """
    mp=np.zeros([num,num])
    idxs=[[i,j]
          for i in range(0,num)
          for j in range(0,num)
          if (i-site[0])**2==(j-site[1])**2
          ]
    for idx in idxs:
        mp[idx[0],idx[1]]=1
    mp[site[0],site[1]]=0
    return mp

class maps():
    def __init__(self):
        self.mps_idxs=init_map_idx(8)   # all the init_queen sites
        self.num=0    # final number of the answers
        self.mps=init_map(self.mps_idxs)   # 横竖不重合的所有可能
        self.answers_idx=[]     # the index of the answer in mps
        self.answers=[]       # the answer maps
        self.iter_mps()       # find out the answers
        self.generate_answers()  # find out the answer_maps

    def iter_mps(self):
        """
        find out answers
        :return:
        """
        for i,idx in enumerate(self.mps_idxs):
            mp = np.copy(self.mps[i])
            for j,k in enumerate(idx):
                cross_map=generate_cross([j,k])
                mp-=cross_map
            obj=np.sum((mp==1)+0)  # 元素1的存在说明这个皇后一直没被-过，所有1的个数没变的话就说明所有皇后位置合理
            if obj == 8:
                self.answers_idx.append(i)  # means i is a corrent answer
                self.num+=1

    def generate_answers(self):
        """
        :return:copy correct answers
        """
        for i in self.answers_idx:
            self.answers.append(self.mps[i])

time1=time.time()
mps=maps()

idxs=mps.answers_idx
time2=time.time()
print(mps.num)
print(mps.answers)
# print(mps.answers_idx)
print(time2-time1)









