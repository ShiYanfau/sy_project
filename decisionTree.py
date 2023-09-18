import math

import pandas as pd
import numpy as np
from math import log
from collections import Counter
from sklearn.model_selection import train_test_split

#model
class decisionTree(object):

    def __init__(self,max_depth,min_sample,impurity_type,features,label,min_gainratio):  # features 是特征的名字，是一个数组，里面全是char，例如'outlook' 'wind'……

        self.max_depth = max_depth
        self.min_sample = min_sample
        self.impurity_type = impurity_type
        self.festures = features            #  特征的名字
        self.label = label  # 目标分类，比如'play'，'win'这种
        self.min_gainratio = min_gainratio

        self.tree = None

    '''
    # Calculate information entropy
    #计算信息熵
'''
    def impurtiy(self,data):
        df = data.copy()  # 感觉这不用copy
        n = Counter(df[self.label])       #统计每个元素出现的次数，比如0和1，就统计0出现的次数和1出现的次数,注：n是字典
        px = [1.0 * n[i] / len(data[self.label]) for i in n]  # 归一化处理
        if self.impurity_type == 'entropy':
            entropy = -np.sum([p * math.log2(p) for p in px])
            return entropy

        gini = 1 - np.sum([p * p for p in px])
        #  print(gini)
        #  print(entropy)
        return gini

    '''
    # Calculate gain ratio
    # 计算增益比
'''
    def gainratio(self,data,feature): #输入data，以及选取的特征名称，输出信息增益比

        df = data.copy()

        entropy_before = self.impurtiy(df)
        number = df.shape[0]

        entropy_after = 0
        splitinformation = 0

        subdata,fvalue = self.df_spilt(df,feature)

        for value in fvalue:

            sub_number = subdata[value].shape[0]
            entropy_after -= 1.0 * self.impurtiy(subdata[value]) * sub_number / number # 子集数据数量/全集数据数量
            splitinformation -= sub_number/ number * math.log2(1.0 * sub_number / number)

        gainratio = 1.0 * (entropy_before - entropy_after) / (splitinformation + 0.05 )  # 避免spilt为0

        return gainratio

    '''
    # Divide data subsets based on feature values
    '''
    def df_spilt(self,data, feature):  # 根据特征取值划分数据子集，feature就是一个char，例如'outlook',返回的是字典，一个取值对应的一份数据

        df = data.copy()
        fvalue = df[feature].unique()
        # print(n)
        res_dict = {value: pd.DataFrame() for value in fvalue}  # 为每个value建立空集
        # print(res_dict)
        # print(res_dict.keys())
        for key in res_dict.keys():
            res_dict[key] = df[:][df[feature] == key]
        # print(res_dict)
        return res_dict, fvalue

    '''
    # Train model
    '''
    def fit(self,data):  # 训练模型

        #训练用的特征数和定义模型的特征数相同
        assert len(data.columns[1:]) == len(self.festures)
        avaiable_features = self.festures.copy()  # 这里定义可用的feature
        self.tree = self.expand_node(data,1,avaiable_features)  # 这里建成一棵完整的树
        print('训练完成')


        # 这里定义avaiable_features


    # Extended node
    #扩展结点
    def expand_node(self, data, depth, avaiable_features_oringal):  # 返回元组，就能完成递归  希望返回的是（feature, {value: },[x,y] )
        '''
        # Four situations
        # Case 1: The node is very pure and does not need to be split.
        # Case 2: The maximum depth is reached, splitting is not allowed, and there are too few split sample points, so it will not be split, and the label with the most occurrences will be returned.
        # Situation 3: The information gain ratio is too small, and it makes no sense to continue splitting.
        # Case 4: Find new features and continue splitting
'''
        # 四种情况
        # 情况1 ： 节点很纯，无需分裂
        # 情况2： 达到最大深度，不允许分裂,分裂样本点太少了，也不分裂了，返回出现最多的标签
        # 情况3： 信息增益比太小，继续分裂没有意义
        # 情况4： 寻找新的特征，继续分裂

        df = data.copy()

        n = Counter(df[self.label])  # 标签种类有几种,感觉Counter能计算的信息比unique要多
        arr = np.array([n.get(0,0), n.get(1,0)])  #建立一个数组, [50 ,20]表示有50个0和20个1

        # 情况1
        if len(n) <= 1:
            return arr  # 返回数组
        if depth >= self.max_depth or len(df[self.label]) <= self.min_sample:  # 是最大深度,3是最小叶子结点，小于5也不再分裂了
            return arr

        # 定义一些要返回的值
        feature_best = None
        max_gainratio = 0
        child_node = {}
        avaiable_features = avaiable_features_oringal.copy()

        # 找到信息增益比最大的特征
        for feature in avaiable_features :      # 这里应该换成 avaiable features
            gainratio = self.gainratio(df,feature)  #计算每个特征分裂的信息增益
            if max_gainratio < gainratio:
                max_gainratio = gainratio
                feature_best = feature

        if max_gainratio <= self.min_gainratio:   # 信息增益比太小，不再分裂
            return arr    #

        # 信息增益比足够大，继续分裂
        avaiable_features.remove(feature_best)

        subdata, fvalue = self.df_spilt(data,feature_best)
        for value in fvalue:
            child_node[value] = self.expand_node(subdata[value],depth+1,avaiable_features)

        return (feature_best, child_node, arr)

    '''
    #预测
    # predict
    '''
    def predcit(self,data): # 这里可能有很多的data，并不是一个data

        assert len(data.shape) == 1 or len(data.shape) == 2 # 确保数据只能是一维或者二维
        df = data.copy()
        assert len(df.columns[1:]) == len(self.festures)
        result = np.full(len(df[self.label]), -1) # 创建一个全是-1的数组，用来保留结果
        for i in range(df.shape[0]):
            result[i] = self.traverse_node(self.tree,df.iloc[[i]])

        return result

    '''
    #遍历模型
    '''
    # Traverse the model
    def traverse_node(self,node, data):

        #
        #情况1 是叶结点，返回数组最大值的下标
        #情况2 不是叶结点，继续遍历树
        #     注： 当父节点应有三个特征值，但分裂子结点时只用到两个，在预测数据特征值为未使用数据时，返回父节点数组最大值下标

        df = data.copy()
        len1 = len(node)
        value = df.at[df.index[0], node[0]] # 这个地方不对

        if node[1].get(value) is None:
            n = np.argmax(node[2])
            return np.argmax(node[2])
        childnode = node[1].get(value)
        if isinstance(childnode, np.ndarray):
            m = np.argmax(childnode)
            return np.argmax(childnode)

        result = self.traverse_node(childnode,df)

        return result

    def gettree(self):
        return self.tree


# 树写完了

#Data preprocessing stage
#数据预处理阶段

data_df = pd.read_csv('high_diamond_ranked_10min.csv')  # 读取文件
data_df = data_df.drop(columns='gameId') # 舍去对局标号列
# print(data_df.iloc[0]) # 输出第一行数据
data_df.describe() # 每列特征的简单统计信息

'''
#Remove unnecessary features
'''
#舍去不需要的特征
drop_features = ['blueGoldDiff', 'redGoldDiff',
                 'blueExperienceDiff', 'redExperienceDiff',
                 'blueCSPerMin', 'redCSPerMin',
                 'blueGoldPerMin', 'redGoldPerMin']          # 需要舍去的特征列
df = data_df.drop(columns=drop_features) # 舍去特征列
info_names = [c[3:] for c in df.columns if c.startswith('red')] # 取出要作差值的特征名字（除去red前缀）
for info in info_names: # 对于每个特征名字
    df['br' + info] = df['blue' + info] - df['red' + info] # 构造一个新的特征，由蓝色特征减去红色特征，前缀为br
# 其中FirstBlood为首次击杀最多有一只队伍能获得，brFirstBlood=1为蓝，0为没有产生，-1为红
df = df.drop(columns=['blueFirstBlood', 'redFirstBlood']) # 原有的FirstBlood可删除

'''
#  data processing, for example , a certain favlue has random values from 100-2000, grade these data
'''
# 某个favlue有100-2000的随机取值，给这些数据分等级
DISCRETE_N = 10  # 取10就意味着变成10档
discrete_df = df.copy() # 先复制一份数据
for c in df.columns[1:]: # 遍历每一列特征，跳过标签列
    if len(df[c].unique()) <= DISCRETE_N: # 对原本取值可能性就较少的列，比如大龙数目，可不作处理
        continue
    else:
        discrete_df[c] = pd.qcut(df[c], DISCRETE_N, precision=0, labels=False, duplicates='drop')
        # precsion=0表示区间间隔数字的精度保持和数据原本精度相同，duplicates='drop'表示去除某些重复的分隔数（即空区间）
        # 调用pandas的qcut函数，表示等密度区间，尽量让每个区间里的样本数相当，但每个区间长度可能不同
        # 调用pandas的cut函数，表示等长的区间，尽量让每个区间的长度相当，但每个区间里的样本数可能不同
        # discrete_df[c] = pd.cut(df[c], 10, precision=0, duplicates='drop')


'''
# Move the label column to the  column 0
'''
# 把标签列弄到第一列
target = 'blueWins'

#print(df)
colums = discrete_df.columns.tolist()   # 把label放到第0列
for i in range(len(colums)):
    if colums[i] == target:
        colums[i], colums[0] = colums[0],colums[i]
discrete_df = discrete_df[colums]
#print(df)
features_name = list(discrete_df.columns[1:])  # 从第1列到最后一列

'''
# Divide the data set
'''
# 划分数据集
RANDOM_SEED = 2020  # 固定随机种子
train_df, test_df = train_test_split(discrete_df, test_size=0.2, random_state=RANDOM_SEED) #

'''
# Create a tree
'''
# 建立完成一棵树
DT = decisionTree(max_depth=15,min_sample=8,impurity_type='entropy',features=features_name,label=target,min_gainratio=0.05)
DT.fit(train_df)
tree = DT.gettree()

# sample = test_df.iloc[[0]]
result = DT.predcit(test_df)
real_result = test_df.iloc[:,0].values
accuracy = np.sum(real_result==result)
accuracy = 1.0 * accuracy / len(result)
print(accuracy)





















