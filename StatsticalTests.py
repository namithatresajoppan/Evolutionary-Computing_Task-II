#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import pickle
from scipy.stats import ttest_ind


# In[2]:


EA = ['NEAT','CMA']
Group = ['1258', '1234']

#welch t test comparing 2 algorithms
for k in Group:
        box1 = pd.read_csv('boxplotdataNEAT'+ k + '.csv')
        box2 = pd.read_csv('boxplotdataCMA'+ k + '.csv')
        print()
        print('For Group:', k)
        print()
        for i in range(1,9):
            print('Enemy:', i)
            print((ttest_ind(box1['{}'.format(i)], box2['{}'.format(i)], equal_var = False)))


# In[3]:


EA = ['NEAT','CMA']
Group = ['1258', '1234']

#welch t test comparing 2 enemy groups
for ea in EA:
        box1 = pd.read_csv('boxplotdata{}'.format(ea)+'1258.csv')
        box2 = pd.read_csv('boxplotdata{}'.format(ea)+'1234.csv')
        print()
        print('For EA:', ea)
        print()
        for i in range(1,9):
            print('Enemy:', i)
            print((ttest_ind(box1['{}'.format(i)], box2['{}'.format(i)], equal_var = False)))


# In[27]:


#mean values
EA = ['NEAT','CMA']
Group = ['1258', '1234']
for ea in EA:
    for k in Group:
        folder = ea + k
        box = pd.read_csv('boxplotdata'+ folder + '.csv')
        temp = box.mean(axis=1)
        print(folder+ '={}'.format(np.mean(temp)))

