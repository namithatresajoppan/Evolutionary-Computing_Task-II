#!/usr/bin/env python
# coding: utf-8

# In[112]:


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import pickle


# # Lineplots

# In[233]:


#intializing for folder access
EA = ['NEAT','CMA']
Group = ['1258', '1234']

exp = np.linspace(1,10,10,dtype = int)
data = pd.DataFrame()
best = pd.DataFrame()
for k in (Group):
    for ea in EA:
        folder = ea + k
        final = {'avg': [], 'best': []}
        data1 = pd.DataFrame()
        best1 = pd.DataFrame()
        for i in exp:
            if(ea == 'CMA'):
                test = pd.read_csv((folder+ '/exp{}_output.csv'.format(i)),names=['index','generation','avg','std','min','best'], skiprows = 1)
            else:
                test = pd.read_csv((folder + '/exp{}_output.csv'.format(i)),names=['best','avg','std'])
            data1['Exp{}'.format(i)] = test['avg']
            best1['best_Exp{}'.format(i)] = test['best']
        data['{}'.format(folder)+'_Average'] = data1.mean(axis =1)
        data['{}'.format(folder)+'_Std'] = data1.std(axis = 1)
        best['{}'.format(folder)+'_Maximum'] = best1.mean(axis = 1) 
        best['{}'.format(folder)+'_Std'] = best1.std(axis = 1)


# In[259]:


#sns.set_style("darkgrid", {"axes.facecolor": "0.92"})
sns.set_style("darkgrid")
color = ['red','blue']
i = 0
for k in (Group):
    plt.figure(figsize = (8,5))
    i = 0
    for ea in EA:
        folder = ea + k
        sns.lineplot(data = data, x = data.index, y = folder+'_Average', label = ea+' Mean', color = color[i])
        plt.plot(best.index, best[folder+'_Maximum'], label = ea+' Maximum', color = color[i], ls = '--')
        plt.fill_between(data.index, data[folder+'_Average'] - data[folder+'_Std'], data[folder+'_Average']+data[folder+'_Std'], color = 'grey', alpha = 0.3)
        plt.fill_between(best.index, best[folder+'_Maximum'] - best[folder+'_Std'], best[folder+'_Maximum']+best[folder+'_Std'], color = 'grey', alpha = 0.3)
        i +=1
    plt.ylabel("Fitness Values", fontsize = 15)  
    plt.ylim(-5,100)
    plt.xlabel("Generations", fontsize = 15)
    if(k == '1258'):
        plt.title("CMA and NEAT for Enemy Group: [1,2,5,8]", fontsize = 12)
    else:
         plt.title("CMA and NEAT for Enemy Group: [1,2,3,4]", fontsize = 12)
    plt.legend(loc = 'upper left')
    plt.savefig("lineplot"+k)


# # Boxplots

# In[250]:


EA = ['NEAT','CMA']
Group = ['1258', '1234']
enemies = np.linspace(1,8,8,dtype = 'int')
boxplotdata = pd.DataFrame()
sns.set_style("darkgrid")
for k in (Group):
    for ea in EA:
        folder = ea + k
        indp_gains ={}
        if(ea == 'CMA'):
            for enemy in enemies:
                gain = [] #mean gain
                res = {'en': [], 'gain': []}
                means = []
                for i in exp:
                    file = pd.DataFrame(pd.read_pickle(folder + '/exp{}'.format(i)+'_gains.p'))
                    gain = (np.mean(file.loc[file['en'] == enemy, 'gain'].values))
                    res['en'].append(enemy)
                    res['gain'].append(gain)
                df = pd.DataFrame(res)
                indp_gains['{}'.format(enemy)] = df['gain'] 
        else:
            for enemy in enemies:
                means=[]
                for i in exp:
                    test = pickle.load(open(folder+ '/en{}_'.format(enemy)+'exp{}'.format(i) + '_gains.p',"rb"))
                    df = pd.DataFrame(test)
                    mean_gain = df['gain'].mean(axis=0)
                    means.append(mean_gain)
                indp_gains['{}'.format(enemy)] = means
            
        df = pd.DataFrame(indp_gains)  
        df.to_csv('boxplotdata'+folder+'.csv',index=False)
        plt.figure(figsize = (11,6))
        ax=sns.boxplot(x="variable", y="value",data=pd.melt(df))
        ax.set_ylabel('Individual gain',fontsize=15)
        ax.set_xlabel('Enemy', fontsize = 15)
        if(k == '1258'):
            ax.set_title('Individual Gain for '+ ea+' Enemy Group: [1,2,5,8] ' ,fontsize=15)
        else:
            ax.set_title('Individual Gain for '+ ea+' Enemy Group: [1,2,3,4] ' ,fontsize=15) 
        plt.savefig("boxplot"+folder)


# In[245]:




