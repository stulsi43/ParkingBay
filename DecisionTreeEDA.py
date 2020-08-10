# -*- coding: utf-8 -*-
"""
Created on Sun Aug  9 12:17:31 2020

@author: Sohail Tulsi
Decision Tree analysis
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('DecisionTreePrediction.csv')


print(df.columns)

df_cat = df[['index', 'Age', 'Gender', 'Finance_Degree', 'Science_Degree',
       'WorkExperience', 'Distance',
       'predicted_transport', 'Salary']]

for i in df_cat.columns:
    cat_num = df_cat[i].value_counts()
    print("graph for %s: total = %d" % (i, len(cat_num)))
    chart = sns.barplot(x=cat_num.index, y=cat_num)
    chart.set_xticklabels(chart.get_xticklabels(),rotation = 90)
    plt.show()
    
pd.pivot_table(df,index = 'predicted_transport',values = 'Salary')    


for i in df_cat.columns:
    print(i)
    print(pd.pivot_table(df_cat, index = i, values = 'Salary').sort_values('Salary',ascending = False))
    
    
    