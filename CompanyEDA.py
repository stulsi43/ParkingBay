# -*- coding: utf-8 -*-
"""
Created on Sun Aug  9 12:47:58 2020

@author: Sohail Tulsi
Decision Tree analysis
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder

df = pd.read_csv('Company_Y_Employees.csv')
df = df.dropna()

print(df.columns)

# One hot encoding gender
ohe = OneHotEncoder(categories='auto')
Xd = ohe.fit_transform(df.Transport.values.reshape(-1, 1)).toarray()
df_ohe = pd.DataFrame(Xd, columns = ["Transport"+str(int(i)) for i in range(Xd.shape[1])])

# Add gender encoded feature to the dataframe
df_to_use1 = pd.concat([df, df_ohe], axis=1)
df_to_use1 = df_to_use1.dropna()


df_to_use1 = df_to_use1.rename(columns={"Transport0": "Car", "Transport1": "Motorbike", "Transport2": "Public Transport"})

df_cat = df_to_use1[[ 'Age', 'Gender', 'Finance_Degree', 'Science_Degree',
       'WorkExperience', 'Distance',
       'Car','Motorbike','Public Transport', 'Salary']]

for i in df_cat.columns:
    cat_num = df_cat[i].value_counts()
    print("graph for %s: total = %d" % (i, len(cat_num)))
    chart = sns.barplot(x=cat_num.index, y=cat_num)
    chart.set_xticklabels(chart.get_xticklabels(),rotation = 90)
    plt.show()
    
pd.pivot_table(df_to_use1,index = 'Transport',values = 'Salary')    

cmap = sns.diverging_palette(220,10, as_cmap=True)
sns.heatmap(df_cat[['Age','Salary','Car','Motorbike','Public Transport','Distance','WorkExperience']].corr(), center=0, cmap=cmap,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})


for i in df_cat.columns:
    print(i)
    print(pd.pivot_table(df_to_use1, index = i, values = 'Salary').sort_values('Salary',ascending = False))