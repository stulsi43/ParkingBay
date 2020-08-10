# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 02:01:28 2020

@author: Sohail Tulsi
"""


# Import libraries
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import OneHotEncoder

# Import data
df = pd.read_csv('C:/Users/stuls/OneDrive/Desktop/GrowthAnalyst/Company_Y_Employees.csv')

# Drop missing entries
df1 = df.dropna()
df2 = df[df['Transport'].isnull()]

# One hot encoding gender
ohe = OneHotEncoder(categories='auto')
Xd = ohe.fit_transform(df1.Gender.values.reshape(-1, 1)).toarray()
df_ohe = pd.DataFrame(Xd, columns = ["Gender"+str(int(i)) for i in range(Xd.shape[1])])

# Add gender encoded feature to the dataframe
df_to_use1 = pd.concat([df1, df_ohe], axis=1)
df_to_use1 = df_to_use1.dropna()
print('ye')
# One hot encoding transport
Xtrans = ohe.fit_transform(df1.Transport.values.reshape(-1, 1)).toarray()
print('ye')
df_ohetrans = pd.DataFrame(Xtrans, columns = ["Transport"+str(int(i)) for i in range(Xtrans.shape[1])])
print('ye')
# Add transport encoded feature to the dataframe
df_to_use2 = pd.concat([df_to_use1, df_ohetrans], axis=1)
df_to_use2 = df_to_use2.dropna()
df_to_use2 = df_to_use2.reset_index()

# Split data into features (X) and response (y)
X = df_to_use2.loc[:,["Age","Gender0","Finance_Degree","Science_Degree","WorkExperience","Salary","Distance"]]
y = df_to_use2.loc[:,["Transport"]] 
y = np.ravel(y)

# Split the data into the training set and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)


# Scale the data
scaler = StandardScaler()  

# Remember to fit using only the training data
scaler.fit(X_train)  
X_train = scaler.transform(X_train)  

# Apply the same transformation to test data
X_test = scaler.transform(X_test)

reg = MLPClassifier(max_iter=5000, hidden_layer_sizes=(5,5), random_state=1, solver = 'lbfgs')
print('ye') 
reg.fit(X_train, y_train)
print('ye')
# Predict
y_pred = reg.predict(X_test)
    
# Accuracy before model parameter optimisation
accuracy = accuracy_score(y_pred,y_test)

# Fit and check accuracy for various numbers of nodes on both layers
# Note this will take some time
validation_scores = {}
print("Nodes |Validation")
print("      | score")

for hidden_layer_size in [(i,j) for i in range(1,10) for j in range(1,10)]:

    reg = MLPClassifier(max_iter=5000, hidden_layer_sizes=hidden_layer_size, random_state=1,solver = 'lbfgs')

    score = cross_val_score(estimator=reg, X=X_train, y=y_train, cv=2)
    validation_scores[hidden_layer_size] = score.mean()
    print(hidden_layer_size, ": %0.5f" % validation_scores[hidden_layer_size])



# Check scores
print("The highest validation score is: %0.4f" % max(validation_scores.values()))  
optimal_hidden_layer_size = [name for name, score in validation_scores.items() 
                              if score==max(validation_scores.values())][0]
print("This corresponds to nodes", optimal_hidden_layer_size )

# Fit data with best parameter
clf = MLPClassifier(max_iter=5000, 
                    hidden_layer_sizes=optimal_hidden_layer_size, 
                    random_state=1,solver = 'lbfgs')
clf.fit(X_train, y_train)

# Predict
y_pred = clf.predict(X_test)

# Accuracy 
best_accuracy = accuracy_score(y_pred,y_test)
print("The highest accuracy is: %0.4f" % best_accuracy)  


