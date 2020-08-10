# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 01:02:52 2020

@author: Sohail Tulsi
"""


import pandas as pd 
from sklearn.tree import DecisionTreeClassifier, plot_tree  
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


df_to_use2 = df_to_use1.reset_index()

# Split data into features (X) and response (y)
X = df_to_use2.loc[:,["Age","Gender0","Finance_Degree","Science_Degree","WorkExperience","Salary","Distance"]]
y = df_to_use2.loc[:,["Transport"]] 
y = np.ravel(y)

# Split the data into the training set and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
# Fit data to a tree-based classification model
classifier = DecisionTreeClassifier(random_state=0)
classifier.fit(X_train, y_train)

# What is the accuracy before pruning?
# Assign the accuracy score to the variable name test_score.
y_pred = classifier.predict(X_test)
test_score = accuracy_score(y_test, y_pred)

# Print the test score.
print("Accuracy score of the tree = {:2.2%}".format(test_score)) 

# Plot the full tree
plt.figure()
plot_tree(classifier,feature_names=X.columns)
plt.show()
print("Tree depth =",classifier.get_depth(),'\n'
      "Number of leaves =",classifier.get_n_leaves())

# Finding the optimal number of samples per leaf
samples = [sample for sample in range(1,30)]     

classifiers = []
for sample in samples:
    classifier2 = DecisionTreeClassifier(random_state=0, 
                                         min_samples_leaf=sample)
    classifier2.fit(X_train, y_train)
    classifiers.append(classifier2)

# Visualise the performance of each subtree on the training and test sets
train_scores = [clf.score(X_train, y_train) for clf in classifiers]
test_scores = [clf.score(X_test, y_test) for clf in classifiers]

fig, ax = plt.subplots()
ax.set_xlabel("Minimum leaf samples")
ax.set_ylabel("Accuracy")
ax.set_title("Comparing the training and test set accuracy")
ax.plot(samples, train_scores, marker='o', label="train",
        drawstyle="steps-post")
ax.plot(samples, test_scores, marker='o', label="test",
        drawstyle="steps-post")
ax.legend()
plt.show()

# In order to find the optimal minimum leaf samples, cross validation is applied
validation_scores = []
for sample in samples:
    classifier3 = DecisionTreeClassifier(random_state=1, min_samples_leaf=sample)
    score = cross_val_score(estimator=classifier3, X=X_train, y=y_train, cv=5)   
    validation_scores.append(score.mean())

# Visualise the validation score in relation to minimum leaf samples
plt.figure()
plt.xlabel("Minimum leaf samples")
plt.ylabel("Validation score")
plt.title("Validation scores at different minimum leaf sample counts")
plt.plot(samples, validation_scores, marker='o', label="train",
        drawstyle="steps-post")
plt.legend()
plt.show()

# Obtain the minimum leaf samples with the highest validation score
samples_optimum = samples[validation_scores.index(max(validation_scores))]
print(samples_optimum)

# Use the optimum  minimun leaf samples to fit a parsimonious tree
classifier4 = DecisionTreeClassifier(random_state=0, min_samples_leaf=samples_optimum)
classifier4.fit(X_train, y_train)

# Visualise the smaller pruned tree
plt.figure()
plot_tree(classifier4, feature_names=X_train.columns)
plt.show()

# Show the first few levels of the tree
plt.figure(figsize=[10,5], dpi=300)
plot_tree(classifier4, max_depth=6, 
          feature_names=X_train.columns, 
          class_names=['Car', 'Motorbike','Public Transport'],
          impurity=False,
          filled=True)
#plt.savefig('HRDataset_tree.png')
plt.show()

# Final test to see how the model performs:
y_pred = classifier4.predict(X_test)
test_score2 = accuracy_score(y_test, y_pred)
print("Accuracy score of the optimal tree = {:2.2%}".format(test_score2))
print("Tree depth =",classifier4.get_depth(),'\n'
      "Number of leaves =",classifier4.get_n_leaves()) 


# one hot encode gender for prediction 

# One hot encoding gender
df2 = df2.reset_index()

Xd_pred = ohe.fit_transform(df2.Gender.values.reshape(-1, 1)).toarray()
df_ohe_pred = pd.DataFrame(Xd_pred, columns = ["Gender"+str(int(i)) for i in range(Xd_pred.shape[1])])

# Add gender encoded feature to the dataframe
df_to_use_pred = pd.concat([df2, df_ohe_pred],axis = 1)


#drop columns not applicable
df_to_use_pred = df_to_use_pred.loc[:,["Age","Gender0","Finance_Degree","Science_Degree","WorkExperience","Salary","Distance"]]

# Fit the final model
best_model = DecisionTreeClassifier(random_state=0, min_samples_leaf=samples_optimum)
best_model.fit(X, y)
prediction = best_model.predict(df_to_use_pred)
#df2.append(prediction)
final_prediction = pd.DataFrame({'predicted_transport': prediction[:]})

frames = [df2, final_prediction]
result = pd.concat(frames, axis=1, sort=False)

result.to_csv('DecisionTreePrediction.csv', index= False)
