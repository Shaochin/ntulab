from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

lungcancer=pd.read_csv("lungcancerstudy.csv")

# over_sample = SMOTE(random_state=0)
# resampled=over_sample.fit_sample(lungcancer)

dataTrain, datatest=train_test_split(lungcancer,test_size=0.2)



rf = RandomForestClassifier(criterion='gini', 
                             n_estimators=1000,
                             min_samples_split=12,
                             min_samples_leaf=1,
                             oob_score=True,
                             random_state=1,
                             n_jobs=-1) 

rf.fit(dataTrain.iloc[:,1:],dataTrain.iloc[:,0])
print("accuracy_score:"+"%.4f" % rf.oob_score_)

result=rf.predict(datatest.iloc[:,1:])

print("score:"+"%.4f"%rf.score(datatest.iloc[:,1:],datatest.iloc[:,0]))



# print(dataTrain)

# print("test:")
# print(datatest)

#sns.countplot(lungcancer['S1'],hue=lungcancer['group'])





# #plt.show()
# #lungcancer.info()
