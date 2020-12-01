#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
plt.figure(figsize=(20, 5))
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn import linear_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import seaborn as sns 
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier 
from sklearn.feature_selection import RFE

def performance_metric(y_true, y_predict, normalize=True):
    score = r2_score(y_true, y_predict)
    return score

data = pd.read_csv("../input/kc_house_data.csv", encoding = "ISO-8859-1")
Y = data["price"]
X = data[["bedrooms", "bathrooms", "sqft_living", "sqft_lot", "floors", "waterfront", "view", "grade", "sqft_above", "sqft_basement", "yr_built", "yr_renovated", "zipcode", "lat", "long"]]
colnames = X.columns

#ranking columns
ranks = {}
def ranking(ranks, names, order=1):
    minmax = MinMaxScaler()
    ranks = minmax.fit_transform(order*np.array([ranks]).T).T[0]
    ranks = [round(x,2) for x in ranks]
    return dict(list(zip(names, ranks)))

for i, col in enumerate(X.columns):
    # 3 plots here hence 1, 3
    plt.subplot(1, 15, i+1)
    x = X[col]
    y = Y
    plt.plot(x, y, 'o')
    # Create regression line
    plt.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)))
    plt.title(col)
    plt.xlabel(col)
    plt.ylabel('prices')
    
   
#Splitting the datasets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=10)

#Models

#Decision Tree Regressor
DTR = tree.DecisionTreeRegressor()
DTR = DTR.fit(X_train,y_train)
ranks["DTR"] = ranking(np.abs(DTR.feature_importances_), colnames)

Y_target_DTR = DTR.predict(X_test)

#Decision Tree Classifier
DTC = DecisionTreeClassifier(max_depth=None, min_samples_split=2, random_state=0)
DTC = DTC.fit(X_train, y_train)
ranks["DTC"] = ranking(np.abs(DTC.feature_importances_), colnames)

Y_target_DTC = DTC.predict(X_test)

#LARS Lasso
LARS_L = linear_model.LassoLars(alpha=.4)
LARS_L = LARS_L.fit(X_train, y_train)
ranks["LARS_L"] = ranking(np.abs(LARS_L.coef_), colnames)

Y_target_lars_l = LARS_L.predict(X_test)

#Bayesian Ridge 
BR = linear_model.BayesianRidge()
BR = BR.fit(X_train, y_train)
ranks["BR"] = ranking(np.abs(BR.coef_), colnames)

Y_target_BR = BR.predict(X_test)

#Random Forest Regressor
RFR = RandomForestRegressor(n_jobs=-1, n_estimators=50, verbose=0)
RFR = RFR.fit(X_train,y_train)
ranks["RFR"] = ranking(RFR.feature_importances_, colnames);
#print(ranks["RFR"])

Y_target_RFR = RFR.predict(X_test)

#Recursive Feature Elimination on Random Forest Regressor
RFE_RFR = RFE(RFR, n_features_to_select=10, step = 1)
RFE_RFR.fit(X_train,y_train)

Y_target_RFE_RFR = RFE_RFR.predict(X_test)

#Extra Trees Classifier

ETC = ExtraTreesClassifier(n_estimators=10, max_depth=None, min_samples_split=2, random_state=0)
ETC = ETC.fit(X_train, y_train)
ranks["ETC"] = ranking(np.abs(ETC.feature_importances_), colnames)

Y_target_ETC = ETC.predict(X_test)

#Recursive Feature Elimination on Decision Tree Regressor
RFE = RFE(DTR, n_features_to_select=10, step =1 )
RFE.fit(X_train,y_train)

Y_target_RFE = RFE.predict(X_test)


#Ranking inputs
r = {}
for name in colnames:
    r[name] = round(np.mean([ranks[method][name] 
                             for method in list(ranks.keys())]), 2)
 
methods = sorted(ranks.keys())
ranks["Mean"] = r
methods.append("Mean")
 
print(("\t%s" % "\t".join(methods)))
for name in colnames:
    print(("%s\t%s" % (name, "\t".join(map(str, 
                         [ranks[method][name] for method in methods])))))
    
#seaborn plot
#create dataframe
meanplot = pd.DataFrame(list(r.items()), columns= ['Feature','Mean Ranking'])
meanplot = meanplot.sort_values('Mean Ranking', ascending=False)
#plot proper
sns.factorplot(x="Mean Ranking", y="Feature", data = meanplot, kind="bar", 
               size=14, aspect=1.9, palette='coolwarm')



#R2 metrics for each model

print("\nR2 score, Decision Tree Regressor:") 
print((performance_metric(y_test, Y_target_DTR)))

print("\nR2 score, Decision Tree Classifier:") 
print((performance_metric(y_test, Y_target_DTC)))

print("\nR2 score, LARS Lasso:") 
print((performance_metric(y_test, Y_target_lars_l)))

print("\nR2 score, Bayesian Ridge:") 
print((performance_metric(y_test, Y_target_BR)))

print("\nR2 score, Random Forest Regressor:") 
print((performance_metric(y_test, Y_target_RFR)))

print("\nR2 score, Recursive Feature Eliminition on Random Forest Regressor:") 
print((performance_metric(y_test, Y_target_RFE_RFR)))

print("\nR2 score, Extra Trees Classifier:") 
print((performance_metric(y_test, Y_target_ETC)))

print("\nR2 score, Recursive Feature Eliminition on Decision Tree Regressor:") 
print((performance_metric(y_test, Y_target_RFE)))


# In[ ]:




