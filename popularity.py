import os
import warnings
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
# %matplotlib inline
sns.set_style("whitegrid")

warnings.filterwarnings("ignore")

dataframe = pd.read_csv(
    os.getcwd()+'\datasets\FeaturesExtracted\SpotifyFeatures.csv')
# dataframe.head()

dataframe.describe()
pd.isnull(dataframe).sum()
sns.distplot(dataframe['popularity']).set_title('Popularity Distribution')
dataframe.corr()
ig = plt.figure(figsize=(18, 10))
sns.heatmap(dataframe.corr(), annot=True)
plt.xticks(rotation=45)

sns.barplot(x='time_signature', y='popularity', data=dataframe)
plt.title('Popularity Based on Time Signature')

sns.barplot(x='key', y='popularity', data=dataframe)
plt.title('Popularity Based on Key')

sns.barplot(x='mode', y='popularity', data=dataframe)
plt.title('Popularity Based on Mode')

sns.barplot(x='mode', y='popularity', hue='key', data=dataframe)
plt.title('Popularity Based on Mode and Key')

sns.jointplot(x='acousticness', y='popularity', data=dataframe)
sns.jointplot(x='loudness', y='popularity', data=dataframe)

popular_above_50 = dataframe[dataframe.popularity > 50]
sns.distplot(popular_above_50['acousticness'])
plt.title('Acoustiness for Songs with More than 50 Popularity')

popular_below_50 = dataframe[dataframe.popularity < 50]
sns.distplot(popular_below_50['acousticness'])
plt.title('Acoustiness for Songs with Less than 50 Popularity')

sns.distplot(popular_above_50['loudness'])
plt.title('Loudness for Songs with More than 50 Popularity')

popular_below_50 = dataframe[dataframe.popularity < 50]
sns.distplot(popular_below_50['loudness'])
plt.title('Loudness for Songs with Less than 50 Popularity')

sns.pairplot(dataframe)
list_of_keys = dataframe['key'].unique()
for i in range(len(list_of_keys)):
    dataframe.loc[dataframe['key'] == list_of_keys[i], 'key'] = i
dataframe.sample(5)


dataframe.loc[dataframe["mode"] == 'Major', "mode"] = 1
dataframe.loc[dataframe["mode"] == 'Minor', "mode"] = 0
dataframe.sample(5)

list_of_time_signatures = dataframe['time_signature'].unique()
for i in range(len(list_of_time_signatures)):
    dataframe.loc[dataframe['time_signature'] ==
                  list_of_time_signatures[i], 'time_signature'] = i
dataframe.sample(5)

dataframe.loc[dataframe['popularity'] < 57, 'popularity'] = 0
dataframe.loc[dataframe['popularity'] >= 57, 'popularity'] = 1
dataframe.loc[dataframe['popularity'] == 1]

features = ["acousticness", "danceability", "duration_ms", "energy", "instrumentalness", "key", "liveness",
            "mode", "speechiness", "tempo", "time_signature", "valence"]

training = dataframe.sample(frac=0.8, random_state=420)
X_train = training[features]
y_train = training['popularity']
X_test = dataframe.drop(training.index)[features]

X_train, X_valid, y_train, y_valid = train_test_split(
    X_train, y_train, test_size=0.2, random_state=420)

LR_Model = LogisticRegression()
LR_Model.fit(X_train, y_train)
LR_Predict = LR_Model.predict(X_valid)
LR_Accuracy = accuracy_score(y_valid, LR_Predict)
print("Accuracy: " + str(LR_Accuracy))

LR_AUC = roc_auc_score(y_valid, LR_Predict)
print("AUC: " + str(LR_AUC))

KNN_Model = KNeighborsClassifier()
KNN_Model.fit(X_train, y_train)
KNN_Predict = KNN_Model.predict(X_valid)
KNN_Accuracy = accuracy_score(y_valid, KNN_Predict)
print("Accuracy: " + str(KNN_Accuracy))

KNN_AUC = roc_auc_score(y_valid, KNN_Predict)
print("AUC: " + str(KNN_AUC))

DT_Model = DecisionTreeClassifier()
DT_Model.fit(X_train, y_train)
DT_Predict = DT_Model.predict(X_valid)
DT_Accuracy = accuracy_score(y_valid, DT_Predict)
print("Accuracy: " + str(DT_Accuracy))

DT_AUC = roc_auc_score(y_valid, DT_Predict)
print("AUC: " + str(DT_AUC))

training_LSVC = training.sample(10000)
X_train_LSVC = training_LSVC[features]
y_train_LSVC = training_LSVC['popularity']
X_test_LSVC = dataframe.drop(training_LSVC.index)[features]
X_train_LSVC, X_valid_LSVC, y_train_LSVC, y_valid_LSVC = train_test_split(
    X_train_LSVC, y_train_LSVC, test_size=0.2, random_state=420)

LSVC_Model = DecisionTreeClassifier()
LSVC_Model.fit(X_train_LSVC, y_train_LSVC)
LSVC_Predict = LSVC_Model.predict(X_valid_LSVC)
LSVC_Accuracy = accuracy_score(y_valid_LSVC, LSVC_Predict)
print("Accuracy: " + str(LSVC_Accuracy))

LSVC_AUC = roc_auc_score(y_valid_LSVC, LSVC_Predict)
print("AUC: " + str(LSVC_AUC))

RFC_Model = RandomForestClassifier()
RFC_Model.fit(X_train, y_train)
RFC_Predict = RFC_Model.predict(X_valid)
RFC_Accuracy = accuracy_score(y_valid, RFC_Predict)
print("Accuracy: " + str(RFC_Accuracy))

RFC_AUC = roc_auc_score(y_valid, RFC_Predict)
print("AUC: " + str(RFC_AUC))

model_performance_accuracy = pd.DataFrame({'Model': ['LogisticRegression',
                                                     'RandomForestClassifier',
                                                     'KNeighborsClassifier',
                                                     'DecisionTreeClassifier',
                                                     'LinearSVC'
                                                     ],
                                           'Accuracy': [LR_Accuracy,
                                                        RFC_Accuracy,
                                                        KNN_Accuracy,
                                                        DT_Accuracy,
                                                        LSVC_Accuracy
                                                        ]})

model_performance_AUC = pd.DataFrame({'Model': ['LogisticRegression',
                                                'RandomForestClassifier',
                                                'KNeighborsClassifier',
                                                'DecisionTreeClassifier',
                                                'LinearSVC'
                                                ],
                                      'AUC': [LR_AUC,
                                              RFC_AUC,
                                              KNN_AUC,
                                              DT_AUC,
                                              LSVC_AUC
                                              ]})

model_performance_accuracy.sort_values(by="Accuracy", ascending=False)

model_performance_AUC.sort_values(by="AUC", ascending=False)
