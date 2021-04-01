

import pandas as pd
import numpy as np

Data1=pd.read_csv('bank_transaction_features.csv')
Data2=pd.read_csv('bank_transaction_labels.csv')

Data = pd.merge(Data1, Data2,on='bank_transaction_id',how='inner')
del Data1, Data2
Data = Data.drop(['bank_transaction_id'], axis=1)
Data = Data.dropna()
Data = Data.reset_index(drop=True)
tData = Data["bank_transaction_dataset"]
Data = Data.drop(["bank_transaction_dataset"], axis=1)


from sklearn.preprocessing import LabelEncoder
#labelencoder_desc= LabelEncoder()
# Data.iloc[:,1]= labelencoder_desc.fit_transform(Data.iloc[:,1])
labelencoder_type= LabelEncoder()
Data.loc[:,"bank_transaction_type"]=labelencoder_type.fit_transform(Data.loc[:,"bank_transaction_type"])
labelencoder_cate= LabelEncoder()
Data.loc[:,"bank_transaction_category"]=labelencoder_cate.fit_transform(Data.loc[:,"bank_transaction_category"])

# remove - sign from amount column and scale data
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
Data.bank_transaction_amount = sc.fit_transform(Data.bank_transaction_amount.abs().values.reshape(-1, 1))

from sklearn.preprocessing import OneHotEncoder
onehotencoder = OneHotEncoder()
df1 = pd.DataFrame(onehotencoder.fit_transform(Data.bank_transaction_type.values.reshape(-1,1)).toarray())

for i in list(df1):
    Data[i] = df1[i]

Data = Data.drop(['bank_transaction_type'], axis=1)

# key work analysis

# finding keywords from description
cat = Data["bank_transaction_description"]
cat = pd.DataFrame([str(i).lower() for i in cat])
import re
string = ""
string = " ".join(str(x) for x in cat[0].values)
string = string.lower()
string = ''.join([i for i in string if not i.isdigit()])
string = re.sub('[!,*)-@#%(&$_?^]', '', string)
keywords = pd.DataFrame(string.split(" "))
keywords = keywords[0].value_counts()
keywords = keywords[1:,]
keywords

def get_description_keywords_score(description):

    description = description.lower()
    description = ''.join([i for i in description if not i.isdigit()])
    description = re.sub('[!,*)-@#%(&$_?^]', '', description)

    description_score = []
    for keyword in keywords.index:
        if keyword in description:
            description_score.append(1)
        else:
            description_score.append(0)
    return pd.DataFrame(np.array(description_score),index=np.array(keywords.index)).T


new_description = pd.DataFrame()
for i in Data["bank_transaction_description"]:
    new_description = new_description.append(get_description_keywords_score(i))

d = new_description
d["bank_transaction_amount"] = np.array(Data["bank_transaction_amount"])
d["0"] = np.array(Data[0])
d["1"] = np.array(Data[1])
d["2"] = np.array(Data[2])
d["3"] = np.array(Data[3])
d["4"] = np.array(Data[4])
d["5"] = np.array(Data[5])



xData = d

from sklearn.decomposition import PCA
pca = PCA(n_components=100)
xData = pd.DataFrame(pca.fit_transform(xData))
explained_variance = pca.explained_variance_ratio_


xData["target_category"] = Data["bank_transaction_category"]
xData["target_type"] = pd.DataFrame(tData)
xData = xData.sample(frac=1).reset_index(drop=True)

xDataTr = xData[xData["target_type"] == 'TRAIN']
xDataV = xData[xData["target_type"] == 'VAL']
yDataTr = xDataTr["target_category"]
yDataV = xDataV["target_category"]
xDataTr = xDataTr.drop(["target_category","target_type"],axis=1)
xDataV = xDataV.drop(["target_category","target_type"],axis=1)

# from sklearn.model_selection import train_test_split
# xDataTr, xDataTe, yDataTr, yDataTe = train_test_split(xDataTr, yDataTr, test_size = 0.2, random_state = 0)

from sklearn.preprocessing import OneHotEncoder
onehotencoder = OneHotEncoder()
yDataTr = pd.DataFrame(onehotencoder.fit_transform(yDataTr.values.reshape(-1,1)).toarray())


from keras.models import Sequential
from keras.layers import Dense
classifier = Sequential()
classifier.add(Dense(100, activation = 'relu', input_dim = xDataTe.shape[1]))
classifier.add(Dense(100, activation = 'relu'))
classifier.add(Dense(100, activation = 'relu'))
classifier.add(Dense(5,activation='sigmoid'))
classifier.compile(optimizer = 'rmsprop', loss = 'mse', metrics = ['accuracy'])

# training the model
classifier.fit(xDataTr,yDataTr, batch_size=20, epochs=500)

# obtaining the predicting
import numpy as np
y_pred = classifier.predict(xDataTe)
y_pred = np.argmax(y_pred, axis=-1)

# find the confusion matrix and accuracy
from sklearn.metrics import confusion_matrix,accuracy_score
cm = confusion_matrix(yDataTe,y_pred)
print("Accuracy is "+str(accuracy_score(yDataTe,y_pred))+"\n")


# Visualizing Outcomes
import matplotlib.pyplot as plt
import seaborn as sns;
fig, ax = plt.subplots(figsize=(10,10))
sns.heatmap(cm,cmap = "Blues",annot=True,square = True,linewidths=.5,ax=ax,fmt='g')


