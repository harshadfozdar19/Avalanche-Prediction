import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
import pickle

data=pd.read_csv('model_dataset.csv')
data.head()

data.dropna(subset = ['avalanche_act'],inplace = True)

if 'danger_level' in data.columns:
  data.drop(columns=['danger_level'],inplace=True)

X_train, X_test, y_train, y_test = train_test_split(data.drop(columns='avalanche_act'), data['avalanche_act'], test_size=0.25)
DTC = DecisionTreeClassifier()
DTC.fit(X_train, y_train)

y_pred = DTC.predict(X_test)
print("Accuracy:", round(metrics.accuracy_score(y_test, y_pred)*100, 2),'%')

if DTC.predict([[2403.0, -7.0, 45.0, 62.0]]) == 1:
  print('LEAVE THE AREA')
else:
  print('SAFE')

with open('decision_tree_model.pkl', 'wb') as file:
    pickle.dump(DTC, file)

with open('decision_tree_model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)


if loaded_model.predict([[24, -7, 45, 64]]) == 1:
  print('LEAVE THE AREA')
else:
  print('SAFE')
