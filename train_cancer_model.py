import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report


cancer = pd.read_csv('Cancer.csv')
y = cancer['diagnosis']
X = cancer.drop(['id','diagnosis','Unnamed: 32'],axis=1)

X_train, X_test, y_train, y_test = train_test_split(X,y, train_size=0.7, random_state=2529)

model = LogisticRegression(max_iter=5000)
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
accuracy_score(y_test,y_pred)
print(classification_report(y_test,y_pred))     

# Save the trained model
pickle.dump(model, open('C:/Users/Krishna/PycharmProjects/EDAIProject/models/cancer_model.sav', 'wb'))

print("Cancer disease model trained and saved.")
