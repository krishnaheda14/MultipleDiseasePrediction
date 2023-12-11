import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import pickle
# Load the dataset
diabetes_dataset = pd.read_csv('C:/Users/Krishna/PycharmProjects/EDAIProject/diabetes.csv')

# Separating the data and labels
X = diabetes_dataset.drop(columns='Outcome', axis=1)
Y = diabetes_dataset['Outcome']

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

# Standardize the features using StandardScaler
standardScaler = StandardScaler()
X_train_scaled = standardScaler.fit_transform(X_train)
X_test_scaled = standardScaler.transform(X_test)

# Initialize the Random Forest model
rf_model = RandomForestClassifier(random_state=42)

# Fit the model on the scaled training data
rf_model.fit(X_train_scaled, Y_train)

# Make predictions on the training set
Y_train_prediction = rf_model.predict(X_train_scaled)

# Calculate and print the training accuracy
training_accuracy = accuracy_score(Y_train, Y_train_prediction)
print(f"Training Accuracy: {training_accuracy*100:.2f}")

#Save the trained model
pickle.dump(rf_model, open('C:/Users/Krishna/PycharmProjects/EDAIProject/models/diabetes_model.sav', 'wb'))

print("Diabetes disease model trained and saved.")

