import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the dataset
heart_dataset = pd.read_csv('C:/Users/Krishna/PycharmProjects/EDAIProject/cardiovascular disease data.csv')
# Encode categorical columns using LabelEncoder
le = LabelEncoder()
for col in heart_dataset.columns:
    if heart_dataset[col].dtype == 'object':
        heart_dataset[col] = le.fit_transform(heart_dataset[col])

# Separating the data and labels
X = heart_dataset.drop(columns=['target'])
Y = heart_dataset['target']

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

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

# Save the trained model
pickle.dump(rf_model, open('C:/Users/Krishna/PycharmProjects/EDAIProject/models/heart_disease_model.sav', 'wb'))

print("Heart disease model trained and saved.")
