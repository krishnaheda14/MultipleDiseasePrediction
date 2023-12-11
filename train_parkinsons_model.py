import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the Parkinson's disease dataset
parkinsons_dataset = pd.read_csv('C:/Users/Krishna/PycharmProjects/EDAIProject/parkinsons.csv')

X_parkinsons = parkinsons_dataset.drop(columns=['name', 'status'], axis=1)
Y_parkinsons = parkinsons_dataset['status']


# Split the data into training and testing sets
X_train_parkinsons, X_test_parkinsons, Y_train_parkinsons, Y_test_parkinsons = train_test_split(
    X_parkinsons, Y_parkinsons, test_size=0.2, random_state=42
)

# Standardize the features using StandardScaler
scaler_parkinsons = StandardScaler()
X_train_scaled_parkinsons = scaler_parkinsons.fit_transform(X_train_parkinsons)
X_test_scaled_parkinsons = scaler_parkinsons.transform(X_test_parkinsons)

# Initialize the Random Forest model
rf_model_parkinsons = RandomForestClassifier(random_state=42)

# Fit the model on the scaled training data
rf_model_parkinsons.fit(X_train_scaled_parkinsons, Y_train_parkinsons)

# Make predictions on the training set
Y_train_prediction_parkinsons = rf_model_parkinsons.predict(X_train_scaled_parkinsons)

# Calculate and print the training accuracy
training_accuracy_parkinsons = accuracy_score(Y_train_parkinsons, Y_train_prediction_parkinsons)
print(f"Training Accuracy for Parkinson's Disease: {training_accuracy_parkinsons:.2f}")

# Save the trained model
pickle.dump(rf_model_parkinsons, open('C:/Users/Krishna/PycharmProjects/EDAIProject/models/parkinsons_model.sav', 'wb'))

print("Parkinson's disease model trained and saved.")
