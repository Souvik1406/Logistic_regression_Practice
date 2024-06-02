import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv('insurance_data.csv')

# Features (X) and target (y)
X = df[['Age']]
y = df['Accepted']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create and train the Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Visualize the decision boundary
sns.set(style="whitegrid")

plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, color='blue', label='Actual values')
plt.scatter(X_test, y_pred, color='red', label='Predicted values', alpha=0.6)

plt.title('Logistic Regression: Insurance Acceptance by Age')
plt.xlabel('Age')
plt.ylabel('Accepted')
plt.legend()
plt.show()

# Custom age input from user
custom_age = float(input("Enter the age to predict insurance acceptance: "))
custom_age_pred = model.predict([[custom_age]])

# Print the prediction result
if custom_age_pred[0] == 1:
    print(f"The model predicts that insurance will be accepted for age {custom_age}.")
else:
    print(f"The model predicts that insurance will not be accepted for age {custom_age}.")
