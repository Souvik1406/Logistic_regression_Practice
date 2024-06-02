import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Set Streamlit page configuration
st.set_page_config(page_title="Insurance Acceptance Prediction", layout="centered")

# Title and description
st.title("Insurance Acceptance Prediction Based on Age")
st.write("This application uses Logistic Regression to predict insurance acceptance based on age.")

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv('insurance_data.csv')
    return df

df = load_data()

# Show dataset
if st.checkbox("Show dataset"):
    st.write(df.head())

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
st.subheader("Model Evaluation")
st.write("Confusion Matrix")
st.write(confusion_matrix(y_test, y_pred))

st.write("Classification Report")
st.write(classification_report(y_test, y_pred))

# Visualize the decision boundary
st.subheader("Visualization")

# Plot the results
fig, ax = plt.subplots()
ax.scatter(X_test, y_test, color='blue', label='Actual values')
ax.scatter(X_test, y_pred, color='red', label='Predicted values', alpha=0.6)
ax.set_title('Logistic Regression: Insurance Acceptance by Age')
ax.set_xlabel('Age')
ax.set_ylabel('Accepted')
ax.legend()
st.pyplot(fig)

# Take custom age input from user and make prediction
st.subheader("Predict Insurance Acceptance")
custom_age = st.number_input("Enter age to predict insurance acceptance", min_value=0, max_value=120, step=1)

if st.button("Predict"):
    custom_prediction = model.predict([[custom_age]])
    result = 'Accepted' if custom_prediction[0] else 'Not Accepted'
    st.write(f"Predicted acceptance for age {custom_age}: {result}")

# Additional decorations
st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    .stButton button {
        background-color: #007BFF;
        color: white;
    }
</style>
""", unsafe_allow_html=True)
