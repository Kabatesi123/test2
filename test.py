import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import io

# Function to train and return the model
def train_model(data):
    # Encode categorical columns
    label_encoder = LabelEncoder()
    data['Previous_Purchase'] = label_encoder.fit_transform(data['Previous_Purchase'])  # Yes=1, No=0
    data['Purchased'] = label_encoder.fit_transform(data['Purchased'])  # Yes=1, No=0

    # Define features (X) and target (y)
    X = data.drop(columns=["Customer_ID", "Purchased"])
    y = data["Purchased"]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the Random Forest Classifier
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Evaluate the model (you can display this if needed)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    return model, label_encoder, accuracy

# Streamlit UI
def main():
    st.title("Customer Purchase Prediction")

    # Upload CSV file
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

    if uploaded_file is not None:
        # Read the uploaded CSV into a DataFrame
        df = pd.read_csv(uploaded_file)
        st.write("Dataset Preview:")
        st.write(df.head())

        # Train model on the uploaded dataset
        model, label_encoder, accuracy = train_model(df)
        st.write(f"Model accuracy: {accuracy:.2f}")

        # Predict using the trained model
        features = df.drop(columns=["Customer_ID", "Purchased"])
        features['Previous_Purchase'] = label_encoder.transform(features['Previous_Purchase'])

        predictions = model.predict(features)

        # Convert predictions to 'Yes'/'No'
        predictions = label_encoder.inverse_transform(predictions)
        df['Predicted_Purchase'] = predictions

        # Show the prediction results with Customer_ID
        st.write("Customer Purchase Predictions:")
        st.write(df[['Customer_ID', 'Predicted_Purchase']])

        # Allow user to download the result as a CSV
        csv = df[['Customer_ID', 'Predicted_Purchase']].to_csv(index=False)
        st.download_button(
            label="Download Predictions",
            data=csv,
            file_name="customer_predictions.csv",
            mime="text/csv"
        )

        # Provide a template to add data
        st.write("Download the template to add new customer data:")
        template_data = {
            "Customer_ID": ["", "", ""],
            "Age": [None, None, None],
            "Income ($)": [None, None, None],
            "Browsing_Time (minutes)": [None, None, None],
            "Items_in_Cart": [None, None, None],
            "Previous_Purchase": ["", "", ""]
        }
        template_df = pd.DataFrame(template_data)
        template_csv = template_df.to_csv(index=False)
        st.download_button(
            label="Download Template",
            data=template_csv,
            file_name="customer_data_template.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    main()
