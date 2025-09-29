import pickle
import numpy as np
import pandas as pd

# Load the trained SVM model
with open('best_svm_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Assuming you have the scaler and encoder used during training
# You would need to save and load these as well
# Example:
# with open('scaler.pkl', 'rb') as f:
#     scaler = pickle.load(f)
# with open('encoder.pkl', 'rb') as f:
#     encoder = pickle.load(f)


def predict_stocking_intent(data):
    """
    Predicts the stocking intent based on input data.

    Args:
        data (dict): A dictionary containing the input features.

    Returns:
        str: The predicted stocking intent ('Ready' or 'Not Ready').
    """
    # Convert input data to a pandas DataFrame
    input_df = pd.DataFrame([data])

    # --- Preprocessing steps (match the training preprocessing) ---
    # This is a simplified example. You would need to apply the same
    # encoding and scaling used during training.

    # Example (assuming you saved the encoder and scaler):
    # categorical_cols = ['Outlet_Type', 'Stock_Availability', 'Return_Policy_Awareness', 'Margin', 'Reason'] # Add other categorical cols if any
    # numerical_cols = ['Girnar', 'Wagh_Bakri', 'Red_Label', 'Granuel_Beans', 'Other', 'Has_Shelf_Space', 'Total_Competitor_Presence', 'Is_Urban'] # Add other numerical cols if any

    # input_encoded = encoder.transform(input_df[categorical_cols])
    # input_encoded_df = pd.DataFrame(input_encoded, columns=encoder.get_feature_names_out(categorical_cols), index=input_df.index)
    # input_df = input_df.drop(categorical_cols, axis=1)
    # input_df = pd.concat([input_df, input_encoded_df], axis=1)

    # input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])
    # --------------------------------------------------------------

    # Make prediction
    prediction = model.predict(input_df)

    # Decode the prediction (assuming you have the label_encoder)
    # Example:
    # predicted_intent = label_encoder.inverse_transform(prediction)
    # return predicted_intent[0]

    # For now, returning the raw prediction as we don't have the label_encoder here
    return "Ready" if prediction[0] == 1 else "Not Ready"

if __name__ == '__main__':
    # Example usage:
    # Create a sample data dictionary with the required features
    sample_data = {
        'Outlet_Type': 'GT',
        'Girnar': 1,
        'Wagh_Bakri': 0,
        'Red_Label': 0,
        'Granuel_Beans': 0,
        'Other': 0,
        'Stock_Availability': 'Sometimes',
        'Return_Policy_Awareness': 'Aware',
        'Margin': 'Low',
        'Reason': 'Brand Trust',
        'Has_Shelf_Space': 1,
        'Total_Competitor_Presence': 1,
        'Is_Urban': 1
    }
    predicted_intent = predict_stocking_intent(sample_data)
    print(f"Predicted Stocking Intent: {predicted_intent}")
