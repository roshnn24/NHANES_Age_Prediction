# import streamlit as st
# import numpy as np
# import joblib
# import tensorflow as tf
#
# # Load the pre-trained model, scaler, and label encoder
# model = tf.keras.models.load_model("/Users/rosh/Downloads/best_model.h5")
# scaler = joblib.load("/Users/rosh/Downloads/final_scaler.pkl")
# label_encoder = joblib.load("/Users/rosh/Downloads/final_label_encoder.pkl")
#
# # Define the features
# features = ['RIAGENDR', 'PAQ605', 'BMXBMI', 'LBXGLU', 'DIQ010', 'LBXGLT', 'LBXIN']
#
# # Streamlit app
# st.title('Age Prediction Model')
#
# # User input
# st.header('Enter the following features:')
# user_input = []
# for feature in features:
#     value = st.number_input(f"Enter {feature}:", value=0.0)
#     user_input.append(value)
#
# # Convert the user input to a NumPy array
# user_input_array = np.array(user_input).reshape(1, -1)
#
# # Predict button
# if st.button('Predict Age Group'):
#     # Scale the user input
#     user_input_scaled = scaler.transform(user_input_array)
#
#     # Make a prediction
#     prediction = model.predict(user_input_scaled)
#
#     # Decode the prediction
#     predicted_class = np.argmax(prediction, axis=1)
#     predicted_age_group = label_encoder.inverse_transform(predicted_class)
#
#     # Display the prediction
#     st.subheader(f'Predicted Age Group: {predicted_age_group[0]}')
#
s = "Hi"
print(s[::-1])