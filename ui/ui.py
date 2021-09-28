import cv2
import requests
import numpy as np
import streamlit as st

REST_URL = "http://34.211.110.81:80/v1/model/predict/tensor"

def main():
    # Streamlit initialization
    html_temp = """
        <div style="background-color:green;padding:5px">
        <h2 style="color:white;text-align:center;">Curacel ML Position Challenge</h2>
        </div>
        """
    st.markdown(html_temp,unsafe_allow_html=True)

    # Create a FileUploader so that the user can upload an image to the UI
    uploaded_file = st.file_uploader(label="Upload an image of any of the Test Car",
                                 type=["png", "jpeg", "jpg"])

    # Display the predict button just when an image is being uploaded
    if not uploaded_file:
        st.warning("Please upload an image before proceeding!")
        st.stop()
    else:
        # image_as_bytes = uploaded_file.read()
        image_as_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(image_as_bytes, 1)
        img_cv = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        st.image(img_cv, use_column_width=True)
        pred_button = st.button("Predict")
    
    if pred_button:
        # predictions = predict_single_image(img)
        # list_predict = list(predictions.flatten())

        # Prepare the data that is going to be sent in the POST request
        json_data = {
            "image_tensor": img.tolist()
        }

        # Send the request to the Prediction API
        response = requests.post(REST_URL, json=json_data).json()

        st.success('Detected:\n\n {}'.format(response['result']['Predictions'][0]))   

if __name__ == '__main__':
    main()
