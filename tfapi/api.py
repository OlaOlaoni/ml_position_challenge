import os
import json
import cv2
import requests
import numpy as np
import tensorflow as tf
from tensorflow import keras
from flask import Flask, request, jsonify

app = Flask(__name__)
port = int(os.environ.get("PORT", 80))
app.config["DEBUG"] = False # turn off in prod

## Load TF Model
model = tf.keras.models.load_model("/app/model_biT")

## Define class names
class_names= ["Damaged Car", "Car In Good Condition"]

def tf_image2tensor(image):
    """
    Receives a image as bytes as input, that will be loaded,
    preprocessed and turned into a Tensor so as to include it
    in the TF-Serving request data.
    """

    # Apply the same preprocessing as during training (resize and rescale)
    img = tf.image.resize(image, [224, 224])
    img = np.expand_dims(img, axis = 0)
    
    return img

def predict_tensor_image(image):
    image = tf_image2tensor(image)
    predicted_prob = model.predict(image)
    topk_prob, topk_id = tf.math.top_k(predicted_prob)
    topk_label = np.array(class_names)[topk_id.numpy()]

    return topk_label

def inference(image_url: str):
    image_reponse = requests.get(image_url)
    image_as_np_array = np.frombuffer(image_reponse.content, np.uint8)
    image = cv2.imdecode(image_as_np_array, cv2.IMREAD_COLOR)
    img = tf.image.resize(image, [224, 224])
    img = np.expand_dims(img, axis = 0)
    # Save the image into a remote DIR
    # filename  = check_dir_exist()
    # cv2.imwrite(filename + ".png", image)

    # Make Predictions
    return predict_single_image(img)

def predict_single_image(imagei):
    predicted_prob = model.predict(imagei)
    topk_prob, topk_id = tf.math.top_k(predicted_prob)
    topk_label = np.array(class_names)[topk_id.numpy()]

    return topk_label

@app.route('/', methods=["GET"])
def index_page():
    '''
    Welcome API
    '''
    return_data = {
            "model_version_status": [
            {
                "version": "1",
                "state": "AVAILABLE",
                "status": {
                    "error_code": "OK",
                    "error_message": "None"
                }
            }
        ]
    }
    return app.response_class(response=json.dumps(return_data), mimetype='application/json'),200

@app.route("/v1/model/predict/tensor", methods = ["POST"])
def model_predict_image():
    try:
        image_tensor = request.json["image_tensor"]
        if image_tensor != None:
            predictions = predict_tensor_image(image_tensor)
            
            response = {
                "result" : {
                    "Predictions": list(predictions.flatten())
                    
                }
            }
        else:
            response = {
                "error" : '1',
                "message": "Invalid Parameters ==> Check if u are passsing the imageUrl in the body of the POST request"          
            }
    except Exception as e:
        response = {
            'error' : '2',
            "message": f"An error occured:{str(e)}"
            }
    return jsonify(response)
    

@app.route("/v1/model/predict", methods=["POST", "GET"])
def model_predict():
    '''
    Object Detection prediction endpoint
    '''
    try:
        image_url = request.json["imageUrl"]
        if image_url != None:
            predictions = inference(image_url)
            
            response = {
                "result" : {
                    "Predictions": list(predictions.flatten())
                    
                }
            }
        else:
            response = {
                "error" : '1',
                "message": "Invalid Parameters ==> Check if u are passsing the imageUrl in the body of the POST request"          
            }
    except Exception as e:
        response = {
            'error' : '2',
            "message": f"An error occured:{str(e)}"
            }
    return jsonify(response)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=port, use_reloader=False)