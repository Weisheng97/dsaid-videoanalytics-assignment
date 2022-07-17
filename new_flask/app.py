from flask import Flask, render_template, request
from flask import jsonify
import numpy as np
import PIL.Image as Image
import os
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow import keras
from pythonping import ping

app = Flask(__name__)

@app.route('/', methods= ['GET'])
def create_page():   #ping and create webpage
    print(ping('127.0.0.1'))
    return render_template('index.html')

@app.route('/', methods= ['POST'])
def prediction():   #infer
    imagefile = request.files['imagefile']
    image_path = "./images/" + imagefile.filename
    imagefile.save(image_path)

    image = load_img(image_path, target_size=(224,224))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    
    loaded_model = tf.keras.models.load_model(('my_model.h5') ,custom_objects={'KerasLayer':hub.KerasLayer})
    prediction_scores = loaded_model.predict(image)
    prediction = np.argmax(prediction_scores)

    return render_template('index.html', prediction = 'Prediction: '+ str(prediction))

if __name__ == '__main__':
    app.run(port=8000, debug= True)