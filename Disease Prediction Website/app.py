from flask import Flask, render_template, request, flash, redirect
from werkzeug.utils import secure_filename
import pickle
import os
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import tensorflow_hub as hub

app = Flask(__name__)


def predict(values, dic):
    if len(values) == 8:
        model = pickle.load(open('models/diabetes.pkl','rb'))
        values = np.asarray(values)
        return ('diabetes',model.predict(values.reshape(1, -1))[0])
    elif len(values) == 26:
        model = pickle.load(open('models/cancer.pkl','rb'))
        values = np.asarray(values)
        return ('breast cancer',model.predict(values.reshape(1, -1))[0])
    elif len(values) == 13:
        model = pickle.load(open('models/heart.pkl','rb'))
        values = np.asarray(values)
        return ('Heart',model.predict(values.reshape(1, -1))[0])
    elif len(values) == 18:
        model = pickle.load(open('models/kidney.pkl','rb'))
        values = np.asarray(values)
        return ('Kidney',model.predict(values.reshape(1, -1))[0])
    else:
        return  'Invalid input'

model = load_model(('models/brain_tumor.h5'), custom_objects={'KerasLayer': hub.KerasLayer})    


def model_predict(img_path, model):
    test_image = image.load_img(img_path, target_size = (224,224))
    test_image = image.img_to_array(test_image)
    test_image = test_image / 255
    test_image = np.expand_dims(test_image, axis=0)
    result = model.predict(test_image)
 
    if result <= 0.5:
        return "The Person has no Brain Tumor"
    else:
        return "The Person has Brain Tumor"



    
@app.route("/")
def home():
    return render_template('home.html')

@app.route("/aboutus")
def aboutus():
    return render_template('aboutus.html')

@app.route("/diabetes", methods=['GET', 'POST'])
def diabetesPage():
    return render_template('diabetes.html')

@app.route("/cancer", methods=['GET', 'POST'])
def cancerPage():
    return render_template('breast_cancer.html')

@app.route("/heart", methods=['GET', 'POST'])
def heartPage():
    return render_template('heart.html')

@app.route("/kidney", methods=['GET', 'POST'])
def chronicPage():
    return render_template('Kidney.html')

@app.route("/brain_tumor", methods=['GET', 'POST'])
def tumorPage():
    return render_template('brain_tumor.html')


@app.route("/predict", methods = ['POST', 'GET'])
def predictPage():
    if request.method == 'POST':
        to_predict_dict = request.form.to_dict()
        to_predict_list = list(map(float, list(to_predict_dict.values())))
        name,pred = predict(to_predict_list, to_predict_dict)
        return render_template("predict.html",pred=pred,name = name)

@app.route("/model_predict", methods = ['POST', 'GET'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['image']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'upload', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        result = model_predict(file_path, model)
        return render_template("predict.html",tumor=1,result=result,pred=0)

if __name__ == '__main__':
	app.run(debug = True,port=5050)
     