from flask import Flask, request, Response,redirect, url_for, flash, jsonify
import jsonpickle
import cv2
import json
from keras.preprocessing.image import  ImageDataGenerator ,load_img,img_to_array
import numpy as np
from keras.models import Model,load_model
from time import sleep
from keras.layers import Dense
from keras.preprocessing.image import  ImageDataGenerator ,load_img,img_to_array
import numpy as np
from keras_applications.resnet import ResNet50
from keras.models import Model,load_model

# Initialize the Flask application`
app = Flask(__name__)
# route http posts to this method
@app.route('/api/classification', methods=['POST','GET'])
def classification():
    """
    import keras.backend.tensorflow_backend as tb
    tb._SYMBOLIC_SCOPE.value = True
    """
    r = request
    if r.method == 'POST':
        nparr = np.fromstring(r.data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        resized_image =cv2.resize(img, (224,224),interpolation = cv2.INTER_AREA)
        model = load_model('Egyptian_Paper_Currency_Detector_Model.h5')
        print(model.summary())
        imgar = img_to_array(resized_image)
        imgpred = np.expand_dims(imgar, axis=0)
        pred = model.predict(imgpred)
        Classes = ['1_F', "1_B", '5_F', "5_B", '10_F', "10_B", '20_F', "20_B", '50_F', "50_B", '100_F', "100_B",'200_F', "200_B"]
        index = pred.argmax(axis=-1)[0]
        print([float(x) for x in pred[0]])
        print(Classes[index])
        resulte=Classes[index]
        if resulte=="1_F"or resulte=="1_B":
            response = {'message':'1'}
        if resulte == "5_F" or resulte == "5_B":
            response = {'message': '5'}
        if resulte == "10_F" or resulte == "10_B":
            response = {'message': '10'}
        if resulte == "20_F" or resulte == "20_B":
            response = {'message': '20'}
        if resulte == "50_F" or resulte == "50_B":
            response = {'message': '50'}
        if resulte == "100_F" or resulte == "100_B":
            response = {'message': '100'}
        if resulte == "200_F" or resulte == "200_B":
            response = {'message': '200'}
            # encode response using jsonpickle
        response_pickled = jsonpickle.encode(response)
        return Response(response=response_pickled, status=200, mimetype="application/json")
# start flask app
app.run(host="0.0.0.0", port=5000,debug=True,threaded=False)