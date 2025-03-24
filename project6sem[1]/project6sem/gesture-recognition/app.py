from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np

# Cvlib,cv2
import cvlib as cv
import cv2

# Keras
#from keras.applications.imagenet_utils import preprocess_input, decode_predictions
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import load_model

# initialize mediapipe
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

# Load the gesture recognizer model
model = load_model('mp_hand_gesture')

# Load class names
f = open('gesture.names', 'r')
classNames = f.read().split('\n')
f.close()
print(classNames)

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

print('Model loaded. Start serving...')

# You can also use pretrained model from Keras
# Check https://keras.io/applications/
#from keras.applications.resnet50 import ResNet50
#model = ResNet50(weights='imagenet')
print('Model loaded. Check http://127.0.0.1:5000/')


def predictor(image_path,model):
    result = hands.process(image_path)
    className = ''

    if result.multi_hand_landmarks:
        landmarks = []
        for handslms in result.multi_hand_landmarks:
            for lm in handslms.landmark:
                # print(id, lm)
                lmx = int(lm.x * x)
                lmy = int(lm.y * y)

                landmarks.append([lmx, lmy])

            # Drawing landmarks on frames
            mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)

            # Predict gesture
            prediction = model.predict([landmarks])
            # print(prediction)
            classID = np.argmax(prediction)
            className = classNames[classID]

    # show the prediction on the frame
    cv2.putText(frame, className, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                   1, (0,0,255), 2, cv2.LINE_AA)

    # img= cv2.imread(image_path)
    # face_crop=img
    # face, confidence = cv.detect_face(img)
    # startX=startY=endX=endY=0
    # if face:
    #     f=face[0]
    #     print(f)

    #     (startX, startY) = f[0] if f[0]>0 else 0, f[1] if f[1]>0 else 0
    #     (endX, endY) = f[2] if f[2]>0 else 0, f[3] if f[3]>0 else 0

    #     print(startX,startY,endX,endY)

    #     cv2.rectangle(img, (startX,startY), (endX,endY), (0,255,0), 2)

    #     face_crop = np.copy(img[startY:endY,startX:endX])

    #     if face_crop.shape[0]==0 : 
    #         face_crop = img
    # else:
    #     face_crop=img

    # face_crop = cv2.resize(face_crop, (64,64))
    # face_crop = face_crop.astype("float") / 255.0
    # face_crop = image.img_to_array(face_crop)
    # face_crop = np.expand_dims(face_crop, axis=0)

    # confidence = model.predict(face_crop)[0]

    # # write predicted gender and confidence on image (top-left corner)
   
    # idx = np.argmax(confidence)
    # label = classes[idx]


    # label=l = "{}: {:.2f}%".format(label, confidence[idx] * 100)

    # if startX is None:
    #     Y = startY - 10 if startY - 10 > 10 else startY + 10
    # else:
    #     Y=0

    # cv2.putText(img, label, (startX, Y),  cv2.FONT_HERSHEY_SIMPLEX,
    #             0.7, (0, 255, 0), 2)



    # #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # cv2.imwrite(image_path,img)
    # return label

def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(64, 64))

    # Preprocessing the image
    x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    x = np.expand_dims(x, axis=0)

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    #x = preprocess_input(x, mode='caffe')

    preds = model.predict_classes(x)
    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['image']

        # Save the file to ./uploadss
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        # preds = predictor(file_path, model)
        preds = predictor(f, model)
        return preds

        # Process your result for human
        pred_class = preds.argmax(axis=-1)            # Simple argmax
        #pred_class = decode_predictions(preds, top=1)   # ImageNet Decode
        #result = str(pred_class[0][0][1])               # Convert to string
        #print(pred_class.item(0))
        #print(pred_class.item(1))
        #if(pred_class.item(0)==0):
        #   result='Male'
        #elif(pred_class.item(0)==1):
        #    result='Female'
        
        #return result
    return None


if __name__ == '__main__':
    # app.run(port=5002, debug=True)

    # Serve the app with gevent
    http_server = WSGIServer(('0.0.0.0', 5000), app)
    http_server.serve_forever()
