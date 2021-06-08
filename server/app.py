#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#Importing libraries
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import accuracy
from trainmodel import get_wav, to_mfcc, create_segmented_mfccs, segment_one
from tensorflow import keras
from gcp_predict import final_prediction

app = Flask(__name__)
model = keras.models.load_model('./baseline_english.h5.h5')

y_predicted = 0

@app.route('/')
def home():
    return render_template('index.html')

american_filename = 'english33'
british_filename = 'english38'
australian_filename = 'english77'
option_dict = {"american":american_filename,'british':british_filename,'australian':australian_filename}
keys_list = list(option_dict)

@app.route('/predict', methods=["POST"])
def predict():
    selection = request.form['accent']
    wav_file = get_wav(option_dict[selection])
    print(wav_file)
    mfcc = to_mfcc(wav_file)
    y_predicted = accuracy.predict_class_audio(segment_one(mfcc), model)
    return render_template('index.html', prediction_text='Accent: {}'.format(keys_list[y_predicted-1]))

@app.route('/cloud-predict', methods=['POST'])
def cloud_predict():
    selection = request.form['accent']
    wav_file = get_wav(option_dict[selection])
    mfcc = to_mfcc(wav_file)
    segmented_mfcc = segment_one(mfcc)
    MFCCs = segmented_mfcc.reshape(segmented_mfcc.shape[0],segmented_mfcc.shape[1],segmented_mfcc.shape[2],1)
    y_predicted = final_prediction(MFCCs=MFCCs)
    return render_template('index.html', prediction_text='Accent: {}'.format(keys_list[y_predicted[0]-1]))

@app.route('/predict-recording', methods=['GET','POST'])
def predict_recording():
    if request.method == 'POST':
        f = request.files['audio_data']
        with open('./static/audio.wav', 'wb') as audio:
            f.save(audio)
        print('file uploaded successfully')
        wav_file = get_wav('audio')
        mfcc = to_mfcc(wav_file)
        y_predicted = accuracy.predict_class_audio(segment_one(mfcc), model)
        print(y_predicted)
        return render_template('index.html', prediction_text='Accent: {}'.format(keys_list[y_predicted-1]))
    else:
        return render_template('index.html', prediction_text='Accent: {}'.format(keys_list[y_predicted-1]))



@app.route('/cloud-predict-recording', methods=['POST'])
def cloud_predict_recording():
    f = request.files['audio_data']
    with open('./static/audio.wav', 'wb') as audio:
        f.save(audio)
    print('file uploaded successfully')
    wav_file = get_wav('audio')
    mfcc = to_mfcc(wav_file)
    print(mfcc)
    segmented_mfcc = segment_one(mfcc)
    MFCCs = segmented_mfcc.reshape(segmented_mfcc.shape[0],segmented_mfcc.shape[1],segmented_mfcc.shape[2],1)
    print(MFCCs)
    y_predicted = final_prediction(MFCCs=MFCCs)
    return render_template('index.html', prediction_text='Accent: {}'.format(keys_list[y_predicted[0]-1]))


@app.route("/test", methods=['POST', 'GET'])
def test():
    if request.method == "POST":
        f = request.files['audio_data']
        print(f)
        with open('./static/audio.wav', 'wb') as audio:
            f.save(audio)
        print('file uploaded successfully')

        return render_template('test.html', request="POST")
    else:
        return render_template("test.html")



if __name__ == "__main__":
    app.run(debug=True)