#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#Importing libraries
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import accuracy
from trainmodel import get_wav, to_mfcc, create_segmented_mfccs, segment_one
from tensorflow import keras

app = Flask(__name__)
model = keras.models.load_model('./baseline_english.h5.h5')


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
    mfcc = to_mfcc(wav_file)
    y_predicted = accuracy.predict_class_audio(segment_one(mfcc), model)
    return render_template('index.html', prediction_text='Accent: {}'.format(keys_list[y_predicted-1]))


if __name__ == "__main__":
    app.run(debug=True)