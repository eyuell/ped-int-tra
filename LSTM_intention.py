"""
Copyright 2021, Eyuell H Gebremedhin

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
"""
Inspiration used from A. Rasouli, I. Kotseruba, T. Kunic, and J. Tsotsos, "PIE: A Large-Scale Dataset and Models for Pedestrian Intention Estimation and
Trajectory Prediction", ICCV 2019.
"""
import numpy as np
import os
import pickle
import pathlib
import sys

from os.path import isfile, exists
from os import listdir, makedirs

from keras.layers import Input, LSTM, Dense, Reshape, Concatenate
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.models import Sequential, Model, load_model

from prettytable import PrettyTable

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score, fbeta_score, recall_score

import time
from datetime import timedelta


def check_path(f_path):
    if not exists(f_path):
        makedirs(f_path)


def get_data(data_name, file_name, data_path):
    if data_name:
        print("\nWorking on", data_name)
    path_line = os.path.join(data_path, file_name)

    if os.path.exists(path_line):
        with open(path_line, 'rb') as fid:
            try:
                dt = pickle.load(fid)
            except:
                dt = pickle.load(fid, encoding='bytes')
        return dt
    else:
        return None


def vanila_LSTM(img_shape, box_shape, n_fil):
    images_input = Input(shape=(img_shape[1], img_shape[2], img_shape[3], img_shape[4]), name = "images_input")

    reshape_input = Reshape((img_shape[1], img_shape[2] * img_shape[3] * img_shape[4]),
                            input_shape=(img_shape[1], img_shape[2], img_shape[3], img_shape[4]))(images_input)

    boxes_input = Input(shape=(box_shape[1], box_shape[2]), name='boxes_input')

    lstm_inputs = Concatenate(axis=2)([reshape_input, boxes_input])

    lstm_output = LSTM(n_fil, activation='relu', name='lstm-layer')(lstm_inputs)

    lstm_dense_output = Dense(n_fil, activation='relu', name='lstm-dense')(lstm_output)

    dense_output = Dense(1, activation='sigmoid', name='output-dense')(lstm_dense_output)

    lstm_model = Model(inputs=[images_input, boxes_input], outputs=dense_output, name='LSTM-Intent-Model')

    lstm_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    lstm_model.summary()

    return lstm_model


# Fetch Dataa and Train
def train_as_new(n_batch, n_epochs, n_filters, model_file, data_path):
    Xtrain = get_data('Xtrain, Ytrain, and Val_data', 'int_xTrain.pkl', data_path)
    Ytrain = get_data('', 'int_yTrain.pkl', data_path)
    Val_data = get_data('', 'int_valData.pkl', data_path)

    if Xtrain is not None and Ytrain is not None and Val_data is not None:
        model = vanila_LSTM(Xtrain[0].shape, Xtrain[1].shape, n_filters)

        early_stop = EarlyStopping(monitor='val_accuracy',
                                    min_delta=0.0001,
                                    patience=10,
                                    verbose=1)

        plateau_sch = ReduceLROnPlateau(monitor='val_accuracy',
                                        factor=0.2,
                                        patience=10,
                                        min_lr=0.0000001,
                                        verbose = 1)

        call_backs = [early_stop, plateau_sch]

        # Train Model
        model.fit(x=Xtrain,
                y=Ytrain,
                batch_size=n_batch,
                validation_data=Val_data,
                epochs=n_epochs,
                #callbacks=call_backs,
                verbose=1)

        # Save trained model
        model.save(model_file)

        #del Xtrain, Ytrain, Val_data #clear memory

        return model
    else:
        return None


# Fetch data and Predict
def predict_intention(model, n_batch, data_path, dataset):

    Xtest = []
    Ytest = []

    test_results = np.array([])

    x_key = 'int_test_data_'
    y_key = 'int_target_data_'

    files_list = listdir (data_path)
    needed_f = list(filter(lambda n: x_key in n, files_list))
    needed_f.extend(list(filter(lambda p: y_key in p, files_list)))

    for k in range(len(needed_f)):
        batch_x = 'Xtest_' + str(k + 1)
        batch_y = 'Ytest_' + str(k + 1)
        x_file = x_key + str(k + 1) + '.pkl'
        y_file = y_key + str(k + 1) + '.pkl'

        if x_file in needed_f and y_file in needed_f:
            key = batch_x + ' and ' + batch_y
            Xtest.extend(get_data(key, x_file, data_path))
            Ytest.extend(get_data('', y_file, data_path))

            if Xtest is not None and Ytest is not None:
                tmp_res = model.predict(Xtest, batch_size=n_batch, verbose=1)
                Xtest = []
                test_results = np.append(test_results, tmp_res)

    size1 = len(np.array(Ytest))
    size2 = len(np.round(test_results))

    if size1 > 0 and size2 > 0 and size1 == size2:
        acc = round( accuracy_score(np.array(Ytest), np.round(test_results)), 4)
        f1 = round( f1_score(np.array(Ytest), np.round(test_results)), 4)
        f2 = round( fbeta_score(np.array(Ytest), np.array(np.round(test_results)), beta=2), 4)
        recall = round( recall_score (np.array(Ytest), np.round(test_results)), 4)

        print("\nPerformance:")
        t = PrettyTable(['Acc', 'F1', 'F2', 'Recall *'])
        tt = 'LSTM Intention, ' + dataset.capitalize() + ':'
        t.title = tt
        t.add_row([acc, f1, f2, recall])
        print(t)
        print("* Recall = True Positive Rate, Sensitivity, Power, or Detection Probability")

    else:
        print("\n\nMissing data for testing\n")


def get_model(n_batch, n_epochs, n_filters, best_model_file, model_file, data_path):
    train_new = False
    model = None

    if isfile(best_model_file):
        print("Do you want to load a trained model?")
        choice = 'a'
        while choice not in {'Y', 'N', 'y', 'n'}:
            print("Choose from (y/n) and press Enter", end=': ')
            choice = input()

        if choice in {'Y', 'y'}:
            model = load_model(best_model_file)
            model.summary()
        else:
            train_new = True
    else:
        train_new = True

    if train_new:
        model = train_as_new(n_batch, n_epochs, n_filters, model_file, data_path)

    return model


def main(data_set="pie"):
    expected_mix = ['pie', 'waymo', 'pie-waymo']

    if data_set not in expected_mix:
        data_set = "pie"
        print("\nWarning: unknown dataset supplied, pie is now allocated\n")
    
    lstm_path = str(pathlib.Path().absolute()) + '/data/for_lstm/'
    data_path = lstm_path + data_set + '/'
    model_path = data_path + 'intent/'

    start = time.time()

    n_batch = 128
    n_epochs = 100
    n_filters = 256

    check_path(lstm_path)
    check_path(data_path)
    check_path(model_path)

    best_model_file = model_path + 'best.lstm_intent.h5'
    model_file = model_path + 'lstm_intent.h5'

    model = get_model(n_batch, n_epochs, n_filters, best_model_file, model_file, data_path)

    if model is not None:
        predict_intention(model, n_batch, data_path, data_set)
    else:
        print("\nMissing data input/s")

    end = time.time()
    elapsed = end - start
    t_del = timedelta(seconds=elapsed)
    print('\nElapsed time = {}'.format(t_del))


if __name__ == '__main__':
    err_msg = 'Usage: python LSTM_intention.py <dataset_name>\n'
    err_msg = err_msg + '         dataset_name: pie, waymo, pie-waymo - in small case letters\n'
    try:
        raw_input = sys.argv
        if len(raw_input) < 2:
            print("\nWarning:",err_msg)
            dataset_name='no value'
        else:
            dataset_name = sys.argv[1]
        main(data_set=dataset_name)
    except ValueError:
        raise ValueError(err_msg)