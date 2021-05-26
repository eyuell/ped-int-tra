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
from os.path import isfile, exists
from os import makedirs
import pathlib
import sys

from keras.layers import Input, LSTM, Dense, Concatenate, Flatten
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.models import Sequential, Model, load_model

from prettytable import PrettyTable

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
        return True, dt
    else:
        return False, None


def get_performance(Ytest, test_results, title, axis_n):
    perf = {}

    performance = np.square(Ytest - test_results)

    perf['mse-5'] = performance[:, 0:5, :].mean(axis=None)
    perf['mse-10'] = performance[:, 0:10, :].mean(axis=None)
    perf['mse-15'] = performance[:, 0:15, :].mean(axis=None)
    perf['mse-15th'] = performance[:,15, :].mean(axis=None)
    perf['mse-30'] = performance[:, 0:30, :].mean(axis=None)
    perf['mse-45'] = performance[:, 0:45, :].mean(axis=None)
    perf['mse-last'] = performance[:, -1, :].mean(axis=None)

    print()
    t = PrettyTable(['MSE'])
    t.title = title
    t.align = "l"

    t.add_row(['mse-5    : ' + str(round(perf['mse-5'], 4)) ])
    t.add_row(['mse-10    : ' + str(round(perf['mse-10'], 4)) ])
    t.add_row(['mse-15    : ' + str(round(perf['mse-15'], 4)) ])
    t.add_row(['mse-15th  : ' + str(round(perf['mse-15th'], 4)) ])
    t.add_row(['mse-30    : ' + str(round(perf['mse-30'], 4)) ])
    t.add_row(['mse-45    : ' + str(round(perf['mse-45'], 4)) ])
    t.add_row(['mse-last  : ' + str(round(perf['mse-last'], 4)) ])

    print(t)


def get_center_perf (Ytest, test_results, X_test_Box_Center, Y_test_Box_Center, title, axis_n):
    perf = {}

    results_org = test_results + np.expand_dims(X_test_Box_Center[0][:, 0, 0:4], axis=1)

    #  Performance measures for centers
    res_centers = np.zeros(shape=(test_results.shape[0], test_results.shape[1], 2))
    centers = np.zeros(shape=(test_results.shape[0], test_results.shape[1], 2))
    for b in range(test_results.shape[0]):
        for s in range(test_results.shape[1]):
            centers[b, s, 0] = (Y_test_Box_Center[b, s, 2] + Y_test_Box_Center[b, s, 0]) / 2
            centers[b, s, 1] = (Y_test_Box_Center[b, s, 3] + Y_test_Box_Center[b, s, 1]) / 2
            res_centers[b, s, 0] = (results_org[b, s, 2] + results_org[b, s, 0]) / 2
            res_centers[b, s, 1] = (results_org[b, s, 3] + results_org[b, s, 1]) / 2

    c_performance = np.square(centers - res_centers)
    perf['c-mse-5'] = c_performance[:, 0:5, :].mean(axis=None)
    perf['c-mse-10'] = c_performance[:, 0:10, :].mean(axis=None)
    perf['c-mse-15'] = c_performance[:, 0:15, :].mean(axis=None)
    perf['c-mse-15th'] = c_performance[:,15, :].mean(axis=None)
    perf['c-mse-30'] = c_performance[:, 0:30, :].mean(axis=None)
    perf['c-mse-45'] = c_performance[:, 0:45, :].mean(axis=None)
    perf['c-mse-last'] = c_performance[:, -1, :].mean(axis=None)

    t = PrettyTable(['C-MSE'])
    t.title = title
    t.align = "l"

    t.add_row(['c-mse-5     : ' + str(round(perf['c-mse-5'], 4)) ])
    t.add_row(['c-mse-10    : ' + str(round(perf['c-mse-10'], 4)) ])
    t.add_row(['c-mse-15    : ' + str(round(perf['c-mse-15'], 4)) ])
    t.add_row(['c-mse-15th  : ' + str(round(perf['c-mse-15th'], 4)) ])
    t.add_row(['c-mse-30    : ' + str(round(perf['c-mse-30'], 4)) ])
    t.add_row(['c-mse-45    : ' + str(round(perf['c-mse-45'], 4)) ])
    t.add_row(['c-mse-last  : ' + str(round(perf['c-mse-last'], 4)) ])

    print("\n")
    print(t)
    print("\n")


def lstm_model (first_shape, second_shape, n_filr, model_name):

    input_layer = Input(shape=(first_shape[1], first_shape[2]), name='input_layer')

    lstm_layer = LSTM(n_filr, return_state=True, return_sequences=True, activation='relu', name='lstm_layer')(input_layer)

    lstm_states = lstm_layer[1:]

    output_dense = Dense(second_shape[1] * second_shape [2], activation='linear', name='output_dense')(lstm_states[0])

    model = Model(inputs=[input_layer], outputs=output_dense, name=model_name)

    #optimizer = RMSprop(lr=0.0001)
    optimizer = 'adam'

    model.compile(optimizer=optimizer, loss='mse', metrics=['accuracy'])

    return model


def shorten_dim(arr_list):
    m, n, o = arr_list.shape
    return arr_list.reshape(m, n * o)


def expand_dim(arr_list, seq):
    m, n = arr_list.shape
    return arr_list.reshape(m, seq, int(n/seq))


def train_bbox(n_batch, n_epochs, call_backs, n_filters, traject_model_file, data_path):
    # For B_Box Prediction
    print("\n=============================================")
    print("Train and Predict on Bounding Box-Trajectory:")
    print("=============================================")

    data_exists, Xtrain = get_data('Xtrain, Ytrain, and Val_data', 'tra_xTrain.pkl', data_path)   # Xtrain 14, 4, BB
    data_exists, Ytrain = get_data('', 'tra_yTrain.pkl', data_path)   # Ytrain 45, 4, BB
    data_exists, Val_data = get_data('', 'tra_valData.pkl', data_path)  # Val_data 14, 4, BB

    if data_exists:
        model = lstm_model(Xtrain[0].shape, Ytrain.shape, n_filters, 'LSTM-Traject-B Box')  #Only the b-box is taken, no intention, no speed
        print()
        model.summary()

        Y = shorten_dim(Ytrain)

        model.fit(x=Xtrain[0],
                    y=Y,
                    validation_data=[Val_data[0][0],shorten_dim(Val_data[1])],
                    batch_size=n_batch,
                    epochs=n_epochs,
                    #callbacks=call_backs,
                    verbose=1)

        # Save trained model
        model.save(traject_model_file)

        #del Xtrain, Ytrain, Val_data # clear memory

        return model
    else:
        print("\nMissing Data\n")
        return None


def get_callback():
    early_stop = EarlyStopping(monitor='val_loss',
                                min_delta=0.0001,
                                patience=10,
                                verbose=1)

    plateau_sch = ReduceLROnPlateau(monitor='val_loss',
                                    factor=0.5,
                                    patience=10,
                                    min_lr=0.0000001,
                                    verbose = 1)

    call_backs = [early_stop, plateau_sch]

    return call_backs


def predict_bbox(model, n_batch, data_path, data_set):
    data_exists, X_test = get_data('Bounding Box Related X and Y data', 'box_intent_speed_data.pkl', data_path)    # F_X_test_Box 14, 4 BB
    data_exists, Y_test = get_data('', 'box_target_speed_data.pkl', data_path)    # F_Y_test_Box 45, 4 BB

    data_exists, X_text_box = get_data('', 'box_centers_test_data.pkl', data_path)  # F_X_test_box_center, 14, 4 BB
    data_exists, Y_text_box = get_data('', 'box_centers_target_data.pkl', data_path)    # F_Y_test_box_center, 45, 4 BB

    if data_exists:
        pass
    else:
        data_exists, X_test = get_data('Bounding Box Related X and Y data', 'tra_test_obs_data.pkl', data_path)    # M_X_test_box 14, 4, BB Option
        data_exists, Y_test = get_data('', 'tra_test_target_data.pkl', data_path) # M_Y_test_box 45, 4, BB Option

        data_exists, X_text_box = get_data('', 'bbox_centers_test_data.pkl', data_path)   # M_X_test_Bbox_center 14, 4, BB Option
        data_exists, Y_text_box = get_data('', 'bbox_centers_target_data.pkl', data_path) # M_Y_test_Bbox_center 45, 4, BB Option

    if data_exists:
        test_results_B = model.predict(X_test[0], batch_size=n_batch, verbose=1)
        tes_res = expand_dim(test_results_B, Y_test.shape[1])

        # Performances on Bounding boxes)
        print("\nPerformances on Bounding Boxes-Trajectory for", data_set.capitalize())
        get_performance(Y_test, tes_res, 'LSTM Traject: B_Box', 1)
        get_center_perf(Y_test, tes_res, X_text_box, Y_text_box, 'LSTM Traject: B_Box-C', 1)
    else:
        print("\nMissing data for bounding Box")


def train_speed(n_batch, n_epochs, call_backs, n_filters, speed_model_file, data_path):
    # For Speed Prediction
    print("=======================================")
    print("Train and Predict on Speed-Trajectory:")
    print("=======================================")

    data_exists, Xtrain_Spe = get_data('Xtrain_Spe, Ytrain_Spe, and Val_data_Spe', 'spe_xTrain.pkl', data_path)   # Xtrain_Spe 14, 1, SP
    data_exists, Ytrain_Spe = get_data('', 'spe_yTrain.pkl', data_path)   # Ytrain_Spe 45, 1, SP
    data_exists, Val_data_Spe = get_data('', 'spe_valData.pkl', data_path)  # Val_data_Spe 14, 1, Sp

    if data_exists:
        model = lstm_model(Xtrain_Spe[0].shape, Ytrain_Spe.shape, n_filters, 'LSTM-Traject-Speed')
        print()
        model.summary()

        model.fit(x=Xtrain_Spe[0],
                    y=shorten_dim(Ytrain_Spe),
                    validation_data=[Val_data_Spe[0][0], shorten_dim(Val_data_Spe[1])],
                    batch_size=n_batch,
                    epochs=n_epochs,
                    callbacks=call_backs,
                    verbose=1)

        # Save trained model
        model.save(speed_model_file)

        #del Xtrain_Spe, Ytrain_Spe, Val_data_Spe # clear memory

        return model
    else:
        print("\nTraining data missing\n")
        return None


def predict_speed(speed_model, n_batch, data_path, data_set):
    data_exists, F_X_test_speed = get_data('Speed related X and Y data', 'speed_test_data.pkl', data_path)  # F_X_test_speed 14, 1, SP
    data_exists, F_Y_test_speed = get_data('', 'speed_target_data.pkl', data_path)    # F_Y_test_speed 45, 1, SP

    if data_exists:
        test_results = speed_model.predict(F_X_test_speed[0], batch_size=n_batch, verbose=1)
        tes_res = expand_dim(test_results, F_Y_test_speed.shape[1])

        # Performances on Speed
        print("\nPerformances on Speed-Trajectory for", data_set.capitalize())
        get_performance(F_Y_test_speed, tes_res, 'LSTM Traject: Speed', 0)
    else:
        print("Data missing for Prediction")


def main(data_set="pie"):
    expected_mix = ['pie', 'waymo', 'pie-waymo']

    if data_set not in expected_mix:
        data_set = "pie"
        print("\nWarning: unknown dataset supplied, pie is now allocated for now\n")

    lstm_path = str(pathlib.Path().absolute()) + '/data/for_lstm/'
    data_path = lstm_path + data_set + '/'
    traject_model_path = data_path + 'traject/'
    speed_model_path = data_path + 'speed/'

    start = time.time()

    check_path(lstm_path)
    check_path(data_path)
    check_path(traject_model_path)
    check_path(speed_model_path)

    n_batch = 64
    n_epochs = 100
    n_filters = 256
    call_backs = get_callback()

    train_speed_new = False
    train_box_new = False

    traject_best_model_file = traject_model_path + 'best.lstm_traject.h5'
    traject_model_file = traject_model_path + 'lstm_traject.h5'
    speed_best_model_file = speed_model_path + 'best.lstm_speed.h5'
    speed_model_file = speed_model_path + 'lstm_speed.h5'

    tra_exists = isfile(traject_best_model_file)
    spe_exists = isfile(speed_best_model_file)

    if tra_exists or spe_exists:
        print("Do you want to load a trained model ?")
        choice = 'a'
        while choice not in {'Y', 'N', 'y', 'n'}:
            print("Choose from (y/n) and press Enter", end=': ')
            choice = input()

        if choice in {'Y', 'y'}:
            if tra_exists:
                box_model = load_model(traject_best_model_file)
            else:
                train_box_new = True

            if spe_exists:
                speed_model = load_model(speed_best_model_file)
            else:
                train_speed_new = True
        else:
            train_speed_new = True
            train_box_new = True
    else:
        train_speed_new = True
        train_box_new = True

    if train_box_new:
        box_model = train_bbox(n_batch, n_epochs, call_backs, n_filters, traject_model_file, data_path)

    if box_model is not None:
        if not train_box_new:
            print()
            box_model.summary()
        predict_bbox(box_model, n_batch, data_path, data_set)

    """
    if train_speed_new:
        speed_model = train_speed(n_batch, n_epochs, call_backs, n_filters, speed_model_file, data_path)

    if speed_model is not None:
        if not train_speed_new:
            print()
            speed_model.summary()
        predict_speed(speed_model, n_batch, data_path, data_set)
    """
    end = time.time()
    elapsed = end - start
    t_del = timedelta(seconds=elapsed)
    print('\nElapsed time = {}'.format(t_del))


if __name__ == '__main__':
    err_msg = 'Usage: python LSTM_trajectory.py <dataset_name>\n'
    err_msg = err_msg + '         dataset_name: pie, waymo, or pie-waymo - in small case letters\n'
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