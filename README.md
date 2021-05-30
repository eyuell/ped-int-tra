# Pedestrian Intention and Trajectory Prediction, Replication and Implementation of PIE Predict Models On Waymo Open Dataset

This repository implements the [PIE Model](https://github.com/aras62/PIEPredict) on some segments of the [Waymo Open Datasets](https://waymo.com/open). This repository consists of PIE codes that are customized for use on Waymo. In addition vanilla LSTM models (for Intention and Trajectory Predictions) are also included for comparison of performances.

### Table of contents
* [Environmental Setup](#en_setup)
* [Execute PIE](#exe_pie)
* [Save Data](#save_data)
* [Execute LSTM](#exe_lstm)
* [Execute On Waymo](#exe_waymo)
* [Clear database](#clr_db)
* [Corresponding author](#author)
* [License](#license)

<a name="en_setup"></a>
## Environmental Setup

Follow the steps on [PIE Model](https://github.com/aras62/PIEPredict) prior to using the documents provided here. The installation of PIEPredict shall include the [PIE dataset and annotations](http://data.nvision2.eecs.yorku.ca/PIE_dataset/). The implementation was successful Ubuntu 18.04 with Python 3.7, CUDA 10.1, cuDNN 8, tensorflow 2.2, and keras 2.2. Allocating a minimum of 3TB storage space and 32GB RAM is needed.

After the PIE Models and Dataset are installed, the documents here will be merged to the PIE. For the waymo part, six segments (videos) have been annotated to fit them to PIE model. 

The annotations are distributed in three folders for training, validating and testing named as `wod1tr`, `wod2va`, and `wod3te` respectively for each of the annotation types. The images from waymo open dataset shall also be distributed in similar folder names. 

The [Quick Start](https://github.com/waymo-research/waymo-open-dataset/blob/master/docs/quick_start.md) on [waymo-open-dataset](https://github.com/waymo-research/waymo-open-dataset) repository could be helpful in installing the waymo-open-dataset module.

If interested, you may use a [data and image extractor](https://github.com/eyuell/waymo-pie-annotation) from waymo open dataset, which is organized by the same author of this implementation. The list of videos matched with the annotations is presented in the [segments_list](segments_list.txt) file.

<a name="exe_pie"></a>
## Execute PIE

After all documents are ready, training and testing on PIE dataset is made by running `python train_test.py 1`, that initiate the Intention and Trajectory predictions by PIE. 

To make training only dataset, use `python train_test.py 0`, and `python train_test.py 2` for testing only.

<a name="save_data"></a>
## Save Data

In order to assure that the LSTM models get same data input as PIE, the PIE model is customized to save data. Since saving data while testing could lead to shortage of memory, the PIE model is customized to save training and testing data separately.

Numbers used outside [0, 1, and 2] will set the code to saving data mode. For numbers less than 0, traininig data, including the validating data, will be saved. Whereas using numbers greater than 2 will save the testing data.

For example to save training and validating data, use `python train_test.py -1`. And use `python train_test.py 3` to save the testing data.

<a name="exe_lstm"></a>
## Execute LSTM

After the data are saved for LSTM in the folder `for_lstm/pie`, the LSTM models can be executed on PIE Dataset. 
Intention prediction on PIE dataset uses the command `python LSTM_intention.py`. For Trajectory prediction, use `python LSTM_trajectory.py`. 

Each time the LSTM models train, the models are saved as `lstm_intent.h5` and `lstm_traject.h5` in the folders of intent and traject respectively. 

There is an option to save best performing model as `best_lstm_intent.h5` and `best_lstm_traject.h5` such that LSTM testings can also be made from a trained model. 

You may opt to backup the models as they could be replaced, especially the `lstm_intent.h5` and `lstm_traject.h5`.


<a name="exe_waymo"></a>
## Execute On Waymo

All the above commands are working on PIE dataset, to use the Waymo Open Dataset (WOD), just add ` waymo` after each of the above command lines.

For example, to train and test on WOD using PIE models, run `python train_test.py 1 waymo`.

To save training and validation data for lstm, `python train_test.py -1 waymo` and for testing data, use `python train_test.py 3 waymo`, and so on.

To give example for LSTM on WOD, `python LSTM_intention.py waymo` and `python LSTM_trajectory.py waymo`

<a name="clr_db"></a>
## Clear database

In order to generate new database from supplied dataset, removing an existing database is necessary. To remove the database, navigate to `/PIE/data_cache` and remove using command `rm pie_database.pkl`. Thereafter, executing the PIE model will generate new database as per supplied dataset.

<a name="author"></a>
## Corresponding author

* **[Eyuell G](https://www.linkedin.com/in/eyuell/)**

<a name="license"></a>
## License
This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.
