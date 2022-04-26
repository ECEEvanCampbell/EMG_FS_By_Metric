import numpy as np
import os
from scipy import signal
import math
import matplotlib.pyplot as plt
import pickle

from Feature_Extractor import Feature_Extractor
from Feature_Selector import Feature_Selector


class Dataset:
    def __init__(self, window_size, window_inc, frequency):
        self.window_size = window_size
        self.window_inc = window_inc
        self.frequency = frequency

        self.data = []

    def import_dataset(self, motion_type, base_dir="Data/Raw_Data"):
        # by default it will look for subject folders in Data/Raw_Data
        if os.path.exists("Data/"+motion_type + "-"+str(self.window_size)+"-"+str(self.window_inc) + ".npy"):
            self.data = np.load("Data/"+motion_type + "-"+str(self.window_size) +
                                "-"+str(self.window_inc) + ".npy", allow_pickle=True)
            return
        subject_folders = os.listdir(base_dir)
        for subject_folder in subject_folders:
            data_files = os.listdir(base_dir + "/" + subject_folder)
            for data_file in data_files:

                file_parts = data_file.split(".")[0].split("_")
                motion_type_identifier = file_parts[1]
                if motion_type_identifier == motion_type:

                    subject_number = int(file_parts[0][1:])-1
                    class_number = int(file_parts[2][1:])-1
                    rep_number = int(file_parts[3][1:])-1
                    raw_data = np.genfromtxt(
                        base_dir + "/" + subject_folder + "/" + data_file, delimiter=",")

                    b, a = signal.iirnotch(60 / (self.frequency/2), 20)
                    notched_data = signal.lfilter(b, a, raw_data, axis=0)
                    b, a = signal.butter(
                        N=4, Wn=[20/(self.frequency/2), 450/(self.frequency/2)], btype="band")
                    filtered_data = signal.lfilter(b, a, notched_data, axis=0)

                    num_windows = math.floor(
                        (filtered_data.shape[0]-self.window_size)/self.window_inc)

                    st = 0
                    ed = st+self.window_size

                    for w in range(num_windows):
                        self.data.append([subject_number,
                                          class_number,
                                          rep_number,
                                          filtered_data[st:ed, :].transpose()])
                        st = st+self.window_inc
                        ed = ed+self.window_inc
        np.save("Data/"+motion_type + "-"+str(self.window_size) +
                "-"+str(self.window_inc) + ".npy", self.data)

    def extract_features(self, feature_list):
        self.num_channels = self.data[0][3].shape[0]
        self.feature_extractor = Feature_Extractor(self.num_channels)
        if feature_list == ['all']:
            feature_list = self.feature_extractor.get_feature_list()
        self.feature_list = feature_list
        windows = np.asarray([i[3] for i in self.data])

        self.features = self.feature_extractor.extract(
            self.feature_list, windows)
        self.features["feature_list"] = self.feature_list

    def save_prepared_data(self, file_location):
        self.features["subject"] = np.array([i[0] for i in self.data])
        self.features["class"] = np.array([i[1] for i in self.data])
        self.features["rep"] = np.array([i[2] for i in self.data])
        with open(file_location, 'wb') as f:
            pickle.dump(self.features, f)


def main(args=None):
    analysis_parameters = {
        "motion_type": "Ramp",
        "metric": ["accuracy", "argmax"],
        "feature_list": ['all'],
        "window_size": 200,
        "window_inc": 100,
        "frequency": 1000
    }
    prepared_data_location = "Data/prepared_dataset-"+analysis_parameters["motion_type"] + \
        "-"+str(analysis_parameters["window_size"]) + \
        "-"+str(analysis_parameters["window_inc"]) + ".pk"

    if not os.path.exists(prepared_data_location):

        # construct dataset object
        dataset = Dataset(analysis_parameters["window_size"],
                          analysis_parameters["window_inc"],
                          analysis_parameters["frequency"])
        # import data from csv files
        dataset.import_dataset(analysis_parameters["motion_type"])
        # extract_features
        dataset.extract_features(analysis_parameters['feature_list'])
        # save the prepared features and metadata
        dataset.save_prepared_data(prepared_data_location)

    with open(prepared_data_location, 'rb') as f:
        prepared_data = pickle.load(f)


    # choose metric to use as selection criterion (filter function / wrapper)
    feature_selector = Feature_Selector(analysis_parameters["metric"], len(prepared_data["feature_list"]))

    # determine optimal feature set according to criterion
    feature_selector.run_selection(prepared_data)
    feature_selector.print_results()




if __name__ == "__main__":
    main()
