# Python Standard Libraries
import os
import typing
import time
import logging
from enum import Enum
from datetime import datetime

# ESA Snappy
# from snappy import ProductIO -- lazy loaded

# Scientific Libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV
from sklearn import metrics
import joblib
import seaborn

# Self Defined Modules
from ground_truth import RosebelPixelClass3, GroundTruthBoundaries
import utility as util

def get_np_from_band(band):
    width = band.getRasterWidth()
    height = band.getRasterHeight()
    data = np.zeros(width * height, np.float32)
    band.readPixels(0, 0, width, height, data)
    data.shape = height, width  # (Number of rows, Number of Columns)
    return data


def get_all_product_filepaths(parent_path):
    paths = []
    for filename in os.listdir(parent_path):
        if filename.endswith(".dim"):
            paths.append(parent_path + "\\" + filename)
    return paths


def normalize_np_for_grayscale(np_arr):
    print("original")
    print(np_arr)
    max = np.amax(np_arr)
    min = np.amin(np_arr)
    range = max - min
    scaled = (np_arr/range) * 255.0
    print("scaled")
    print(scaled)
    return scaled


def extract_feature_array_from_product(file_path: str) -> (np.ndarray, int, int, int, list):
    """
    Extract bands from an ESA data product
    :param file_path: path to a ESA SNAP dim file
    :return: (numpy array of shape (bands, pixels), image width, image height, number of pixels)
    """
    print('Extracting feature array from product', file_path)
    util.start_timer()
    from snappy import ProductIO
    p = ProductIO.readProduct(file_path)
    bands = [p.getBand(x) for x in p.getBandNames()]

    if len(bands) == 0:
        raise Exception("No bands found in product")
    image_width = bands[0].getRasterWidth()
    image_height = bands[0].getRasterHeight()
    number_of_pixels = image_width*image_height
    feature_array = np.array([band.readPixels(0, 0, image_width, image_height, np.zeros(number_of_pixels, np.float32))
                              for band in bands])
    band_names = [band.getName() for band in bands]
    util.end_timer_print_duration()
    return feature_array, image_width, image_height, number_of_pixels, band_names


def generate_filtered_data_frame_with_pixel_index_and_labels(number_of_pixels: int, feature_array: np.ndarray, label_array: np.ndarray) -> np.ndarray:
    """
    Adds a pixel index row, and a label row, before transposing to a data-frame. Then filters unlabelled pixels.
    \n(WARNING: PIXEL INDEX MUST BE REMOVED BEFORE USING DATA TO FIT MODEL)
    :param number_of_pixels:
    :param feature_array:
    :param label_array:
    :return: data_frame of shape (number of pixels, number of features + 2), 1 extra column for pixel index, 1 for label
    """
    print('Generating data frame with pixel index and labels')
    util.start_timer()
    data_frame = np.insert(feature_array, 0, [x for x in range(number_of_pixels)], axis=0) # Insert Index Row
    data_frame = np.append(data_frame, [label_array], axis=0)
    data_frame = np.transpose(data_frame)
    data_frame = np.array(list(filter(lambda row: row[-1] != RosebelPixelClass3.na.value, data_frame)))
    util.end_timer_print_duration()
    return data_frame


def balance_data_frame(data_frame: np.ndarray) -> np.ndarray:
    """
    Balances the given data frame to the class label with the least occurrence; Label must be at last column.
    :param data_frame: must be pre-filtered of NA labels
    :return: data frame with equal rows of each class; shuffled.
    """
    print("Balancing data frame")
    util.start_timer()
    labels = np.transpose(data_frame)[-1]
    unique_labels, unique_label_counts = np.unique(labels, return_counts=True)
    min_count = min(unique_label_counts)
    # min_count = 3038

    balanced_labelled_data = []
    for idx, unique_label in enumerate(unique_labels):
        unique_label_data = np.array([x for x in filter(lambda row: row[-1] == unique_label, data_frame)])

        # Random re-balance if there is excess data of this label
        if unique_label_counts[idx] != min_count:
            np.random.shuffle(unique_label_data)  # in-place shuffle
            unique_label_data = np.delete(unique_label_data, [x for x in range(unique_label_counts[idx]-min_count)], axis=0)

        balanced_labelled_data.extend(unique_label_data)

    # Shuffle data frame before returning
    balanced_labelled_data = np.array(balanced_labelled_data)
    np.random.shuffle(balanced_labelled_data)

    util.end_timer_print_duration()
    return balanced_labelled_data


# Fit scaler to x_train, return transformed copies of x_train, x_test with index column still intact
def scale_but_ignore_index_column(x_train, x_test, scaler):
    # Do not transform the input; Transform a copy and return the copy to avoid side effects.

    sc = scaler()

    x_train_indexes = np.copy(np.transpose(x_train)[0])
    x_test_indexes = np.copy(np.transpose(x_test)[0])

    # np.delete returns a copy
    x_train_feature_values = np.delete(x_train, 0, axis=1)
    x_test_feature_values = np.delete(x_test, 0, axis=1)

    # fit on train set
    sc.fit(x_train_feature_values)

    # transform both train, test sets
    x_train_feature_values = sc.transform(x_train_feature_values)
    x_test_feature_values = sc.transform(x_test_feature_values)

    x_train_scaled = np.transpose(np.insert(np.transpose(x_train_feature_values), 0, x_train_indexes, axis=0))
    x_test_scaled = np.transpose(np.insert(np.transpose(x_test_feature_values), 0, x_test_indexes, axis=0))

    return x_train_scaled, x_test_scaled, sc


# DOES NOT BALANCE INPUT
def experiment_original_single_model_lr(result_folder, gt, image_width, enum, data_frame, error_discover=False,
                                        save_directory=None):

    print("Single Model LR Experiment")

    # Generate train, test sets
    print("Generating train-test split")
    util.start_timer()

    data_frame_labels = np.transpose(data_frame)[-1]
    data_frame_features_with_pixel_index = np.delete(data_frame, -1, axis=1)

    x_train, x_test, y_train, y_test = train_test_split(
        data_frame_features_with_pixel_index, data_frame_labels, test_size=0.33)
    util.end_timer_print_duration()

    # SCALING
    x_train_scaled, x_test_scaled, fitted_scaler = scale_but_ignore_index_column(x_train, x_test, StandardScaler)
    # x_train_scaled = x_train
    # x_test_scaled = x_test
    x_train_scaled_no_index = np.delete(x_train_scaled, 0, axis=1)
    x_test_scaled_no_index = np.delete(x_test_scaled, 0, axis=1)

    # Fit one LR model
    print("Fitting Logistic Regression Model with Training Data")
    util.start_timer()
    lr_model = LogisticRegression(multi_class='ovr', solver='liblinear')
    lr_model.fit(x_train_scaled_no_index, y_train)
    util.end_timer_print_duration()

    # Evaluate model
    print("Evaluating Logistic Regression Model with Test Data")
    util.start_timer()
    accuracy_score = lr_model.score(x_test_scaled_no_index, y_test)
    test_predictions = lr_model.predict(x_test_scaled_no_index)
    confusion_matrix = metrics.confusion_matrix(y_test, test_predictions)
    util.end_timer_print_duration()

    # Summarize Results
    print("\n@@@@@@@@@@@@@@@@@@@@@@@@\n" + "CLASSIFICATION SUMMARY" + "\n@@@@@@@@@@@@@@@@@@@@@@@@\n")
    util.print_train_test_pixel_summary(y_train, y_test, RosebelPixelClass3)
    print("Accuracy:\n" + str(accuracy_score))
    print("\nConfusion Matrix:\n", confusion_matrix)
    util.print_translate_confusion_matrix(confusion_matrix, RosebelPixelClass3, lambda x: x + 1)

    # Error Discovery
    if error_discover:
        print("\n@@@@@@@@@@@@@@@@@@@@@@@@\n", "ERROR DISCOVERY", "\n@@@@@@@@@@@@@@@@@@@@@@@@\n")
        util.print_error_discovery(confusion_matrix, enum, lr_model, x_test_scaled, y_test, gt, image_width)

    # Model Persistence TODO pipeline the scaler and predictor into a single estimator
    if save_directory is not None:
        joblib.dump(lr_model, save_directory + "/lr_model.joblib")
        joblib.dump(fitted_scaler, save_directory + "/std_scaler.joblib")


# DOES NOT BALANCE INPUT
def experiment_original_multi_model_lr(data_frame):
    # Multi-model Methodology
    print("\n@@@@@@@@@@@@@@@@@@@@@@@@\n", "MULTI-MODEL CLASSIFICATION", "\n@@@@@@@@@@@@@@@@@@@@@@@@\n")

    # Generate train, test sets
    print("Generating train-test split")
    util.start_timer()
    data_frame_labels = np.transpose(data_frame)[-1]
    data_frame_features_with_pixel_index = np.delete(data_frame, -1, axis=1)

    x_train, x_test, y_train, y_test = train_test_split(
        data_frame_features_with_pixel_index, data_frame_labels, test_size=0.33)
    util.end_timer_print_duration()

    x_train_scaled, x_test_scaled, fitted_scaler = scale_but_ignore_index_column(x_train, x_test, StandardScaler)

    print("Starting multi-model training")
    util.start_timer()

    # Forest classifier
    x_train_forest_clf = np.copy(x_train_scaled)
    y_train_forest_clf = np.copy(y_train)

    # Relabel
    for idx, original_label in enumerate(y_train_forest_clf):
        if original_label != RosebelPixelClass3.forest.value:
            # The value used is arbitrary as long as its consistent -- use lower class value for the complement
            y_train_forest_clf[idx] = RosebelPixelClass3.na.value

    # Re-balance
    forest_clf_frame = np.transpose(np.append(np.transpose(x_train_forest_clf), np.array([y_train_forest_clf]), axis=0))
    balanced_data_frame = balance_data_frame(forest_clf_frame)
    x_train_forest_clf = np.delete(balanced_data_frame, -1, axis=1)
    x_train_forest_clf_no_pixel_index = np.delete(x_train_forest_clf, 0, axis=1)
    y_train_forest_clf = np.transpose(balanced_data_frame)[-1]

    # Fit forest classifier
    forest_lr_classifier = LogisticRegression(multi_class='ovr', solver='liblinear')
    forest_lr_classifier.fit(x_train_forest_clf_no_pixel_index, y_train_forest_clf)

    # Water classifier
    x_train_water_clf = np.copy(x_train_scaled)
    y_train_water_clf = np.copy(y_train)

    #   relabel
    for idx, original_label in enumerate(y_train_water_clf):
        if original_label != RosebelPixelClass3.water.value:
            # The value used is arbitrary as long as its consistent -- use lower class number for the complement
            y_train_water_clf[idx] = RosebelPixelClass3.na.value

    # Re-balance
    water_clf_frame = np.transpose(np.append(np.transpose(x_train_water_clf), np.array([y_train_water_clf]), axis=0))
    balanced_data_frame = balance_data_frame(water_clf_frame)
    x_train_water_clf = np.delete(balanced_data_frame, -1, axis=1)
    x_train_water_clf_no_pixel_index = np.delete(x_train_water_clf, 0, axis=1)
    y_train_water_clf = np.transpose(balanced_data_frame)[-1]

    # Fit water classifier
    water_lr_classifier = LogisticRegression(multi_class='ovr', solver='liblinear')
    water_lr_classifier.fit(x_train_water_clf_no_pixel_index, y_train_water_clf)

    util.end_timer_print_duration()

    # Use both classifiers to generate a prediction array
    combined_predictor_result = []  # use normal list first

    x_test_no_pixel_index = np.delete(x_test_scaled, 0, axis=1)

    for test_pixel in x_test_no_pixel_index:
        forest_predictor_result = forest_lr_classifier.predict([test_pixel])[0]
        water_predictor_result = water_lr_classifier.predict([test_pixel])[0]

        # Neither forest or water
        if (forest_predictor_result != RosebelPixelClass3.forest.value) and (
                water_predictor_result != RosebelPixelClass3.water.value):
            combined_predictor_result.append(RosebelPixelClass3.mines.value)
        # Forest and not water
        elif (forest_predictor_result == RosebelPixelClass3.forest.value) and (
                water_predictor_result != RosebelPixelClass3.water.value):
            combined_predictor_result.append(RosebelPixelClass3.forest.value)
        # Water and not forest
        elif (water_predictor_result == RosebelPixelClass3.water.value) and (
                forest_predictor_result != RosebelPixelClass3.forest.value):
            combined_predictor_result.append(RosebelPixelClass3.water.value)
        # Conflict -- forest predictor says forest, water predictor says water, choose 1 and we'll be right 50% of the time
        else:
            combined_predictor_result.append(RosebelPixelClass3.water.value)

    # Evaluate our results
    accuracy_score = metrics.accuracy_score(y_test, np.array(combined_predictor_result))
    confusion_matrix = metrics.confusion_matrix(y_test, combined_predictor_result)

    print("Accuracy:\n", accuracy_score)
    print("\nConfusion Matrix:\n", confusion_matrix)

    for row_index in range(confusion_matrix.shape[0]):
        #  Print class prediction accuracy first
        value = confusion_matrix[row_index][row_index]
        print('\n' + RosebelPixelClass3(row_index + 1).name,  # Account for na class in enum
              'pixel prediction accuracy: %.2f%%' % (value * 100 / sum(confusion_matrix[row_index])))

        #  Print out incorrectness
        for col_index in range(confusion_matrix.shape[1]):
            if row_index != col_index:
                value = confusion_matrix[row_index][col_index]
                print(RosebelPixelClass3(row_index + 1).name, 'pixels mis-predicted as',
                      RosebelPixelClass3(col_index + 1).name + ': %.2f%%' % (
                                  value * 100 / sum(confusion_matrix[row_index])))


# DOES NOT BALANCE INPUT -  RFE with Cross validation
def experiment_cross_validated_recursive_feature_elimination(data_frame, band_names, results_folder):
    print("Experiment Started: RFECV")
    util.start_timer()

    label_array = np.transpose(data_frame)[-1]
    feature_data = np.delete(data_frame, 0, axis=1)
    feature_data = np.delete(feature_data, -1, axis=1)

    sc = StandardScaler()
    feature_data = sc.fit_transform(feature_data)

    X = feature_data
    Y = label_array

    lr_model = LogisticRegression(multi_class='ovr', solver='liblinear')

    rfecv = RFECV(lr_model, step=1, cv=3)
    rfecv = rfecv.fit(X, Y)

    util.end_timer_print_duration()
    print("Feature Names:", band_names)
    print("RFECV RESULTS")
    print("Optimal Number of features:", rfecv.n_features_)
    print("Selected Features:", rfecv.support_)
    print("Feature Ranking:", rfecv.ranking_)
    print("Grid Scores:", (range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_))

    plt.figure()
    plt.xlabel("Number of features selected")
    plt.ylabel("3-Fold Cross validation score (mean accuracy)")
    plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
    plt.savefig(results_folder + 'rfecv.png')

    print("RFE RESULTS ")
    for i in range(1, len(band_names) + 1):
        lr_model_2 = LogisticRegression(multi_class='ovr', solver='liblinear')
        rfe = RFE(lr_model_2, i)
        rfe = rfe.fit(X, Y)
        print("Selected Number of features:", rfe.n_features_)
        print("Selected Features:", rfe.support_)
        print("Feature Ranking:", rfe.ranking_)


# DOES NOT BALANCE - Exhaustive Search
def experiment_feature_selection_exhaustive_search(data_frame, band_names):
    pixel_index_array = np.transpose(data_frame)[0]
    label_array = np.transpose(data_frame)[-1]
    feature_data = np.delete(data_frame, 0, axis=1)
    feature_data = np.delete(feature_data, -1, axis=1)

    print("Configs: Standard Scaler, Logistic Regression")
    print(csv_string_builder_header_row(band_names, 3, RosebelPixelClass3, lambda z: z+1))
    for i in range(512):
        x = bin(i).split('b')[1]
        prepend_bit_count = 9 - len(x)
        x = (prepend_bit_count * '0') + x
        y = [i == '1' for i in x]  # 1 / True is included, 0 / False is removed
        indexes_to_be_removed = []
        indexes_to_be_included = []
        for j in range(len(y)):
            if y[j] == False:
                indexes_to_be_removed.append(j)
            else:
                indexes_to_be_included.append(j)

        # Skip remove all features
        if len(indexes_to_be_included) == 0:
            continue

        feature_data_subset = np.delete(feature_data, indexes_to_be_removed, axis=1)

        x_train, x_test, y_train, y_test = train_test_split(
            feature_data_subset, label_array, test_size=0.33)

        sc = StandardScaler()
        sc.fit(x_train)  # Fit to training data only

        x_train = sc.transform(x_train)
        x_test = sc.transform(x_test)

        lr_model = LogisticRegression(solver='liblinear', multi_class='ovr')
        lr_model.fit(x_train, y_train)
        predictions = lr_model.predict(x_test)

        accuracy_score = lr_model.score(x_test, y_test)
        confusion_matrix = metrics.confusion_matrix(y_test, predictions)

        print(csv_string_builder_data(y, accuracy_score, confusion_matrix))


def csv_string_builder_header_row(band_names, expected_confusion_matrix_size, enum, map_cf_index_to_enum):
    result_str = ""
    for x in band_names:
        result_str += x
        result_str += ","

    result_str += "overall_accuracy"
    result_str += ","
    for i in range(expected_confusion_matrix_size):
        for j in range(expected_confusion_matrix_size):
            if i == j:
                result_str += enum(map_cf_index_to_enum(i)).name
                result_str += "_accuracy,"
            else:
                result_str += enum(map_cf_index_to_enum(i)).name
                result_str += "_mispredict_as_"
                result_str += enum(map_cf_index_to_enum(j)).name
                result_str += ","

    result_str = result_str[:-1]
    return result_str


# todo add overall accuracy score
def csv_string_builder_data(feature_included, overall_accuracy, confusion_matrix):
    result_str = ""
    for x in feature_included:
        result_str += str(x)
        result_str += ","
    result_str += str('%.2f' % (overall_accuracy*100))
    result_str += ","

    for i in range(len(confusion_matrix)):
        row_sum = sum(confusion_matrix[i])
        for j in range(len(confusion_matrix)):
            percentage = str('%.2f' % (confusion_matrix[i][j]*100/row_sum))
            result_str += percentage
            result_str += ","
    result_str = result_str[:-1]  # remove trailing comma
    return result_str


#todo
def experiment_exclude_forest2(data_frame):
    pass


if __name__ == "__main__":
    # Experiments Setup ------------------------------------------------------------------------------------------------
    START = str(int(time.time()))
    EXPERIMENTS_TITLE = "W13_Visualizations" + START
    SINGLE_PRODUCT_PATH = "..//data//processed//Subset_S1A_IW_GRDH_1SDV_20170903T092838_20170903T092903_018209_01E9A9_D2A2_Orb_NR_Cal_Spk_TC_GLCM.dim"
    PINS_TEXT_FILE_PATH = "..//data//pins//rosebel_pins.txt"
    PRE_COMPUTED_LABELS_PATH = "..//data//labels//rosebel_grd_3_class_labels_original.npy"
    CLASS_LABEL_ENUM = RosebelPixelClass3
    RESULTS_FOLDER = "..//results//Rosebel_GRD//"

    print(EXPERIMENTS_TITLE)
    print("Products Used:\n", SINGLE_PRODUCT_PATH)
    print("Pins File Used:\n", PINS_TEXT_FILE_PATH)
    print("Pre-Computed Label File Used:\n", PRE_COMPUTED_LABELS_PATH)
    print("Class Label Enum Used:\n", CLASS_LABEL_ENUM, CLASS_LABEL_ENUM.__members__)
    # print('PERSISTED FILE USED', 'balanced_filtered_data_frame_20170903.npy')
    print("Generating & persisting New Data-frames for this experiment")

    # Basic Initial Processing ----------------------------------------------------------------------------------------
    gt = GroundTruthBoundaries(PINS_TEXT_FILE_PATH, CLASS_LABEL_ENUM)
    feature_array, image_width, image_height, number_of_pixels, band_names = extract_feature_array_from_product(SINGLE_PRODUCT_PATH)

    # Image Label Array
    # image_label_array = gt.get_labels_npy_exclude_some_polygons(image_width, number_of_pixels, [('forest', '2')])
    image_label_array = np.load(PRE_COMPUTED_LABELS_PATH)

    # Base Data Frame
    # data_frame = np.load(RESULTS_FOLDER + 'unbalanced_filtered_data_frame_data_converted_20170903.npy')
    data_frame = generate_filtered_data_frame_with_pixel_index_and_labels(number_of_pixels, feature_array, image_label_array)

    # Balance Data Frame
    # balanced_data_frame = np.load(RESULTS_FOLDER + 'balanced_filtered_data_frame_data_converted_20170903.npy')
    balanced_data_frame = balance_data_frame(data_frame)

    # Persisting
    np.save(RESULTS_FOLDER + 'filtered_np_df_pixel_id_label.npy', data_frame)
    np.save(RESULTS_FOLDER + 'balanced_filtered_np_df_pixel_id_label.npy', balanced_data_frame)

    # Experiments
    # experiment_original_multi_model_lr(balanced_data_frame)
    # experiment_cross_validated_recursive_feature_elimination(balanced_data_frame, band_names, RESULTS_FOLDER)
    # experiment_feature_selection_exhaustive_search(balanced_data_frame, band_names)
    experiment_original_single_model_lr(RESULTS_FOLDER,
                                        gt,
                                        image_width,
                                        RosebelPixelClass3,
                                        balanced_data_frame,
                                        error_discover=False,
                                        save_directory="results/W13_Visualizations"
                                        )
