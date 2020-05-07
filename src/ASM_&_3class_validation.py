# Python standard Libraries
import time
from enum import Enum

# ESA Snappy
from matplotlib.colors import LinearSegmentedColormap
from snappy import ProductIO

# Scientific Libraries
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn import metrics

# Self Defined Modules
from ground_truth import GroundTruthBoundaries

# ROSESEL_PATH = "F:\FYP\Processed_Data\Rosebel_GRD\ASM\\"
# PRODUCT_NAME = "Collocated_S1A_20190917T092852_S2_20190922T141049_uint8scale.dim"
# NDWI_PRED_PATH = "ndwi_pred_0.5threshold_corrected.npy"
# LAND_WATER_PRED_PATH = "land_water_pred_0.5threshold_corrected.npy"

MERIAN_PATH = "F:\FYP\Processed_Data\Merian\\"
PRODUCT_NAME = "Merian_Collocated_S1B_20190830T215119_S2B_20190902_UINT8_Corrected.dim"
LAND_WATER_PRED_PATH = "Merian_Land_Water_20190902_pred_0.5_threshold.npy"

WINDOW_SIZE = 16
water_pred_no = 1
land_pred_no = 2
forest_pred_no = 3
water_threshold = 0.01
forest_threshold = 1
PIN_EXPORT_FILEPATH = "F:\FYP\Processed_Data\Rosebel_GRD\ASM\\asm_pins.txt"
CLASS_LABEL_FILENAME = "rosebel_asm_class_labels.npy"
CLUSTERING_PATH = "F:\FYP\CG4001\machine_learning\clustering_results\\"


def print_duration_string(start_time):
    t = time.time() - start_time
    print("    Completed in - " + str(int(t // 3600)) + " hours " + str(int((t // 60) % 60)) + " minutes " + str(
        int(t % 60)) + " seconds ")


class RosebelPixelASMClass(Enum):
    na = 0
    asm = 1
    non_asm = 2


def export_results(gt_labels, prediction_label, save_filepath):
    confusion_matrix = metrics.confusion_matrix(gt_labels, prediction_label)
    accuracy_score = metrics.accuracy_score(gt_labels, prediction_label)
    classification_report = metrics.classification_report(gt_labels, prediction_label)

    print("Writing results to file")
    f = open(save_filepath, "w+")
    f.write("Total labelled ASM pixel count:" +
            str(len(np.argwhere(np.array(gt_labels) == RosebelPixelASMClass.asm.value))) + '\n')
    f.write("Overall accuracy score: " + str(accuracy_score) + '\n')
    f.write("Classification report: \n" + str(classification_report) + '\n')

    for row_index in range(confusion_matrix.shape[0]):
        #  Print class prediction accuracy first
        value = confusion_matrix[row_index][row_index]
        f.write('\n' + RosebelPixelASMClass(row_index + 1).name +
                ' pixel prediction accuracy: %.2f%%' % (value * 100 / sum(confusion_matrix[row_index])) + '\n')

        #  Print out incorrectness
        for col_index in range(confusion_matrix.shape[1]):
            if row_index != col_index:
                value = confusion_matrix[row_index][col_index]
                f.write(RosebelPixelASMClass(row_index + 1).name + ' pixels mis-predicted as ' +
                        RosebelPixelASMClass(col_index + 1).name + ': %.2f%%' % (
                                value * 100 / sum(confusion_matrix[row_index])) + '\n')
    f.close()


if __name__ == "__main__":
    start_time = time.time()

    print("Reading product:" + MERIAN_PATH + PRODUCT_NAME)
    p = ProductIO.readProduct(MERIAN_PATH + PRODUCT_NAME)

    print('Extracting Bands')
    band_names = p.getBandNames()
    bands = []
    number_of_bands = 0
    for band_name in band_names:
        if "Gamma0" in str(band_name):
            print(str(number_of_bands + 1) + ": " + str(band_name))
            number_of_bands += 1
            bands.append(p.getBand(band_name))
    print("Number of bands in product: " + str(number_of_bands))

    print('Extracting Feature Data from Bands')
    features = []
    for band in bands:
        w = band.getRasterWidth()
        h = band.getRasterHeight()
        x = np.zeros(w * h, np.float32)
        band.readPixels(0, 0, w, h, x)
        features.append(x)
    image_width = bands[0].getRasterWidth()
    image_height = bands[0].getRasterHeight()
    number_of_pixels = image_height * image_width
    print("Number of pixels: " + str(number_of_pixels))

    print("Processing Ground Truth")
    # gt = GroundTruthBoundaries(PIN_EXPORT_FILEPATH, RosebelPixelASMClass)
    # image_label_array = gt.get_labels_npy(image_width, number_of_pixels)
    # unique, counts = np.unique(image_label_array, return_counts=True)
    # yield_dict = dict(zip(unique, counts))
    # print(yield_dict)
    # np.save(MERIAN_PATH + CLASS_LABEL_FILENAME, image_label_array)
    # exit()
    # labels = np.load(MERIAN_PATH + CLASS_LABEL_FILENAME)

    features = np.array(features)
    print("Preparing Features for Random Forest")
    rf_features = np.transpose(features)

    # print("Importing Random Forest model")
    # rf_model = joblib.load(
    #     CLUSTERING_PATH + "rf_R_O_M_g_model_balanced.joblib")
    # print("Predicting rf assignments")
    # rf_predictions = rf_model.predict(rf_features).astype(int)
    # np.save(MERIAN_PATH + "rf_ROM_g_model_prediction.npy", rf_predictions)
    # print(rf_predictions)
    # exit()
    print("Importing rf assignments")
    rf_predictions = np.load(MERIAN_PATH + "rf_ROM_g_model_prediction.npy").reshape(-1, 1)

    # print("Exporting image based on rf assignments")
    # rf_predictions = np.reshape(rf_predictions, (image_height, image_width))
    # print(rf_predictions.shape)
    # imgplot = plt.imshow(rf_predictions, cmap='brg')
    # imgplot.write_png("C:\\Users\\royce\\Desktop\\merian_2019.png")

    print("Importing Land/water predictions")
    landwater_predictions = np.load(MERIAN_PATH + LAND_WATER_PRED_PATH).reshape(-1, 1)
    if landwater_predictions.size != rf_predictions.size:
        print("Land/water and 3class prediction numpys have different size. Aborting...")
        exit()

    # print("Evaluating Land/water predictions before 3 class model validation")
    # polygon_labels = []
    # polygon_pred_results = []
    # for index in range(number_of_pixels):
    #     if labels[index] != RosebelPixelASMClass.na.value:
    #         polygon_labels.append(labels[index])
    #         if landwater_predictions[index]:
    #             polygon_pred_results.append(RosebelPixelASMClass.asm.value)
    #         else:
    #             polygon_pred_results.append(RosebelPixelASMClass.non_asm.value)
    # export_results(polygon_labels, polygon_pred_results,
    #                MERIAN_PATH + "results_landwater_pred_before_s1_filter.txt")

    landwater_predictions = np.reshape(landwater_predictions, (image_height, image_width))
    rf_predictions = np.reshape(rf_predictions, (image_height, image_width))
    i = 0
    number_of_ASM_removed = 0
    while i < landwater_predictions.shape[0]:
        j = 0
        while j < landwater_predictions.shape[1]:
            if i + WINDOW_SIZE <= landwater_predictions.shape[0] and j + WINDOW_SIZE <= landwater_predictions.shape[1]:
                landwater_window = landwater_predictions[i:i + WINDOW_SIZE, j:j + WINDOW_SIZE].reshape((-1, 1))
                if landwater_window[0] == True:  # check for first value as the rest should have the same value
                    rf_window = rf_predictions[i:i + WINDOW_SIZE, j:j + WINDOW_SIZE].reshape((-1, 1))
                    # Calculate rf prediction yields in the window
                    unique, counts = np.unique(rf_window, return_counts=True)
                    yield_dict = dict(zip(unique, counts))
                    water_ratio = 0.0
                    forest_ratio = 0.0
                    if water_pred_no in yield_dict.keys():
                        water_ratio = float(yield_dict[water_pred_no] / sum(yield_dict.values()))
                    if forest_pred_no in yield_dict.keys():
                        forest_ratio = float(yield_dict[forest_pred_no] / sum(yield_dict.values()))
                    # Removes ASM predictions if majority predictions are water or forest
                    if water_ratio > water_threshold or forest_ratio > forest_threshold:
                        landwater_predictions[i:i + WINDOW_SIZE, j:j + WINDOW_SIZE] = False
                        number_of_ASM_removed += 1
            j += WINDOW_SIZE
        i += WINDOW_SIZE
    print("Number of ASM windows removed from original S2 prediction: " + str(number_of_ASM_removed))

    colors = ["white", "red"]
    cmap = LinearSegmentedColormap.from_list("test", colors)
    imgplot = plt.imshow(landwater_predictions, cmap=cmap)
    imgplot.write_png(MERIAN_PATH + "merian_landwater_s1_filter.png")

    # print("Evaluating Land/water predictions after 3 class model validation")
    # landwater_predictions = landwater_predictions.reshape((-1, 1))
    # polygon_labels = []
    # polygon_pred_results = []
    # for index in range(number_of_pixels):
    #     if labels[index] != RosebelPixelASMClass.na.value:
    #         polygon_labels.append(labels[index])
    #         if landwater_predictions[index]:
    #             polygon_pred_results.append(RosebelPixelASMClass.asm.value)
    #         else:
    #             polygon_pred_results.append(RosebelPixelASMClass.non_asm.value)
    # export_results(polygon_labels, polygon_pred_results,
    #                MERIAN_PATH + "results_landwater_pred_after_s1_filter.txt")

    np.save(MERIAN_PATH + "ASM_LandWater_S2_adjusted_0.01_water_threshold.npy", landwater_predictions)
    print_duration_string(start_time)



##################### Unused code #########################
    # x=[0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95]
    # y=[63.19, 69.30, 77.50, 82.12, 85.10, 87.21, 90.09]
    # plt.plot(x,y)
    # plt.style.use('fivethirtyeight')
    # plt.xlabel("Forest-yield filter threshold")
    # plt.ylabel("ASM accuracy/%")
    # plt.show()
    # exit()

    # Use this condition check if right-most and bottom-most pixels should be taken into account
    # if i+WINDOW_SIZE > ndwi_predictions.shape[0] and j+WINDOW_SIZE <= ndwi_predictions.shape[1]:
    #     ndwi_window = ndwi_predictions[i:ndwi_predictions.shape[0], j:j+WINDOW_SIZE].reshape((-1, 1))
    #     rf_window = rf_predictions[i:rf_predictions.shape[0], j:j+WINDOW_SIZE].reshape((-1, 1))
    # elif i+WINDOW_SIZE <= ndwi_predictions.shape[0] and j+WINDOW_SIZE > ndwi_predictions.shape[1]:
    #     ndwi_window = ndwi_predictions[i:i+WINDOW_SIZE, j:ndwi_predictions.shape[1]].reshape((-1, 1))
    #     rf_window = rf_predictions[i:i + WINDOW_SIZE, j:rf_predictions.shape[1]].reshape((-1, 1))
    # elif i+WINDOW_SIZE > ndwi_predictions.shape[0] and j+WINDOW_SIZE > ndwi_predictions.shape[1]:
    #     ndwi_window = ndwi_predictions[i:ndwi_predictions.shape[0], j:ndwi_predictions.shape[1]].reshape((-1, 1))
    #     rf_window = rf_predictions[i:rf_predictions.shape[0], j:rf_predictions.shape[1]].reshape((-1, 1))
    # else:
    #     ndwi_window = ndwi_predictions[i:i+WINDOW_SIZE, j:j+WINDOW_SIZE].reshape((-1, 1))
    #     rf_window = rf_predictions[i:i + WINDOW_SIZE, j:j + WINDOW_SIZE].reshape((-1, 1))
