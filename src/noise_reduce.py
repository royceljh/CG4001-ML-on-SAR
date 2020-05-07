# Python standard Libraries
import time
import os.path

# ESA Snappy
from snappy import ProductIO

# Scientific Libraries
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn import metrics

# Self Defined Modules
from ground_truth import RosebelPixelClass3, GroundTruthBoundaries

PRODUCT_1A_PATH = "..\\data\\processed\\Rosebel_GRD\\Subset_S1A_IW_GRDH_1SDV_20170903T092838_20170903T092903_018209_01E9A9_D2A2_Orb_NR_Cal_Spk_TF_TC_Gamma.dim"
PRODUCT_1B_PATH = "..\\data\\processed\\Rosebel_GRD\\Subset_S1B_IW_GRDH_1SDV_20170902T215854_20170902T215945_007219_00CB9B_FFA6_Orb_NR_Cal_Asm_Spk_TF_TC.dim"
CLASS_LABEL_FILENAME = "..\\data\\labels\\rosebel_grd_3_class_labels_original.npy"
SAVE_DIR = "..\\results\\Rosebel_GRD\\"
PIN_EXPORT_FILEPATH = "..\\data\\pins\\rosebel_grd_pins.txt"

# PRODUCT_1A_PATH = "..\\data\\processed\\Obuasi\\Subset_S1A_IW_GRDH_1SDV_20190208T182602_20190208T182631_025842_02E01C_899E_Orb_NR_Cal_Spk_TF_TC_Gamma.dim"
# PRODUCT_1B_PATH = "..\\data\\processed\\Obuasi\\Subset_S1B_IW_GRDH_1SDV_20190209T181715_20190209T181740_014873_01BC28_A9A8_Orb_NR_Cal_Spk_TF_TC.dim"
# CLASS_LABEL_FILENAME = "..\\data\\labels\\obuasi_3_class_labels_original.npy"
# SAVE_DIR = "..\\results\\Obuasi\\"
# PIN_EXPORT_FILEPATH = "..\\data\\pins\\obuasi_pins.txt"

LABEL_PATH = "..\\data\\labels\\"
MODEL_PATH = "..\\data\\models\\"

def print_duration_string(start_time):
    t = time.time() - start_time
    print("    Completed in - " + str(int(t // 3600)) + " hours " + str(int((t // 60) % 60)) + " minutes " + str(
        int(t % 60)) + " seconds ")

def export_results(gt_labels, prediction_label, save_filepath):
    confusion_matrix = metrics.confusion_matrix(gt_labels, prediction_label)
    accuracy_score = metrics.accuracy_score(gt_labels, prediction_label)
    classification_report = metrics.classification_report(gt_labels, prediction_label)

    print("Writing results to file")
    f = open(save_filepath, "w+")
    f.write("Total labelled water pixel count:" +
            str(len(np.argwhere(np.array(labels) == RosebelPixelClass3.water.value))) + '\n')
    f.write("Total labelled mines pixel count:" +
            str(len(np.argwhere(np.array(labels) == RosebelPixelClass3.mines.value))) + '\n')
    f.write("Total labelled forest pixel count:" +
            str(len(np.argwhere(np.array(labels) == RosebelPixelClass3.forest.value))) + '\n')
    f.write("Confusion matrix: \n" + np.array_str(confusion_matrix) + '\n')
    f.write("Overall accuracy score: " + str(accuracy_score) + '\n')
    f.write("Classification report: \n" + str(classification_report) + '\n')

    for row_index in range(confusion_matrix.shape[0]):
        #  Print class prediction accuracy first
        value = confusion_matrix[row_index][row_index]
        f.write('\n' + RosebelPixelClass3(row_index + 1).name +
                ' pixel prediction accuracy: %.2f%%' % (value * 100 / sum(confusion_matrix[row_index])) + '\n')

        #  Print out incorrectness
        for col_index in range(confusion_matrix.shape[1]):
            if row_index != col_index:
                value = confusion_matrix[row_index][col_index]
                f.write(RosebelPixelClass3(row_index + 1).name + ' pixels mis-predicted as ' +
                        RosebelPixelClass3(col_index + 1).name + ': %.2f%%' % (
                                    value * 100 / sum(confusion_matrix[row_index])) + '\n')
    f.close()

if __name__ == "__main__":
    start_time = time.time()

    if os.path.exists(LABEL_PATH + "rf_1A_1B_predictions.npy"):
        rf_predictions = np.load(LABEL_PATH + "rf_1A_1B_predictions.npy")
    else:
        print("Reading product:" + PRODUCT_1A_PATH)
        p1A = ProductIO.readProduct(PRODUCT_1A_PATH)

        print('Extracting Bands')
        band_names = p1A.getBandNames()
        bands = []
        number_of_bands = 0
        for band_name in band_names:
            print(str(number_of_bands+1) + ": " + str(band_name))
            number_of_bands += 1
            bands.append(p1A.getBand(band_name))
        print("Number of bands in product: " + str(number_of_bands))

        print('Extracting Feature Data from Bands')
        features_1A = []
        for band in bands:
            w = band.getRasterWidth()
            h = band.getRasterHeight()
            x = np.zeros(w * h, np.float32)
            band.readPixels(0, 0, w, h, x)
            features_1A.append(x)
        image_width_1A = bands[0].getRasterWidth()
        image_height_1A = bands[0].getRasterHeight()
        number_of_pixels_1A = image_height_1A * image_width_1A
        print("Image width = " + str(image_width_1A))
        print("Image height = " + str(image_height_1A))
        print("Number of pixels: " + str(number_of_pixels_1A))
        features_1A = np.array(features_1A)
        features_1A = np.delete(features_1A, 2, axis=0).transpose()

        print("Reading product:" + PRODUCT_1B_PATH)
        p1B = ProductIO.readProduct(PRODUCT_1B_PATH)

        print('Extracting Bands')
        band_names = p1B.getBandNames()
        bands = []
        number_of_bands = 0
        for band_name in band_names:
            print(str(number_of_bands + 1) + ": " + str(band_name))
            number_of_bands += 1
            bands.append(p1B.getBand(band_name))
        print("Number of bands in product: " + str(number_of_bands))

        print('Extracting Feature Data from Bands')
        features_1B = []
        for band in bands:
            w = band.getRasterWidth()
            h = band.getRasterHeight()
            x = np.zeros(w * h, np.float32)
            band.readPixels(0, 0, w, h, x)
            features_1B.append(x)
        image_width_1B = bands[0].getRasterWidth()
        image_height_1B = bands[0].getRasterHeight()
        number_of_pixels_1B = image_height_1B * image_width_1B
        print("Image width = " + str(image_width_1B))
        print("Image height = " + str(image_height_1B))
        print("Number of pixels: " + str(number_of_pixels_1B))
        features_1B = np.array(features_1B)
        features_1B = np.delete(features_1B, 2, axis=0).transpose()

        if image_width_1A != image_width_1B or image_height_1A != image_height_1B:
            print("Unequal number of pixels found between products, aborting...")
            exit()

        print("Importing Random Forest model")
        rf_model = joblib.load(MODEL_PATH + "rf_R_O_M_g_model_balanced.joblib")
        print("Predicting rf assignments for Sentinel 1A and 1B products")
        rf_1A_predictions = rf_model.predict(features_1A).astype(int)
        rf_1A_proba = rf_model.predict_proba(features_1A)
        rf_1B_predictions = rf_model.predict(features_1B).astype(int)
        rf_1B_proba = rf_model.predict_proba(features_1B)

        rf_predictions = []
        for index in range(number_of_pixels_1A):
            if rf_1A_predictions[index] == RosebelPixelClass3.water.value or rf_1B_predictions[index] == RosebelPixelClass3.water.value:
                rf_predictions.append(RosebelPixelClass3.water.value)
            elif rf_1A_predictions[index] == rf_1B_predictions[index]:
                rf_predictions.append(rf_1A_predictions[index])
            else:
                # rf_predictions.append(RosebelPixelClass3.forest.value)    # for strict noise reduction
                ave_mines_confidence = (float(rf_1A_proba[index][RosebelPixelClass3.mines.value-1])
                                        + float(rf_1B_proba[index][RosebelPixelClass3.mines.value-1])) / 2.0
                if ave_mines_confidence > 0.5:
                    rf_predictions.append(RosebelPixelClass3.mines.value)
                else:
                    rf_predictions.append(RosebelPixelClass3.forest.value)

        rf_predictions = np.array(rf_predictions)
        np.save(LABEL_PATH + "rf_1A_1B_predictions.npy", rf_predictions)

    print("Loading Ground Truth for results validation")
    labels = np.load(SAVE_DIR + CLASS_LABEL_FILENAME)
    polygon_labels = []
    polygon_rfassign = []
    for index in range(number_of_pixels_1A):
        if labels[index] != RosebelPixelClass3.na.value:
            polygon_labels.append(labels[index])
            polygon_rfassign.append(rf_predictions[index])

    print("Evaluating rf prediction:")
    save_rf_results = SAVE_DIR + "results_rf.txt"
    export_results(polygon_labels, polygon_rfassign, save_rf_results)

    print("Exporting image based on rf assignments")
    rf_predictions = np.reshape(rf_predictions, (image_height_1A, image_width_1A))
    print(rf_predictions.shape)
    imgplot = plt.imshow(rf_predictions, cmap='brg')
    imgplot.write_png(SAVE_DIR + "rf_image.png")

    print_duration_string(start_time)