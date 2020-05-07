# Python standard Libraries
import time
import operator

# ESA Snappy
from snappy import ProductIO

# Scientific Libraries
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from pykml import parser

# Self Defined Modules
from ground_truth import RosebelPixelClass3, GroundTruthBoundaries

# PRODUCT_PATH = "..\\data\\processed\\Rosebel_GRD\\Subset_S1A_IW_GRDH_1SDV_20170903T092838_20170903T092903_018209_01E9A9_D2A2_Orb_NR_Cal_Spk_TC_GLCM.dim"
# PRODUCT_PATH = "..\\data\\processed\\Rosebel_GRD\\Subset_S1A_IW_GRDH_1SDV_20170903T092838_20170903T092903_018209_01E9A9_D2A2_Orb_NR_Cal_Spk_TF_TC_Gamma.dim"
# CLASS_LABEL_FILENAME = "..\\data\\labels\\rosebel_grd_3_class_labels_original.npy"
# CLASS_LABEL_FILENAME = "..\\data\\labels\\rosebel_grd_2_class_labels.npy"
# SAVE_DIR = "..\\results\\Rosebel_GRD\\"
# PIN_EXPORT_FILEPATH = "..\\data\\pins\\rosebel_grd_pins.txt"

PRODUCT_PATH = "..\\data\\processed\\Rosebel_SLC\\Subset_S1A_IW_SLC__1SDV_20170903T092838_20170903T092905_018209_01E9A9_63BD_Orb_TNR_Cal_deb_Spk_TF_TC_5m_Gamma.dim"
# CLASS_LABEL_FILENAME = "..\\data\\labels\\rosebel_slc_3_class_labels.npy"
CLASS_LABEL_FILENAME = "..\\data\\labels\\rosebel_slc_2_class_labels.npy"
SAVE_DIR = "..\\results\\Rosebel_SLC\\"
PIN_EXPORT_FILEPATH = "..\\data\\pins\\rosebel_slc_pins.txt"

# PRODUCT_PATH = "..\\data\\processed\\Obuasi\\Subset_S1A_IW_GRDH_1SDV_20190208T182602_20190208T182631_025842_02E01C_899E_Orb_NR_Cal_Spk_TC.dim"
# PRODUCT_PATH = "..\\data\\processed\\Obuasi\\Subset_S1A_IW_GRDH_1SDV_20190208T182602_20190208T182631_025842_02E01C_899E_Orb_NR_Cal_Spk_TF_TC_Gamma.dim"
# CLASS_LABEL_FILENAME = "..\\data\\labels\\obuasi_3_class_labels_original.npy"
# SAVE_DIR = "..\\results\\Obuasi\\"
# PIN_EXPORT_FILEPATH = "..\\data\\pins\\obuasi_pins.txt"

# PRODUCT_PATH = "..\\data\\processed\\Merian\\Subset_S1B_IW_GRDH_1SDV_20170112T215059_20170112T215124_003821_00691E_8AF1_Orb_NR_Cal_Spk_TC.dim"
# PRODUCT_PATH = "..\\data\\processed\\Merian\\Subset_S1B_IW_GRDH_1SDV_20170112T215059_20170112T215124_003821_00691E_8AF1_Orb_NR_Cal_Spk_TF_TC_Gamma.dim"
# CLASS_LABEL_FILENAME = "..\\data\\labels\\merian_3_class_labels_original.npy"
# SAVE_DIR = "..\\results\\Merian\\"
# PIN_EXPORT_FILEPATH = "..\\data\\pins\\merian_pins.txt"

# PRODUCT_PATH = "..\\data\\processed\\LaPampa\\Subset_S1A_IW_GRDH_1SDV_20160909T101448_20160909T101513_012974_01487C_E55B_Orb_NR_Cal_Spk_TC.dim"
# PRODUCT_PATH = "..\\data\\processed\\LaPampa\\Subset_S1A_IW_GRDH_1SDV_20160909T101448_20160909T101513_012974_01487C_E55B_Orb_NR_Cal_Spk_TF_TC_Gamma.dim"
# CLASS_LABEL_FILENAME = "..\\data\\labels\\lapampa_3_class_labels_original.npy"
# SAVE_DIR = "..\\results\\LaPampa\\"
# PIN_EXPORT_FILEPATH = "..\\data\\pins\\lapampa_pins.txt"

MODEL_PATH = "..\\data\\models\\"
LABELS_DIR = "..\\data\\labels\\"
POTENTIAL_NOISE_LABEL = 0

def print_duration_string(start_time):
    t = time.time() - start_time
    print("    Completed in - " + str(int(t // 3600)) + " hours " + str(int((t // 60) % 60)) + " minutes " + str(
        int(t % 60)) + " seconds ")

def convert_kml_to_placemark(kml_filepath, placemark_filepath):
    print("Extracting placement xml file from kml file at " + str(kml_filepath))
    plm = open(placemark_filepath, "w+")
    plm.write("<?xml version=\"1.0\" encoding=\"ISO-8859-1\"?>\n")
    plm.write("<Placemarks>\n")
    with open(kml_filepath) as kml:
        doc = parser.parse(kml).getroot()
    placemarks = doc.findall('.//{http://www.opengis.net/kml/2.2}Placemark')
    for pin in placemarks:
        name = str(pin.name)
        plm.write("\t<Placemark name=\"" + name + "\">\n")
        plm.write("\t\t<LABEL>" + name + "</LABEL>\n")
        plm.write("\t\t<DESCRIPTION />\n")
        lon, lat = str(pin.Point.coordinates).split(",")[0:2]
        plm.write("\t\t<LATITUDE>" + str(lat) + "</LATITUDE>\n")
        plm.write("\t\t<LONGITUDE>" + str(lon) + "</LONGITUDE>\n")
        plm.write("\t\t<PIXEL_X>0</PIXEL_X>\n")
        plm.write("\t\t<PIXEL_Y>0</PIXEL_Y>\n")
        if "forest" in name:
            plm.write("\t\t<STYLE_CSS>fill:#00ff00; fill-opacity:0.7; stroke:#ffffff; stroke-opacity:1.0; stroke-width:0.5; symbol:pin</STYLE_CSS>\n")
        elif "water" in name:
            plm.write("\t\t<STYLE_CSS>fill:#0000ff; fill-opacity:0.7; stroke:#ffffff; stroke-opacity:1.0; stroke-width:0.5; symbol:pin</STYLE_CSS>\n")
        elif "mines" in name:
            plm.write("\t\t<STYLE_CSS>fill:#ff0000; fill-opacity:0.7; stroke:#ffffff; stroke-opacity:1.0; stroke-width:0.5; symbol:pin</STYLE_CSS>\n")
        else:
            plm.write("\t\t<STYLE_CSS>fill:#000000; fill-opacity:0.7; stroke:#ffffff; stroke-opacity:1.0; stroke-width:0.5; symbol:pin</STYLE_CSS>\n")
        plm.write("\t</Placemark>\n")
    plm.write("</Placemarks>")
    plm.close()
    print("Placemark file extracted to " + str(placemark_filepath))

def export_results(gt_labels, prediction_label, save_filepath):
    confusion_matrix = metrics.confusion_matrix(gt_labels, prediction_label)
    accuracy_score = metrics.accuracy_score(gt_labels, prediction_label)
    classification_report = metrics.classification_report(gt_labels, prediction_label)

    print("Writing results to file")
    f = open(save_filepath, "w+")
    f.write("Total labelled water pixel count:" +
            str(len(np.argwhere(np.array(gt_labels) == RosebelPixelClass3.water.value))) + '\n')
    f.write("Total labelled mines pixel count:" +
            str(len(np.argwhere(np.array(gt_labels) == RosebelPixelClass3.mines.value))) + '\n')
    f.write("Total labelled forest pixel count:" +
            str(len(np.argwhere(np.array(gt_labels) == RosebelPixelClass3.forest.value))) + '\n')
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

    # kml_filepath = "C:\\Users\\royce\\Desktop\\Philemon\\ASM_PINS.kml"
    # placemark_filepath = "C:\\Users\\royce\\Desktop\\pin.placemark"
    # convert_kml_to_placemark(kml_filepath, placemark_filepath)
    # exit()

    print("Reading product:" + PRODUCT_PATH)
    p = ProductIO.readProduct(PRODUCT_PATH)

    print('Extracting Bands')
    band_names = p.getBandNames()
    bands = []
    number_of_bands = 0
    for band_name in band_names:
        print(str(number_of_bands+1) + ": " + str(band_name))
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
    image_height = bands[0].getRasterHeight()
    number_of_pixels = image_height * image_width
    print("Number of pixels: " + str(number_of_pixels))

    print("Processing Ground Truth")
    gt = GroundTruthBoundaries(PIN_EXPORT_FILEPATH, RosebelPixelClass3)
    image_label_array = gt.get_labels_npy(image_width, number_of_pixels)
    np.save(LABELS_DIR + CLASS_LABEL_FILENAME, image_label_array)
    exit()
    labels = np.load(LABELS_DIR + CLASS_LABEL_FILENAME)

    features = np.array(features)
    print("Preparing Features for Logistic Regression, Random Forest, KMeans")
    # rf_rosebel_merian_features = np.delete(features, 2, axis=0).transpose()
    rf_features = np.delete(features, 2, axis=0).transpose()
    # lr_features = np.transpose(features)
    # rf_features = lr_features
    # print(lr_features.shape)
    # print("Preparing Features for K-Means")
    # kmeans_features_to_drop = [3, 5, 6, 8]
    # kmeans_features = np.delete(features, kmeans_features_to_drop, axis=0)
    # kmeans_features = np.transpose(kmeans_features)
    # print(kmeans_features.shape)

    # print("Importing KMeans model")
    # kmeans_model = joblib.load(MODEL_PATH + "KMeans_40_model.joblib")
    # # kmeans_scaler = joblib.load(MODEL_PATH + "KMeans_scaler.joblib")
    # kmeans_scaler = StandardScaler()
    # kmeans_scaler.fit(kmeans_features)
    # kmeans_features = kmeans_scaler.transform(kmeans_features)
    # print("Predicting cluster assignments")
    # clusassign = kmeans_model.predict(kmeans_features)
    # number_of_clusters = len(set(clusassign))
    # print(str(number_of_clusters) + " clusters found")

    # print("Importing Logistic Regression model")
    # lr_model = joblib.load(MODEL_PATH + "multi_lr_model.joblib")     # lr_model.joblib
    # lr_scaler = joblib.load(MODEL_PATH + "multi_std_scaler.joblib")  # std_scaler.joblib
    # # lr_scaler = StandardScaler()
    # # lr_scaler.fit(lr_features)
    # lr_features = lr_scaler.transform(lr_features)
    # print("Predicting lr assignments")
    # lr_predictions = lr_model.predict(lr_features).astype(int)
    # print(lr_predictions)

    print("Importing Random Forest model")
    rf_model = joblib.load(MODEL_PATH + "rf_R_O_M_g_model_balanced.joblib")      # clf / multi_rf_model / rf_R_O_M_model_balanced / rf_Rosebel_Obuasi_model
    print("Predicting rf assignments")
    rf_predictions = rf_model.predict(rf_features).astype(int)
    print(rf_predictions)

    # print("Importing RF Rosebel-Merian model")
    # rf_rosebel_merian_model = joblib.load("C:\\Users\\royce\\Desktop\\rf_rosebel.joblib")
    # print("Predicting rf_rosebel_merian assignments")
    # rf_rosebel_merian_prediction = rf_rosebel_merian_model.predict(rf_rosebel_merian_features).astype(int)
    # print(rf_rosebel_merian_prediction)

    # print("Mapping cluster assignments to Random Forest predictions")
    # clus_rf_mappings = {}
    # for index in range(len(clusassign)):
    #     cluster_number = clusassign[index]
    #     if cluster_number not in clus_rf_mappings:
    #         clus_rf_mappings[cluster_number] = {}
    #         clus_rf_mappings[cluster_number][rf_predictions[index]] = 1
    #     elif rf_predictions[index] not in clus_rf_mappings[cluster_number]:
    #         clus_rf_mappings[cluster_number][rf_predictions[index]] = 1
    #     else:
    #         clus_rf_mappings[cluster_number][rf_predictions[index]] += 1
    # print(clus_rf_mappings)
    #
    # for cluster_number, mappings in clus_rf_mappings.items():
    #     highest_yield_key = max(mappings.items(), key=operator.itemgetter(1))[0]
    #     clus_rf_mappings[cluster_number] = highest_yield_key
    # print(clus_rf_mappings)

    # for cluster_number, mappings in clus_rf_mappings.items():
    #     potential_noise_flag = 0
    #     highest_yield_key = max(mappings.items(), key=operator.itemgetter(1))[0]
    #     highest_yield_val = max(mappings.items(), key=operator.itemgetter(1))[1]
    #     for key, val in mappings.items():
    #         if key != highest_yield_key:
    #             percent_val_diff = float((highest_yield_val-val)/highest_yield_val) * 100
    #             if percent_val_diff < 20.0:
    #                 potential_noise_flag = 1
    #     if potential_noise_flag:
    #         clus_rf_mappings[cluster_number] = POTENTIAL_NOISE_LABEL
    #     else:
    #         clus_rf_mappings[cluster_number] = highest_yield_key
    # print(clus_rf_mappings)

    # print("Remapping " + str(number_of_clusters) + " clusters into 3 main classes - water, mines and forest")
    # for index in range(len(clusassign)):
    #     clusassign[index] = clus_rf_mappings[clusassign[index]]

    print("Evaluating kmeans-rf ensemble method and rf prediction separately")
    polygon_labels = []
    # polygon_clusassign = []
    polygon_rfassign = []
    # polygon_lrassign = []
    # polygon_rf_rosebel_merian_assign = []
    for index in range(number_of_pixels):
        if labels[index] != RosebelPixelClass3.na.value:
            polygon_labels.append(labels[index])
            # polygon_clusassign.append(clusassign[index])
            polygon_rfassign.append(rf_predictions[index])
            # polygon_lrassign.append(lr_predictions[index])
            # polygon_rf_rosebel_merian_assign.append(rf_rosebel_merian_prediction[index])

    # print("Evaluating lr prediction:")
    # save_kmeans_results = SAVE_DIR + "results_lr.txt"
    # export_results(polygon_labels, polygon_lrassign, save_kmeans_results)
    #
    # print("Evaluating kmeans-rf ensemble:")
    # save_kmeans_results = SAVE_DIR + "results_kmeans_rf.txt"
    # export_results(polygon_labels, polygon_clusassign, save_kmeans_results)

    print("Evaluating rf prediction:")
    save_rf_results = SAVE_DIR + "results_rf.txt"
    export_results(polygon_labels, polygon_rfassign, save_rf_results)

    # print("Evaluating rf rosebel-merian prediction:")
    # save_rf_rosebel_merian_results = SAVE_DIR + "results_rf_rosebel_merian.txt"
    # export_results(polygon_labels, polygon_rf_rosebel_merian_assign, save_rf_rosebel_merian_results)

    # print("Exporting image based on cluster assignments")
    # clusassign = np.reshape(clusassign, (image_height, image_width))
    # print(clusassign.shape)
    # imgplot = plt.imshow(clusassign, cmap='brg')   # gist_rainbow/gist_ncar
    # imgplot.write_png(SAVE_DIR + "clus_image.png")

    print("Exporting image based on rf assignments")
    rf_predictions = np.reshape(rf_predictions, (image_height, image_width))
    print(rf_predictions.shape)
    imgplot = plt.imshow(rf_predictions, cmap='brg')
    imgplot.write_png(SAVE_DIR + "rf_image.png")

    # print("Exporting image based on lr assignments")
    # lr_predictions = np.reshape(lr_predictions, (image_height, image_width))
    # print(lr_predictions.shape)
    # imgplot = plt.imshow(lr_predictions, cmap='brg')
    # imgplot.write_png(SAVE_DIR + "lr_image.png")

    # print("Exporting image based on rf rosebel-merian assignments")
    # rf_rosebel_merian_prediction = np.reshape(rf_rosebel_merian_prediction, (image_height, image_width))
    # print(rf_rosebel_merian_prediction.shape)
    # imgplot = plt.imshow(rf_rosebel_merian_prediction, cmap='brg')
    # imgplot.write_png(SAVE_DIR + "rf_rosebel_merian_image.png")

    print_duration_string(start_time)