# Python standard Libraries
import time
import operator

# ESA Snappy
from snappy import ProductIO

# Scientific Libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import joblib

from ground_truth import RosebelPixelClass3, GroundTruthBoundaries

PRODUCT_PATH = "F:\FYP\Processed_Data\\Rosebel_GRD\\Subset_S1A_IW_GRDH_1SDV_20170903T092838_20170903T092903_018209_01E9A9_D2A2_Orb_NR_Cal_Spk_TC_GLCM.dim"
SURINAME_GROUND_TRUTH_PINS_TEXT_FILE_PATH = "F:\FYP\CG4001\esa_data_products_snap_processed\suriname_rosebel_all\ground_truth_vector_files\pin_exports\pin.txt"
CLUSTERING_PATH = "F:\FYP\CG4001\machine_learning\clustering_results\\"
POTENTIAL_NOISE_LABEL = 0

def print_duration_string(start_time):
    t = time.time() - start_time
    print("    Completed in - " + str(int(t // 3600)) + " hours " + str(int((t // 60) % 60)) + " minutes " + str(
        int(t % 60)) + " seconds ")

if __name__ == "__main__":
    start_time = time.time()

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
    number_of_pixels = image_height * image_width
    print("Number of pixels: " + str(number_of_pixels))

    features = np.array(features)
    print("Preparing Features for Logistic Regression")
    lr_features = np.transpose(features)
    rf_features = lr_features
    print(lr_features.shape)
    print("Preparing Features for K-Means")
    kmeans_features_to_drop = [3, 5, 6, 8]
    kmeans_features = np.delete(features, kmeans_features_to_drop, axis=0)
    kmeans_features = np.transpose(kmeans_features)
    print(kmeans_features.shape)

    print("Importing KMeans model")
    kmeans_model = joblib.load(CLUSTERING_PATH + "KMeans_40_model.joblib")
    kmeans_scaler = joblib.load(CLUSTERING_PATH + "KMeans_scaler.joblib")
    kmeans_features = kmeans_scaler.transform(kmeans_features)
    print("Predicting cluster assignments")
    clusassign = kmeans_model.predict(kmeans_features)
    number_of_clusters = max(clusassign) - min(clusassign) + 1
    print(str(number_of_clusters) + " clusters found")

    print("Importing Logistic Regression model")
    lr_model = joblib.load(CLUSTERING_PATH + "multi_lr_model.joblib")     # multi_lr_model.joblib
    lr_scaler = joblib.load(CLUSTERING_PATH + "multi_std_scaler.joblib")    # multi_std_scaler.joblib
    lr_features = lr_scaler.transform(lr_features)
    print("Predicting lr assignments")
    lr_predictions = lr_model.predict(lr_features)
    lr_predictions = lr_predictions.astype(int)
    # lr_predictions = np.reshape(lr_predictions, (image_height, image_width))
    # imgplot = plt.imshow(lr_predictions, cmap='brg')
    # imgplot.write_png("C:\\Users\\royce\\Desktop\\image.png")
    # exit()

    # print("Importing Random Forest model")
    # rf_model = joblib.load(CLUSTERING_PATH + "multi_rf_model.joblib")  # clf.joblib
    # print("Predicting rf assignments")
    # rf_predictions = rf_model.predict(rf_features).astype(int)
    # print(rf_predictions)
    # rf_predictions = np.reshape(rf_predictions, (image_height, image_width))
    # imgplot = plt.imshow(rf_predictions, cmap='brg')
    # imgplot.write_png("C:\\Users\\royce\\Desktop\\image.png")
    # exit()

    print("Mapping cluster assignments to Logistic Regression predictions")
    clus_lr_mappings = {}
    for index in range(len(clusassign)):
        cluster_number = clusassign[index]
        if cluster_number not in clus_lr_mappings:
            clus_lr_mappings[cluster_number] = {}
            clus_lr_mappings[cluster_number][lr_predictions[index]] = 1
        elif lr_predictions[index] not in clus_lr_mappings[cluster_number]:
            clus_lr_mappings[cluster_number][lr_predictions[index]] = 1
        else:
            clus_lr_mappings[cluster_number][lr_predictions[index]] += 1
    print(clus_lr_mappings)

    # for cluster_number, mappings in clus_lr_mappings.items():
    #     highest_yield_key = max(mappings.items(), key=operator.itemgetter(1))[0]
    #     clus_lr_mappings[cluster_number] = highest_yield_key
    # print(clus_lr_mappings)

    for cluster_number, mappings in clus_lr_mappings.items():
        potential_noise_flag = 0
        highest_yield_key = max(mappings.items(), key=operator.itemgetter(1))[0]
        highest_yield_val = max(mappings.items(), key=operator.itemgetter(1))[1]
        for key, val in mappings.items():
            if key != highest_yield_key:
                percent_val_diff = float((highest_yield_val-val)/highest_yield_val) * 100
                if percent_val_diff < 30.0:
                    potential_noise_flag = 1
        if potential_noise_flag:
            clus_lr_mappings[cluster_number] = POTENTIAL_NOISE_LABEL
        else:
            clus_lr_mappings[cluster_number] = highest_yield_key
    print(clus_lr_mappings)
    # exit()
    print("Remapping " + str(number_of_clusters) + " clusters into 3 main classes - water, mines and forest")
    for index in range(len(clusassign)):
        clusassign[index] = clus_lr_mappings[clusassign[index]]

    print("Loading Labels from Ground Truth")
    labels = np.load("pixel_labels_from_pin_file.npy")
    polygon_labels = []
    polygon_clusassign = []
    for index in range(number_of_pixels):
        if labels[index] != RosebelPixelClass3.na.value:
            polygon_labels.append(labels[index])
            polygon_clusassign.append(clusassign[index])

    print("Exporting image based on cluster assignments")
    clusassign = np.reshape(clusassign, (image_height, image_width))
    print(clusassign.shape)
    imgplot = plt.imshow(clusassign, cmap='brg')
    imgplot.write_png(CLUSTERING_PATH + "ensemble_clusimage_reduce_" + str(number_of_clusters) + ".png")

    confusion_matrix = metrics.confusion_matrix(polygon_labels, polygon_clusassign)
    accuracy_score = metrics.accuracy_score(polygon_labels, polygon_clusassign)
    classification_report = metrics.classification_report(polygon_labels, polygon_clusassign)

    print("Writing results to file")
    f = open(CLUSTERING_PATH + "report_ensemble_reduce_" + str(number_of_clusters) + ".txt", "w+")
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
        f.write('\n' + RosebelPixelClass3(row_index+1).name +
              ' pixel prediction accuracy: %.2f%%' % (value*100 / sum(confusion_matrix[row_index])) + '\n')

        #  Print out incorrectness
        for col_index in range(confusion_matrix.shape[1]):
            if row_index != col_index:
                value = confusion_matrix[row_index][col_index]
                f.write(RosebelPixelClass3(row_index+1).name + ' pixels mis-predicted as ' +
                      RosebelPixelClass3(col_index+1).name + ': %.2f%%' % (value*100 / sum(confusion_matrix[row_index])) + '\n')
    f.close()

    print_duration_string(start_time)
