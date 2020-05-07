# Python standard Libraries
import time
import operator

# ESA Snappy
from snappy import ProductIO

# Scientific Libraries
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics

# Self Defined Modules
from ground_truth import RosebelPixelClass3, GroundTruthBoundaries
import main as m
import utility as util

# PRODUCT_1_PATH = "..\\data\\processed\\Rosebel_GRD\\Subset_S1A_IW_GRDH_1SDV_20170903T092838_20170903T092903_018209_01E9A9_D2A2_Orb_NR_Cal_Spk_TC_GLCM.dim"
# PRODUCT_1_PATH = "..\\data\\processed\\Rosebel_GRD\\Subset_S1A_IW_GRDH_1SDV_20170903T092838_20170903T092903_018209_01E9A9_D2A2_Orb_NR_Cal_Spk_TF_TC_Gamma.dim"
PRODUCT_1_PATH = "..\\data\\processed\\Rosebel_SLC\\Subset_S1A_IW_SLC__1SDV_20170903T092838_20170903T092905_018209_01E9A9_63BD_Orb_TNR_Cal_deb_Spk_TF_TC_5m_Gamma.dim"
# PRODUCT_1_LABELS_PATH = "..\\data\\labels\\rosebel_grd_3_class_labels_original.npy"
# PRODUCT_1_LABELS_PATH = "..\\data\\labels\\rosebel_grd_2_class_labels.npy"
# PRODUCT_1_LABELS_PATH = "..\\data\\labels\\rosebel_slc_3_class_labels.npy"
PRODUCT_1_LABELS_PATH = "..\\data\\labels\\rosebel_slc_2_class_labels.npy"

# PRODUCT_2_PATH = "..\\data\\processed\\Obuasi\\Subset_S1A_IW_GRDH_1SDV_20190208T182602_20190208T182631_025842_02E01C_899E_Orb_NR_Cal_Spk_TC.dim"
PRODUCT_2_PATH = "..\\data\\processed\\Obuasi\\Subset_S1A_IW_GRDH_1SDV_20190208T182602_20190208T182631_025842_02E01C_899E_Orb_NR_Cal_Spk_TF_TC_Gamma.dim"
PRODUCT_2_LABELS_PATH = "..\\data\\labels\\obuasi_3_class_labels_original.npy"

# PRODUCT_3_PATH = "..\\data\\processed\\Merian\Subset_S1B_IW_GRDH_1SDV_20170112T215059_20170112T215124_003821_00691E_8AF1_Orb_NR_Cal_Spk_TC.dim"
PRODUCT_3_PATH = "..\\data\\processed\\Merian\Subset_S1B_IW_GRDH_1SDV_20170112T215059_20170112T215124_003821_00691E_8AF1_Orb_NR_Cal_Spk_TF_TC_Gamma.dim"
PRODUCT_3_LABELS_PATH = "..\\data\\labels\\merian_3_class_labels_original.npy"
MODEL_PATH = "..\\data\\models\\"

def print_duration_string(start_time):
    t = time.time() - start_time
    print("    Completed in - " + str(int(t // 3600)) + " hours " + str(int((t // 60) % 60)) + " minutes " + str(
        int(t % 60)) + " seconds ")

if __name__ == "__main__":
    start_time = time.time()

    products = []
    products.append(PRODUCT_1_PATH)
    # products.append(PRODUCT_2_PATH)
    # products.append(PRODUCT_3_PATH)
    labels = []
    labels.append(PRODUCT_1_LABELS_PATH)
    # labels.append(PRODUCT_2_LABELS_PATH)
    # labels.append(PRODUCT_3_LABELS_PATH)
    print("Listing all products to be considered for training/testing: " + str(products))

    data_frame_collated = []
    for product in products:
        print("Reading product: " + product)
        p = ProductIO.readProduct(product)

        print('\tExtracting Bands')
        band_names = p.getBandNames()
        bands = []
        number_of_bands = 0
        for band_name in band_names:
            print("\t" + str(number_of_bands + 1) + ": " + str(band_name))
            if str(band_name) == "elevation":
                print("\tRemoving elevation band")
            else:
                number_of_bands += 1
                bands.append(p.getBand(band_name))
            # elif str(band_name) == "Gamma0_VV" or str(band_name) == "Gamma0_VH" or str(band_name) == "Sigma0_VV" or str(band_name) == "Sigma0_VH":
            #     number_of_bands += 1
            #     bands.append(p.getBand(band_name))
            # else:
            #     print("skipping band: " + str(band_name))
        print("\tNumber of bands in product: " + str(number_of_bands))

        print('\tExtracting Feature Data from Bands')
        features = []
        for band in bands:
            w = band.getRasterWidth()
            h = band.getRasterHeight()
            x = np.zeros(w * h, np.float32)
            band.readPixels(0, 0, w, h, x)
            features.append(x)
        image_width = bands[0].getRasterWidth()
        print("\tImage width: " + str(image_width))
        image_height = bands[0].getRasterHeight()
        print("\tImage height: " + str(image_height))
        number_of_pixels = image_height * image_width
        print("\tNumber of pixels: " + str(number_of_pixels))
        # Filing dummy data to be removed later once actual data is present
        if isinstance(data_frame_collated, list):
            data_frame_collated.append([0] * (len(features)+2))
            data_frame_collated = np.array(data_frame_collated)
        features = np.array(features)

        print("\tImporting labels")
        image_label_array = np.load(labels[products.index(product)])

        print("\tPreparing data frame")
        data_frame = m.generate_filtered_data_frame_with_pixel_index_and_labels(number_of_pixels, features, image_label_array)
        data_frame_labels = list(map(int, np.transpose(data_frame)[-1].tolist()))
        print("\tBefore balance: " + str(Counter(data_frame_labels)))
        balanced_data_frame = m.balance_data_frame(data_frame)
        # balanced_data_frame = data_frame
        data_frame_labels = list(map(int, np.transpose(balanced_data_frame)[-1].tolist()))
        print("\tAfter balance: " + str(Counter(data_frame_labels)))

        print("\tData frame shape: " + str(balanced_data_frame.shape))
        if "numpy" in str(type(data_frame_collated)):
            if data_frame_collated.shape == (1, len(features)+2):
                data_frame_collated = np.concatenate((data_frame_collated, balanced_data_frame), axis=0)
                data_frame_collated = np.delete(data_frame_collated, 0, axis=0)
            else:
                data_frame_collated = np.concatenate((data_frame_collated, balanced_data_frame), axis=0)
        print("\tCollated data frame shape: " + str(data_frame_collated.shape))

    # exit()
    # Train LR/RF model
    np.random.shuffle(data_frame_collated)
    data_frame_labels = np.transpose(data_frame_collated)[-1]
    data_frame_features_with_pixel_index = np.delete(data_frame_collated, -1, axis=1)
    x_train, x_test, y_train, y_test = train_test_split(
        data_frame_features_with_pixel_index, data_frame_labels, test_size=0.33)
    x_train_scaled, x_test_scaled, fitted_scaler = m.scale_but_ignore_index_column(x_train, x_test, StandardScaler)
    x_train_scaled_no_index = np.delete(x_train_scaled, 0, axis=1)
    x_test_scaled_no_index = np.delete(x_test_scaled, 0, axis=1)
    x_train_no_index = np.delete(x_train, 0, axis=1)
    x_test_no_index = np.delete(x_test, 0, axis=1)

    util.start_timer()
    # print("Fitting Logistic Regression Model with Training Data")
    # lr_model = LogisticRegression(multi_class='ovr', solver='liblinear')
    # lr_model.fit(x_train_scaled_no_index, y_train)
    print("Fitting Random Forest Model with Training Data")
    rf_model = RandomForestClassifier(n_estimators=50, max_depth=50, min_samples_leaf=22)
    rf_model.fit(x_train_no_index, y_train)
    util.end_timer_print_duration()

    # # Evaluate model
    # print("Evaluating Logistic Regression Model with Test Data")
    # util.start_timer()
    # accuracy_score = lr_model.score(x_test_scaled_no_index, y_test)
    # test_predictions = lr_model.predict(x_test_scaled_no_index)
    # confusion_matrix = metrics.confusion_matrix(y_test, test_predictions)
    # util.end_timer_print_duration()
    #
    # # Summarize Results
    # print("\n@@@@@@@@@@@@@@@@@@@@@@@@\n" + "CLASSIFICATION SUMMARY" + "\n@@@@@@@@@@@@@@@@@@@@@@@@\n")
    # util.print_train_test_pixel_summary(y_train, y_test, RosebelPixelClass3)
    # print("Accuracy:\n" + str(accuracy_score))
    # print("\nConfusion Matrix:\n", confusion_matrix)
    # util.print_translate_confusion_matrix(confusion_matrix, RosebelPixelClass3, lambda x: x + 1)

    # joblib.dump(lr_model, MODEL_PATH + "multi_lr_model.joblib")
    # joblib.dump(fitted_scaler, MODEL_PATH + "multi_std_scaler.joblib")

    # Evaluate model
    print("Evaluating Random Forest Model with Test Data")
    util.start_timer()
    accuracy_score = rf_model.score(x_test_no_index, y_test)
    test_predictions = rf_model.predict(x_test_no_index)
    confusion_matrix = metrics.confusion_matrix(y_test, test_predictions)
    util.end_timer_print_duration()

    # Summarize Results
    print("\n@@@@@@@@@@@@@@@@@@@@@@@@\n" + "CLASSIFICATION SUMMARY" + "\n@@@@@@@@@@@@@@@@@@@@@@@@\n")
    util.print_train_test_pixel_summary(y_train, y_test, RosebelPixelClass3)
    print("Accuracy:\n" + str(accuracy_score))
    print("\nConfusion Matrix:\n", confusion_matrix)
    util.print_translate_confusion_matrix(confusion_matrix, RosebelPixelClass3, lambda x: x + 1)

    joblib.dump(rf_model, MODEL_PATH + "rf_rosebel_slc_g_water_model.joblib")
    print_duration_string(start_time)