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
# PRODUCT_PATH = "..\\data\\processed\\Rosebel_GRD\\Subset_S1B_IW_GRDH_1SDV_20170902T215854_20170902T215945_007219_00CB9B_FFA6_Orb_NR_Cal_Asm_Spk_TF_TC.dim"
# PRODUCT_PATH = "..\\data\\processed\\Rosebel_SLC\\Subset_S1A_IW_SLC__1SDV_20170903T092838_20170903T092905_018209_01E9A9_63BD_Orb_NR_Cal_Deb_Spk_TC_5m.dim"
# PRODUCT_PATH = "..\\data\\processed\\Rosebel_SLC\\Subset_S1A_IW_SLC__1SDV_20170903T092838_20170903T092905_018209_01E9A9_63BD_Orb_TNR_Cal_deb_Spk_TF_TC_5m_Gamma.dim"
PRODUCT_PATH = "..\\data\\processed\\Rosebel_GRD\\Temp\\Subset_S1A_IW_GRDH_1SDV_20190917T092852_20190917T092917_029059_034C31_1942_Orb_NR_Cal_TF_Spk_TC.dim"

# PRODUCT_PATH = "..\\data\\processed\\Obuasi\\Subset_S1A_IW_GRDH_1SDV_20190208T182602_20190208T182631_025842_02E01C_899E_Orb_NR_Cal_Spk_TF_TC_Gamma.dim"

MODEL_PATH = "..\\data\\models\\"

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

    features = np.array(features)
    print("Preparing Features for Logistic Regression, Random Forest, KMeans")

    ################## Remove elevation band only ##################
    rf_features = np.delete(features, 2, axis=0).transpose()

    ######## Remove all features except for base VV and VH bands #########
    # print("Removing all features except for base VV and VH bands")
    # feature_indexes_to_drop = [2, 3, 4, 5, 6, 7, 8]
    # rf_features = np.delete(features, feature_indexes_to_drop, axis=0).transpose()
    # print(rf_features.shape)
    #####################################################################

    print("Importing Random Forest model")
    rf_model = joblib.load(
        MODEL_PATH + "rf_R_O_M_g_model_balanced.joblib")   # rf_R_O_M_g_model_balanced / rf_rosebel_water_model_VV_VH / rf_rosebel_slc_g_model / rf_rosebel_slc_g_water_model
    print("Predicting rf assignments")
    rf_predictions = rf_model.predict(rf_features).astype(int)
    # rf_predictions_proba = rf_model.predict_proba(rf_features)
    print(rf_predictions)
    # unique, counts = np.unique(rf_predictions, return_counts=True)
    # print(dict(zip(unique, counts)))
    # print(rf_predictions_proba)
    # exit()

    print("Exporting image based on rf assignments")
    rf_predictions = np.reshape(rf_predictions, (image_height, image_width))
    print(rf_predictions.shape)
    imgplot = plt.imshow(rf_predictions, cmap='brg')
    imgplot.write_png("C:\\Users\\royce\\Desktop\\rosebel_2019.png")

    print_duration_string(start_time)