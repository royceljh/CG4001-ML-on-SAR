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
import seaborn

from ground_truth import RosebelPixelClass3, GroundTruthBoundaries

PRODUCT_PATH = "F:\FYP\Processed_Data\\Subset_S1A_IW_GRDH_1SDV_20170903T092838_20170903T092903_018209_01E9A9_D2A2_Orb_NR_Cal_Spk_TC_GLCM.dim"
SURINAME_GROUND_TRUTH_PINS_TEXT_FILE_PATH = "F:\FYP\CG4001\esa_data_products_snap_processed\suriname_rosebel_all\ground_truth_vector_files\pin_exports\pin.txt"
CLUSTERING_PATH = "F:\FYP\CG4001\machine_learning\clustering_results\\"

def print_duration_string(start_time):
    t = time.time() - start_time
    print("    Completed in - " + str(int(t // 3600)) + " hours " + str(int((t // 60) % 60)) + " minutes " + str(
        int(t % 60)) + " seconds ")


def elbow_method(data):
    # Cluster anaysis for 1-9 clusters
    clusters = range(1, 9)
    meandist = []
    for k in clusters:
        model = KMeans(n_clusters=k)
        model.fit(data)
        clusassign = model.predict(data)
        meandist.append(sum(np.min(cdist(data, model.cluster_centers_, "euclidean"), axis=1)) / data.shape[0])
    plt.plot(clusters, meandist)
    plt.xlabel("Number of clusters")
    plt.ylabel("Average distance")
    plt.title("Selecting k with the Elbow method")
    plt.show()


if __name__ == "__main__":
    start_time = time.time()

    print("Reading product:" + PRODUCT_PATH)
    p = ProductIO.readProduct(PRODUCT_PATH)

    print('Extracting Bands')
    band_names = p.getBandNames()
    bands = []
    number_of_bands = 0
    for band_name in band_names:
        print(str(band_name))
        number_of_bands += 1
        bands.append(p.getBand(band_name))
    print("Number of bands: " + str(number_of_bands))

    print('Extracting Feature Data from Bands')
    features = []
    for band in bands:
        w = band.getRasterWidth()
        h = band.getRasterHeight()
        x = np.zeros(w * h, np.float32)
        band.readPixels(0, 0, w, h, x)
        features.append(x)
    features = np.array(features)
    # VV_VH_features = features
    # feature_indexes_to_drop = [0, 2, 3, 5]     # test.dim product
    feature_indexes_to_drop = [3, 5, 6, 8]
    VV_VH_features = np.delete(features, feature_indexes_to_drop, axis=0)
    VV_VH_features = np.transpose(VV_VH_features)
    print(VV_VH_features.shape)

    # Feature scaling
    print("Scaling features")
    scaler = StandardScaler()
    scaler.fit(VV_VH_features)
    VV_VH_features = scaler.transform(VV_VH_features)
    # joblib.dump(scaler, CLUSTERING_PATH + "KMeans_scaler.joblib")

    image_width = bands[0].getRasterWidth()
    image_height = bands[0].getRasterHeight()
    number_of_pixels = image_height * image_width
    print("Number of pixels: " + str(number_of_pixels))

    # Load Labels
    print("Loading Labels from Ground Truth")
    labels = np.load("pixel_labels_from_pin_file.npy")

    # elbow_method(VV_VH_features)

    # Initialise the number of clusters
    number_of_clusters = 40
    while number_of_clusters < 41:
        # Train model
        header = str(number_of_clusters) + "clusters"
        # print("Training KMeans model using " + str(number_of_clusters) + " clusters")
        # model = KMeans(n_clusters=number_of_clusters)
        # model.fit(VV_VH_features)
        # clusassign = model.predict(VV_VH_features)
        # np.save(CLUSTERING_PATH + "clusassign_"+ header + ".npy", clusassign)
        # joblib.dump(model, CLUSTERING_PATH + "KMeans_40_model.joblib")
        print_duration_string(start_time)
        exit(1)

        print("Loading persisted clustering results")
        clusassign = np.load(CLUSTERING_PATH + "clusassign_"+ header + ".npy")

        print_duration_string(start_time)

        orig_clusassign = np.reshape(clusassign, (image_height, image_width))
        imgplot = plt.imshow(orig_clusassign)
        imgplot.write_png(CLUSTERING_PATH + "orig_clusimage_" + header + ".png")

        # Calculating yield
        print("Calculating assigned cluster yield based on ground truth labels")
        water_yield = {}
        mines_yield = {}
        for index in range(number_of_pixels):
            if labels[index] == RosebelPixelClass3.water.value:
                cluster_number = clusassign[index]
                if cluster_number not in water_yield:
                    water_yield[cluster_number] = 1
                else:
                    water_yield[cluster_number] += 1
            elif labels[index] == RosebelPixelClass3.mines.value:
                cluster_number = clusassign[index]
                if cluster_number not in mines_yield:
                    mines_yield[cluster_number] = 1
                else:
                    mines_yield[cluster_number] += 1
        print(water_yield)
        print(mines_yield)
        water_cluster = max(water_yield.items(), key=operator.itemgetter(1))[0]
        print("Water cluster number = " + str(water_cluster))

        mines_cluster = max(mines_yield.items(), key=operator.itemgetter(1))[0]
        print("Mines cluster number = " + str(mines_cluster))
        # mines_yield_sorted = sorted(mines_yield.items(), key=(lambda i: i[1]))
        # mines_cluster2 = mines_yield_sorted[-3][0]

        print("Mapping clusters to labels based on calculated yields")
        for index in range(number_of_pixels):
            if clusassign[index] == water_cluster:
                clusassign[index] = 1
            elif clusassign[index] == mines_cluster:   # or (clusassign[index] == mines_cluster2):
                clusassign[index] = 2
            else:
                clusassign[index] = 3


        # # Brute Force cluster mappings to mines
        # print("Obtaining all possible combinations of Mines clusters")
        # mines_cluster_list = []
        # for i in range(2**number_of_clusters):
        #     index = 0
        #     mines_cluster = []
        #     combi = bin(i).split('b')[1]
        #     prepend_bit_count = number_of_clusters - len(combi)
        #     combi = (prepend_bit_count * '0') + combi
        #     combi_array = map(int, str(combi))
        #     for digit in combi_array:
        #         if digit == 1:
        #             mines_cluster.append(index)
        #         index += 1
        #     mines_cluster_list.append(mines_cluster)
        # print(mines_cluster_list)
        #
        # print("Exhaustive search for combination with highest mines and forest accuracy")
        # max_mines_cluster = []
        # max_weighted_accuracy = 0.0
        # corres_mines_accuracy = 0.0
        # corres_forest_accuracy = 0.0
        # for mines_cluster in mines_cluster_list:
        #     print("Starting mines cluster mines_list: " + str(mines_cluster))
        #     for index in range(number_of_pixels):
        #         value = int(clusassign[index])
        #         if value == water_cluster:
        #             clusassign[index] = 1
        #         elif value in mines_cluster:
        #             clusassign[index] = 2
        #         else:
        #             clusassign[index] = 3
        #
        #     polygon_labels = []
        #     polygon_clusassign = []
        #     for index in range(number_of_pixels):
        #         if labels[index] != RosebelPixelClass3.na.value:
        #             polygon_labels.append(labels[index])
        #             polygon_clusassign.append(clusassign[index])
        #
        #     confusion_matrix = metrics.confusion_matrix(polygon_labels, polygon_clusassign)
        #     # accuracy_score = metrics.accuracy_score(polygon_labels, polygon_clusassign)
        #     # classification_report = metrics.classification_report(polygon_labels, polygon_clusassign)
        #
        #     mines_accuracy = float(confusion_matrix[1][1]*100/sum(confusion_matrix[1]))
        #     forest_accuracy = float(confusion_matrix[2][2]*100/sum(confusion_matrix[2]))
        #     weighted_accuracy = mines_accuracy*2 + forest_accuracy
        #     print(mines_accuracy)
        #     print(forest_accuracy)
        #     print("Weighted accuracy = " + str(weighted_accuracy))
        #     if weighted_accuracy > max_weighted_accuracy:
        #         corres_mines_accuracy = mines_accuracy
        #         corres_forest_accuracy = forest_accuracy
        #         max_mines_cluster = mines_cluster
        #
        # print("Max Mines Accuracy = " + str(corres_mines_accuracy))
        # print("Corresponding Accuracy = " + str(corres_forest_accuracy))
        # print("Mines cluster combination : " + str(max_mines_cluster))

        print("Exporting image based on cluster assignments")
        clusassign = np.reshape(clusassign, (image_height, image_width))
        print(clusassign.shape)
        imgplot = plt.imshow(clusassign, cmap='brg')
        imgplot.write_png(CLUSTERING_PATH + "clusimage_" + header + ".png")

        polygon_labels = []
        polygon_clusassign = []
        for index in range(number_of_pixels):
            if labels[index] != RosebelPixelClass3.na.value:
                polygon_labels.append(labels[index])
                polygon_clusassign.append(clusassign[index])

        confusion_matrix = metrics.confusion_matrix(polygon_labels, polygon_clusassign)
        accuracy_score = metrics.accuracy_score(polygon_labels, polygon_clusassign)
        classification_report = metrics.classification_report(polygon_labels, polygon_clusassign)

        print("Writing results to file")
        f = open(CLUSTERING_PATH + "report_" + header + ".txt", "w+")
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

        plt.figure(figsize=(20,11))
        seaborn.heatmap(confusion_matrix, annot=True, linewidths=.5, square=True, cmap='Blues_r')
        plt.ylabel('Actual Pixel Class')
        plt.xlabel('Predicted Pixel Class')
        accuracy_title = "Accuracy Score: {0}".format(accuracy_score)
        plt.title(accuracy_title, size=15)
        plt.savefig(CLUSTERING_PATH + "confusion_matrix_" + header + ".png", bbox_inches='tight')

        print_duration_string(start_time)

        number_of_clusters += 1