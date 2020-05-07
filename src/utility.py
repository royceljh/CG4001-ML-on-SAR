import time
import numpy as np
# import logging

t = time.time()


def start_timer():
    global t
    t = time.time()


def end_timer_print_duration():
    global t
    duration = time.time() - t

    print("Completed in - " + str(int(duration // 3600)) + " hours " + str(int((duration // 60) % 60))
          + " minutes " + str(int(duration % 60)) + " seconds ")

#
# def configure_logger(text_file_path):
#     logging.basicConfig(level=logging.INFO, format='%(message)s', filename=text_file_path,
#                         filemode='w')
#     console = logging.StreamHandler()
#     console.setLevel(logging.INFO)
#     logging.getLogger('main').addHandler(logging.StreamHandler())
#
#
# # Redefine print
# def print(*input_tuple):
#     x = ""
#     for y in input_tuple:
#         x += str(y) + " "
#     if x[-1] == ' ':
#         x = x[:-1]
#     logging.getLogger('main').info(x)


def print_train_test_pixel_summary(y_train, y_test, enum):
    unique_train_labels, unique_train_counts = np.unique(y_train, return_counts=True)
    unique_test_labels, unique_test_counts = np.unique(y_test, return_counts=True)

    # Print total labelled pixels used for this experiment
    print("Total Labelled Pixel Count:", len(y_train) + len(y_test))
    for idx, unique_label in enumerate(unique_train_labels):
        print("Total", enum(unique_label).name, "Pixel Count:", unique_test_counts[idx] + unique_train_counts[idx])

    # Print total train pixels used for this experiment
    print("Total Train Pixels:", len(y_train))
    for idx, unique_label in enumerate(unique_train_labels):
        print("Training", enum(unique_label).name, "Pixel Count:", unique_train_counts[idx])

    # Print total test pixels used for this experiment
    print("Total Test Pixels:", len(y_test))
    for idx, unique_label in enumerate(unique_test_labels):
        print("Test", enum(unique_label).name, " Pixel Count:", unique_test_counts[idx])


def print_translate_confusion_matrix(confusion_matrix, enum, index_to_enum_value_mapper):
    """
    Prints accuracy & mis-accuracy for each class

    :param confusion_matrix: results from evaluation
    :param enum: Enum class for classification labels
    :param index_to_enum_value_mapper: function that maps confusion matrix index to the actual Enum class value
    :return:
    """
    print("Translating confusion matrix")
    start_timer()
    for row_index in range(confusion_matrix.shape[0]):
        #  Print class prediction accuracy first
        value = confusion_matrix[row_index][row_index]
        print('\n' + enum(index_to_enum_value_mapper(row_index)).name,  # Account for na class in enum
              'pixel prediction accuracy: %.2f%%' % (value*100 / sum(confusion_matrix[row_index])))

        #  Print out incorrectness
        for col_index in range(confusion_matrix.shape[1]):
            if row_index != col_index:
                value = confusion_matrix[row_index][col_index]
                print(enum(index_to_enum_value_mapper(row_index)).name, 'pixels mis-predicted as',
                      enum(index_to_enum_value_mapper(col_index)).name + ': %.2f%%' % (value*100 / sum(confusion_matrix[row_index])))
    end_timer_print_duration()

def print_error_discovery(confusion_matrix, enum, model, x_test, y_test, gt, image_width):
    """
    Interprets & tracks model prediction inaccuracy of a test set onto specific ground truth polygons

    :param confusion_matrix: of which errors are to be located onto ground truth polygons
    :param enum: classification labels
    :param model: predictor that generated the confusion matrix
    :param x_test: test x with original pixel index in first column, that generated the confusion matrix with model
    :param y_test: test y, that generated the confusion matrix with model
    :param gt: ground truth instance with polygons
    :param image_width: original product image width, for polygon lookup
    :return:
    """
    print("Error discovery, locating polgyons")
    start_timer()
    # keys are actual class
    error_polygons = {x: {} for x in filter(lambda y: y != 'na', enum.__members__)}
    for polygon_key in error_polygons:
        # add mistaken class
        error_polygons[polygon_key].update(
            {x: {} for x in filter(lambda y: y != 'na' and y != polygon_key, enum.__members__)})
    for array_idx, pixel in enumerate(x_test):
        pixel_index_in_image = pixel[0]
        predicted_class = model.predict([pixel[1:]])[0]  # remove pixel image index
        actual_class = y_test[array_idx]

        if predicted_class != actual_class:
            polygon_class_name, polygon_index = gt.polygon_lookup(pixel_index=pixel[0], image_width=image_width)
            #  Count number of pixels in each offending polygon
            if not error_polygons[enum(actual_class).name][
                enum(predicted_class).name].__contains__(polygon_index):
                error_polygons[enum(actual_class).name][enum(predicted_class).name][
                    polygon_index] = 1
            else:
                error_polygons[enum(actual_class).name][enum(predicted_class).name][
                    polygon_index] += 1
    for actual_class_name in error_polygons:
        print("Actual %s pixels" % actual_class_name)
        for wrongly_predicted_class_name in error_polygons[actual_class_name]:
            print('    predicted as %s pixels:' % wrongly_predicted_class_name)
            for polygon_index in error_polygons[actual_class_name][wrongly_predicted_class_name]:
                count = error_polygons[actual_class_name][wrongly_predicted_class_name][polygon_index]
                print('        %s_%s' % (actual_class_name, polygon_index), ": %i" % count,
                      '%.2f%%' % (count * 100 / confusion_matrix[enum[actual_class_name].value - 1][
                          enum[wrongly_predicted_class_name].value - 1]))
    end_timer_print_duration()
