from enum import Enum
import utility as util

# Pins file, field indexes for ground truth
class PinsFileRowIndex(Enum):
    x = 1
    y = 2
    lon = 3
    lat = 4
    name = 6


# Pixel classes Enum for supervised learning
class RosebelPixelClass3(Enum):
    na = 0
    water = 1  # 0
    mines = 2  # 1
    forest = 3  # 2


# Contains all ground truth pixel boundaries
class GroundTruthBoundaries:
    def __init__(self, pins_file_path, pixel_class_enum):
        print("Generating ground truth instance from pins file:", pins_file_path, "enum:", pixel_class_enum)
        util.start_timer()
        self.pixel_class_enum = pixel_class_enum
        self.boundaries = {x: {} for x in filter(lambda y: y != 'na', pixel_class_enum.__members__)}
        with open(pins_file_path, 'r') as fd:
            text = fd.read()
            rows = text.split('\n')[6:]  # First 6 rows are comments of the file
            for row in rows:
                fields = row.split('\t')  # Each field is separated by a Tab character

                if len(fields) != 8:
                    continue

                name_fields = fields[PinsFileRowIndex.name.value].split('_')
                class_name = name_fields[0]
                polygon_index = name_fields[1]
                pin_type = name_fields[2]

                if class_name not in list(pixel_class_enum.__members__):
                    continue

                if polygon_index not in self.boundaries[class_name].keys():
                    self.boundaries[class_name][polygon_index] = {}

                self.boundaries[class_name][polygon_index]['x_min' if pin_type == 'TL' else 'x_max'] = fields[
                    PinsFileRowIndex.x.value]
                self.boundaries[class_name][polygon_index]['y_min' if pin_type == 'TL' else 'y_max'] = fields[
                    PinsFileRowIndex.y.value]
        util.end_timer_print_duration()

    def get_boundaries(self):
        return self.boundaries

    def check_pixel_class(self, x: int, y: int):
        for pixel_class in filter(lambda i: i != 'na', self.pixel_class_enum.__members__):
            pixel_class_boundaries = self.boundaries[pixel_class]
            for boundary in pixel_class_boundaries:
                if int(float(pixel_class_boundaries[boundary]['x_min'])) <= x <= int(float(pixel_class_boundaries[boundary]['x_max'])) \
                        and int(float(pixel_class_boundaries[boundary]['y_min'])) <= y <= int(float(pixel_class_boundaries[boundary]['y_max'])):
                    return self.pixel_class_enum[pixel_class]
        return self.pixel_class_enum.__members__['na']

    # Use to generate labels array at runtime
    def get_labels_npy(self, image_width: int, number_of_pixels: int) -> []:
        print("Label npy generation from ground truth instance")
        util.start_timer()
        labels = []
        for i in range(number_of_pixels):
            x = i % image_width
            y = i // image_width
            labels.append(self.check_pixel_class(x, y).value)
        util.end_timer_print_duration()
        return labels

    def get_labels_npy_exclude_some_polygons(self, image_width: int, number_of_pixels: int, polygons_to_exclude: [()]) -> []:
        print("Label npy generation from ground truth instance")
        util.start_timer()
        labels = []
        for i in range(number_of_pixels):
            x = i % image_width
            y = i // image_width

            try:
                pixel_parent_polygon = self.polygon_lookup(x,y)
            except Exception:
                labels.append(self.check_pixel_class(x,y).value) # Can just assign na
            else:
                labels.append(
                    self.check_pixel_class(x, y).value if polygons_to_exclude.count(self.polygon_lookup(x, y)) == 0
                    else self.pixel_class_enum.__members__['na'].value)
        util.end_timer_print_duration()
        return labels

    # either x and y are provided or pixel_index and image_width are provided
    def polygon_lookup(self, x: int = None, y: int = None, pixel_index: int = None, image_width: int = None) -> (str, int):
        if x is None and y is None and pixel_index is None and image_width is None:
            raise Exception('No input provided')

        if pixel_index is not None and image_width is not None:  # Override arguments x, y
            x = pixel_index % image_width
            y = pixel_index // image_width

        for class_name in self.boundaries:
            for polygon_index in self.boundaries[class_name]:
                boundary = self.boundaries[class_name][polygon_index]
                if int(float(boundary['x_min'])) <= x <= int(float(boundary['x_max'])) and int(float(boundary['y_min'])) <= y <= int(float(boundary['y_max'])):
                    return str(class_name), str(polygon_index)

        raise Exception('No polygon found for input x, y')

    def polygon_pixel_count(self, class_name: str, polygon_index: str) -> int:
        polygon = self.boundaries[class_name][polygon_index]
        return (int(float(polygon['x_max'])) - int(float(polygon['x_min']))) * (int(float((polygon['y_max'])) - int(float(polygon['y_min']))))
