# This is basically a copy of object_detection_tutorial.ipynb with some minor
# changes.


from PIL import Image
from collections import Counter
from collections import defaultdict
from io import StringIO
import os
import sys
import tarfile
import zipfile

from matplotlib import pyplot as plt

import numpy as np
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
import tensorflow as tf


flags = tf.app.flags
flags.DEFINE_string('model_checkpoint', '', 'Path to frozen graph.pb file')
flags.DEFINE_string('label_map_path', '', 'Path to label map proto')
flags.DEFINE_string('input_image_glob', '', 'Glob to the image(s) to detect on')
flags.DEFINE_string('output_path', '', 'Path to output')
flags.DEFINE_bool('summary', False, 'If true, just prints summary info')
FLAGS = flags.FLAGS


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


def main(_):
    model_checkpoint = FLAGS.model_checkpoint
    label_map_path = FLAGS.label_map_path
    input_image_glob = FLAGS.input_image_glob
    output_path = FLAGS.output_path

    label_map = label_map_util.load_labelmap(label_map_path)
    NUM_CLASSES = len(label_map.item)
    categories = label_map_util.convert_label_map_to_categories(
        label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(model_checkpoint, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    # Size, in inches, of the output images.
    IMAGE_SIZE = (12, 8)
    TEST_IMAGE_PATHS = tf.gfile.Glob(input_image_glob)

    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            # Definite input and output Tensors for detection_graph
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            # Each box represents a part of the image where a particular object was detected.
            detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')

            for image_path in TEST_IMAGE_PATHS:
                image = Image.open(image_path)
                # the array based representation of the image will be used later in order to prepare the
                # result image with boxes and labels on it.
                try:
                    image_np = load_image_into_numpy_array(image)
                except:
                    print('Failed to load image:', image_path)
                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                image_np_expanded = np.expand_dims(image_np, axis=0)
                # Actual detection.
                (boxes, scores, classes, num) = sess.run(
                    [detection_boxes, detection_scores, detection_classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})

                accepted_indexes = [i for i, x in enumerate(scores[0]) if x > .5]
                det_classes = [category_index[classes[0][i]]['name'] for i in accepted_indexes]
                class_count = Counter(det_classes)

                orbs = [n for n in det_classes if 'orb' in n]
                portraits = [n for n in det_classes if n == 'portrait']
                found_board = 'board' in det_classes

                print('finished', image_path)
                if FLAGS.summary:
                    print('found {} orbs, {} portraits, board={}'.format(
                        len(orbs), len(portraits), found_board))
                    continue

                print()
                print('found {} orbs'.format(len(orbs)))
                print('found {} portraits'.format(len(portraits)))
                print('found board: ', found_board)
                print()

                for item, count in sorted(class_count.items()):
                    print(item, count)

                # Visualization of the results of a detection.
                vis_util.visualize_boxes_and_labels_on_image_array(
                    image_np,
                    np.squeeze(boxes),
                    np.squeeze(classes).astype(np.int32),
                    np.squeeze(scores),
                    category_index,
                    use_normalized_coordinates=True,
                    line_thickness=8,
                    max_boxes_to_draw=100)
                plt.figure(figsize=IMAGE_SIZE)
                plt.imshow(image_np)
                plt.show()


if __name__ == '__main__':
    tf.app.run()
