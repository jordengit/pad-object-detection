# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
#
# This is a modified version of the original object_detction script:
# https://github.com/tensorflow/models/blob/master/research/object_detection/dataset_tools/create_pascal_tf_record.py
#
# I think I also cribbed from some other tfrecord creation script but I don't
# recall which one.
#
# Changes:
# 1) Removed VOC dataset support
# 2) PNG support
# 3) Accept arbitrary data directory
# 4) Create validation dataset as a subset of training data
# 5) Auto scale down images to a maximum dimension, maintaining aspect ratio

r"""Convert images/annotations to TFRecord for object_detection.

Example usage:
    python scripts/create_pascal_tf_record_generic.py \
        --data_dir=images \
        --annotations_dir=annotations \
        --output_path=training/inputs/orb_data \
        --label_map_path=training/inputs/orb_data/orb_label_map.pbtxt
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import PIL.Image
import hashlib
import io
import logging
from lxml import etree
import os
import random

from object_detection.utils import dataset_util
from object_detection.utils import label_map_util
import tensorflow as tf


flags = tf.app.flags
flags.DEFINE_string('data_dir', 'images',
                    '(Relative) path to images directory')
flags.DEFINE_string('annotations_dir', 'annotations',
                    '(Relative) path to annotations directory')
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
flags.DEFINE_string('label_map_path', '', 'Path to label map proto')
flags.DEFINE_boolean('ignore_difficult_instances', False, 'Whether to ignore '
                     'difficult instances')
FLAGS = flags.FLAGS

RANDOM_SEED = 4242
VALIDATION_PCT = .1
MAX_IMG_DIM = 1024.0


def dict_to_tf_example(data,
                       data_dir,
                       label_map_dict,
                       ignore_difficult_instances=False):
    """Convert XML derived dict to tf.Example proto.

    Notice that this function normalizes the bounding box coordinates provided
    by the raw data.

    Args:
      data: dict holding PASCAL XML fields for a single image (obtained by
        running dataset_util.recursive_parse_xml_to_dict)
      data_dir: Path to directory containing images
      label_map_dict: A map from string label names to integers ids.
      ignore_difficult_instances: Whether to skip difficult instances in the
        dataset  (default: False).

    Returns:
      example: The converted tf.Example.

    Raises:
      ValueError: if the image pointed to by data['filename'] is not a valid JPEG/PNG
    """
    full_path = os.path.join(data_dir, data['filename'])

    with tf.gfile.GFile(full_path, 'rb') as fid:
        encoded_image = fid.read()
    encoded_image_io = io.BytesIO(encoded_image)
    image = PIL.Image.open(encoded_image_io)
    image_format = image.format
    if image_format not in ('JPEG', 'PNG'):
        raise ValueError('Image format not JPEG/PNG')
    key = hashlib.sha256(encoded_image).hexdigest()

    max_dim = float(max(image.size[0], image.size[1]))
    if max_dim > MAX_IMG_DIM:
        scale_factor = 1.0 - (max_dim - MAX_IMG_DIM) / max_dim
        if image.size[0] > image.size[1]:
            new_width = MAX_IMG_DIM
            new_height = image.size[1] * scale_factor
        else:
            new_width = image.size[0] * scale_factor
            new_height = MAX_IMG_DIM

        image = image.resize((int(new_width), int(new_height)), PIL.Image.ANTIALIAS)
        encoded_image_io = io.BytesIO()
        image.save(encoded_image_io, image_format)
        encoded_image = encoded_image_io.getvalue()

    width = int(data['size']['width'])
    height = int(data['size']['height'])

    xmin = []
    ymin = []
    xmax = []
    ymax = []
    classes = []
    classes_text = []
    truncated = []
    poses = []
    difficult_obj = []
    if 'object' in data:
        for obj in data['object']:
            difficult = bool(int(obj['difficult']))
            if ignore_difficult_instances and difficult:
                continue

            class_text = obj['name']
            if class_text not in label_map_dict:
                # We're not interested in outputting this class
                continue
            class_value = label_map_dict[class_text]

            difficult_obj.append(int(difficult))

            xmin.append(float(obj['bndbox']['xmin']) / width)
            ymin.append(float(obj['bndbox']['ymin']) / height)
            xmax.append(float(obj['bndbox']['xmax']) / width)
            ymax.append(float(obj['bndbox']['ymax']) / height)
            classes_text.append(class_text.encode('utf8'))
            classes.append(class_value)
            truncated.append(int(obj['truncated']))
            poses.append(obj['pose'].encode('utf8'))
    else:
        print('warning, no objects found in {}'.format(full_path))

    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(image.size[1]),
        'image/width': dataset_util.int64_feature(image.size[0]),
        'image/filename': dataset_util.bytes_feature(
            data['filename'].encode('utf8')),
        'image/source_id': dataset_util.bytes_feature(
            data['filename'].encode('utf8')),
        'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
        'image/encoded': dataset_util.bytes_feature(encoded_image),
        'image/format': dataset_util.bytes_feature(image_format.lower().encode('utf8')),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmin),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmax),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymin),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymax),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
        'image/object/difficult': dataset_util.int64_list_feature(difficult_obj),
        'image/object/truncated': dataset_util.int64_list_feature(truncated),
        'image/object/view': dataset_util.bytes_list_feature(poses),
    }))
    return example


def main(_):
    image_dir = FLAGS.data_dir
    output_path = FLAGS.output_path
    annotations_dir = FLAGS.annotations_dir
    annotation_files = tf.gfile.Glob(os.path.join(annotations_dir, '*.xml'))

    train_file = os.path.join(output_path, 'train.tfrecord')
    val_file = os.path.join(output_path, 'val.tfrecord')
    train_writer = tf.python_io.TFRecordWriter(train_file)
    val_writer = tf.python_io.TFRecordWriter(val_file)

    label_map_dict = label_map_util.get_label_map_dict(FLAGS.label_map_path)

    logging.info('Processing %s images.', len(annotation_files))

    random.seed(RANDOM_SEED)
    random.shuffle(annotation_files)

    for idx, annotation_file in enumerate(annotation_files):
        print(annotation_file)
        if idx % 100 == 0:
            logging.info('On image %d of %d', idx, len(annotation_files))

        is_validation_img = idx < len(annotation_files) * VALIDATION_PCT

        with tf.gfile.GFile(annotation_file, 'r') as fid:
            xml_str = fid.read()
        xml = etree.fromstring(xml_str)
        data = dataset_util.recursive_parse_xml_to_dict(xml)['annotation']

        tf_example = dict_to_tf_example(data, FLAGS.data_dir, label_map_dict,
                                        FLAGS.ignore_difficult_instances)

        if is_validation_img:
            val_writer.write(tf_example.SerializeToString())
        else:
            train_writer.write(tf_example.SerializeToString())

    val_writer.close()
    train_writer.close()


if __name__ == '__main__':
    tf.app.run()
