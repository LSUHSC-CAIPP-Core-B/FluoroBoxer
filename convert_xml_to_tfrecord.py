"""
TensorFlow XML-to-TFRecord Converter

This module converts Pascal VOC XML annotations to TensorFlow Record (TFRecord) format,
suitable for training object detection models with TensorFlow 1.x.

Usage:
    python3 xml_to_tfrecord.py -x XML_DIR -l LABELS_PATH -o OUTPUT_PATH [-i IMAGE_DIR] [-c CSV_PATH]

Arguments:
    -h, --help            Show this help message and exit
    -x XML_DIR, --xml_dir XML_DIR
                          Path to directory containing input .xml files (required)
    -l LABELS_PATH, --labels_path LABELS_PATH
                          Path to labels (.pbtxt) file (required)
    -o OUTPUT_PATH, --output_path OUTPUT_PATH
                          Path for output TFRecord (.record) file (required)
    -i IMAGE_DIR, --image_dir IMAGE_DIR
                          Path to directory containing images. Defaults to XML_DIR if not specified
    -c CSV_PATH, --csv_path CSV_PATH
                          Path for optional output CSV file. If not provided, no CSV will be written

Example:
    python xml_to_tfrecord.py -x ./annotations -l labels.pbtxt -o dataset.record -i ./images
"""

import os
import glob
import pandas as pd
import io
import xml.etree.ElementTree as ET
import argparse

# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow.compat.v1 as tf
from PIL import Image
from object_detection.utils import dataset_util, label_map_util
from collections import namedtuple

# Initialize argument parser
parser = argparse.ArgumentParser(
    description="Convert Pascal VOC XML annotations to TensorFlow Record format")
parser.add_argument(
    "-x", "--xml_dir",
    help="Path to directory containing input .xml files (required)",
    type=str, required=True)
parser.add_argument(
    "-l", "--labels_path",
    help="Path to labels (.pbtxt) file (required)", 
    type=str, required=True)
parser.add_argument(
    "-o", "--output_path",
    help="Path for output TFRecord (.record) file (required)", 
    type=str, required=True)
parser.add_argument(
    "-i", "--image_dir",
    help="Path to directory containing images. Defaults to XML_DIR if not specified",
    type=str, default=None)
parser.add_argument(
    "-c", "--csv_path",
    help="Path for optional output CSV file. If not provided, no CSV will be written",
    type=str, default=None)

args = parser.parse_args()

# Use XML directory as image directory if not specified
if args.image_dir is None:
    args.image_dir = args.xml_dir

# Load label map
label_map = label_map_util.load_labelmap(args.labels_path)
label_map_dict = label_map_util.get_label_map_dict(label_map)


def xml_to_csv(path):
    """
    Convert Pascal VOC XML annotations to Pandas DataFrame.
    
    Iterates through all .xml files in the given directory and extracts bounding box
    information from each object annotation, combining them into a single DataFrame.
    
    Parameters:
        path (str): Directory path containing .xml files (Pascal VOC format)
    
    Returns:
        pd.DataFrame: DataFrame with columns [filename, width, height, class, xmin, ymin, xmax, ymax]
    
    Raises:
        FileNotFoundError: If the specified path does not exist
        ET.ParseError: If an XML file is malformed
    """
    xml_list = []
    
    for xml_file in glob.glob(path + '/*.xml'):
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            
            for member in root.findall('object'):
                value = (
                    root.find('filename').text,
                    int(root.find('size')[0].text),
                    int(root.find('size')[1].text),
                    member[0].text,
                    int(member[4][0].text),
                    int(member[4][1].text),
                    int(member[4][2].text),
                    int(member[4][3].text)
                )
                xml_list.append(value)
        except ET.ParseError as e:
            print(f"Warning: Could not parse {xml_file}: {e}")
            continue
    
    column_names = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_names)
    return xml_df


def class_text_to_int(row_label):
    """
    Convert class label text to integer class ID.
    
    Parameters:
        row_label (str): Class label text
    
    Returns:
        int: Corresponding class ID from label map
    """
    return label_map_dict[row_label]


def split(df, group):
    """
    Group DataFrame by a specified column.
    
    Parameters:
        df (pd.DataFrame): Input DataFrame
        group (str): Column name to group by
    
    Returns:
        list: List of named tuples containing (filename, grouped_data)
    """
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]


def create_tf_example(group, path):
    """
    Create a TensorFlow Example protobuf from image and annotation data.
    
    Converts image and bounding box coordinates into TensorFlow Example format
    suitable for object detection model training.
    
    Parameters:
        group (namedtuple): Contains filename and grouped annotation data
        path (str): Path to directory containing images
    
    Returns:
        tf.train.Example: TensorFlow Example protobuf with image and annotation features
    """
    with tf.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:
        encoded_jpg = fid.read()
    
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    filename = group.filename.encode('utf8')
    image_format = b'png'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    # Process each object annotation in the image
    for index, row in group.object.iterrows():
        # Normalize bounding box coordinates
        xmins.append(row['xmin'] / width)
        xmaxs.append(row['xmax'] / width)
        ymins.append(row['ymin'] / height)
        ymaxs.append(row['ymax'] / height)
        classes_text.append(row['class'].encode('utf8'))
        classes.append(class_text_to_int(row['class']))

    # Create TensorFlow Example
    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example


def main(_):
    """
    Main entry point for TFRecord generation.
    
    Orchestrates the conversion process:
    1. Parse XML annotations into DataFrame
    2. Group by filename
    3. Create TensorFlow Examples
    4. Write to TFRecord file
    5. Optionally export to CSV
    """
    print("Starting TFRecord conversion...")
    
    # Initialize TFRecord writer
    writer = tf.python_io.TFRecordWriter(args.output_path)
    path = os.path.join(args.image_dir)
    
    # Convert XML to CSV format
    print(f"Reading XML files from: {args.xml_dir}")
    examples = xml_to_csv(args.xml_dir)
    print(f"Loaded {len(examples)} annotations")
    
    # Group by filename and create TF Examples
    grouped = split(examples, 'filename')
    print(f"Processing {len(grouped)} images...")
    
    for idx, group in enumerate(grouped):
        tf_example = create_tf_example(group, path)
        writer.write(tf_example.SerializeToString())
        
        if (idx + 1) % 50 == 0:
            print(f"  Processed {idx + 1}/{len(grouped)} images")
    
    writer.close()
    print(f"✓ Successfully created TFRecord: {args.output_path}")
    
    # Optionally save to CSV
    if args.csv_path is not None:
        examples.to_csv(args.csv_path, index=None)
        print(f"✓ Successfully created CSV: {args.csv_path}")


if __name__ == '__main__':
    tf.app.run()