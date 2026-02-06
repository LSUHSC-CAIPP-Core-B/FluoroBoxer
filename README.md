# Data preprocessing
Everything connected to working with the dataset.

## Cell Image Projection
`cell_image_projection.py`   
Interactive Streamlit application for viewing cell images and validating processing pipeline outputs. Provides real-time visualization of each processing step with adjustable parameters.

**How to use:**
Select a dataset from the sidebar dropdown to begin processing images. Adjust preprocessing parameters using the sliders for brightness, contrast, threshold, erosion, and dilation. Use the toggle switches to overlay cell boxing on any processing stage. Navigate through images using the Back, Next, or Random buttons. Click "Enlarge" on any image to view it in full-screen mode. Press "Save Parameters" to store your current configuration in `params.csv` with a timestamp and description. The "Reset Parameters" button restores all sliders to their default values.

## Labeling
`label_cell_images.ipynb`    
Iterate over images in the provided directory and for each image crop part of it to gen mask and label (yolo format).
We cover the whole image while croping. The Crop params and path might need to be adjusted for different datasets to match image size and desired overlap of crops.

`get_background_images.ipynb`   
Use this to get background images that do not contain dead cells. It does the exact same process that the previous script does but instead of getting the images with prediction, it looks for the empty ones.

## Augmentation
`augment_cell_iamges.ipynb`   
Augmentations are only a change in brightness and contrast. We augment images, labels, and masks at the same time.

## Spliting
`split_dataset.ipynb`   
Split augmented files into train, val, and test parts.

## Labels formats etc.
`yolo_to_xml.py` and `xml_to_tfrecord.py`   
These files change the format of labels to needed formats (xml, tfrecord).
Specific params and paths might have to be adjusted depending on the dataset.
   
`change_yolo_class_num.py`   
Script to change yolo class numbers in label files.   

## Visualization
`box_rand_images.ipynb`   
Notebook to visualize random images with bounding boxes.   

## CellProcessor
File with variety of functions used along all the data processing. Functions are used exactly for labeling, reading images, converting labels and general visualization.
