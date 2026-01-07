import os
import csv
import random
from datetime import datetime
import streamlit as st

from CellProcessor import *

PreprocessVal = CellProcessor.PreprocessVal
PHASE_PATH, GREEN_PATH = CellProcessor.use_Dataset(1)

# Page config
st.set_page_config(layout="wide", page_title="Cell Image Processor")

# Initialize session state
if 'curr_image_num' not in st.session_state:
    st.session_state.curr_image_num = 0
    st.session_state.images = os.listdir(GREEN_PATH)
    st.session_state.len_images = len(st.session_state.images)

# Initialize zoom state
if 'zoom_mode' not in st.session_state:
    st.session_state.zoom_mode = False
    st.session_state.zoom_image = None

# Initialize reset counter
if 'reset_counter' not in st.session_state:
    st.session_state.reset_counter = 0

# Sidebar controls
st.sidebar.header("Image Processing Parameters")

# Create unique keys using reset counter
reset_key = st.session_state.reset_counter

brightness_val = st.sidebar.slider('Brightness', -35.0, 50.0, float(PreprocessVal.brightness), key=f'brightness_{reset_key}')
contrast_val = st.sidebar.slider('Contrast', -5.0, 25.0, float(PreprocessVal.contrast), key=f'contrast_{reset_key}')
threshold_val = st.sidebar.slider('Threshold', 0, 255, int(PreprocessVal.threshold), key=f'threshold_{reset_key}')
i_erode_val = st.sidebar.slider('Erosion', 0, 15, int(PreprocessVal.i_erode), key=f'erosion_{reset_key}')
i_dialate_val = st.sidebar.slider('Dilation', 0, 15, int(PreprocessVal.i_dialate), key=f'dilation_{reset_key}')

# Reset button
if st.sidebar.button('Reset Parameters'):
    st.session_state.reset_counter += 1
    st.rerun()

# Save parameters
saveParameters = st.sidebar.popover("Save Parameters")
saveParamInput = saveParameters.text_input("Description", placeholder="Enter Description here")

if saveParamInput:
    save_time = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    new_row = [save_time, brightness_val, contrast_val, threshold_val, i_erode_val, i_dialate_val, saveParamInput]
    
    with open('params.csv', 'a', newline='') as fd:
        writer = csv.writer(fd)
        writer.writerow(new_row)
    st.sidebar.success('Parameters saved!')

# Image navigation
st.sidebar.header("Image Navigation")
col1, col2, col3 = st.sidebar.columns(3)

with col1:
    if st.button('⬅ Back'):
        if st.session_state.curr_image_num > 0:
            st.session_state.curr_image_num -= 1
            st.rerun()

with col2:
    if st.button('🎲 Random'):
        st.session_state.curr_image_num = random.randint(0, st.session_state.len_images - 1)
        st.rerun()

with col3:
    if st.button('Next ➡'):
        if st.session_state.curr_image_num < st.session_state.len_images - 1:
            st.session_state.curr_image_num += 1
            st.rerun()

st.sidebar.write(f"Image {st.session_state.curr_image_num + 1} / {st.session_state.len_images}")

# Load and process current image
green_image_path = GREEN_PATH + st.session_state.images[st.session_state.curr_image_num]
base_image_path = PHASE_PATH + st.session_state.images[st.session_state.curr_image_num]

st.sidebar.info(f"Current: {st.session_state.images[st.session_state.curr_image_num]}")

# Read and process image
img, img_base = read_image(green_image_path, base_image_path)
img_br, img_thr, img_erosion, img_dilation = process_image(
    img, contrast_val, brightness_val, threshold_val, i_erode_val, i_dialate_val, plot=True
)
img_out = draw_contours(img_dilation, img_base)

if not st.session_state.zoom_mode:
    # Main display
    st.title("Cell Processor")

    # Display images in a grid
    col1, col2, col3 = st.columns(3)

    with col1:
        input_col_title, input_col_zoom = st.columns([3, 1])
        with input_col_title:
            st.subheader("Input")
        with input_col_zoom:
            if st.button("Zoom", key="zoom_input"):
                st.session_state.zoom_mode = True
                st.session_state.zoom_image = "Input"
                st.rerun()
        st.image(img, use_container_width=True, clamp=True)
        
        erosion_col_title, erosion_col_zoom = st.columns([3, 1])
        with erosion_col_title:
            st.subheader("Erosion")
        with erosion_col_zoom:
            if st.button("Zoom", key="zoom_input2"):
                st.session_state.zoom_mode = True
                st.session_state.zoom_image = "Erosion"
                st.rerun()
        st.image(img_erosion, use_container_width=True, clamp=True)

    with col2:
        brightness_col_title, brightness_col_zoom = st.columns([3, 1])
        with brightness_col_title:
            st.subheader("Brightness/Constrast")
        with brightness_col_zoom:
            if st.button("Zoom", key="brightness_zoom_input"):
                st.session_state.zoom_mode = True
                st.session_state.zoom_image = "Brightness/Contrast"
                st.rerun()
        st.image(img_br, use_container_width=True, clamp=True)
        
        dilation_col_title, dilation_col_zoom = st.columns([3, 1])
        with dilation_col_title:
            st.subheader("Dilation")
        with dilation_col_zoom:
            if st.button("Zoom", key="dilation_zoom_input"):
                st.session_state.zoom_mode = True
                st.session_state.zoom_image = "Dilation"
                st.rerun()
        st.image(img_dilation, use_container_width=True, clamp=True)

    with col3:
        threshold_col_title, threshold_col_zoom = st.columns([3, 1])
        with threshold_col_title:
            st.subheader("Threshold")
        with threshold_col_zoom:
            if st.button("Zoom", key="threshold_zoom_input"):
                st.session_state.zoom_mode = True
                st.session_state.zoom_image = "Threshold"
                st.rerun()
        st.image(img_thr, use_container_width=True, clamp=True)
        
        output_col_title, output_col_zoom = st.columns([3, 1])
        with output_col_title:
            st.subheader("Output")
        with output_col_zoom:
            if st.button("Zoom", key="output_zoom_input"):
                st.session_state.zoom_mode = True
                st.session_state.zoom_image = "Output"
                st.rerun()
        st.image(img_out, use_container_width=True, clamp=True)
else:
    # Zoom view
    st.title(f"Cell Image Processor - {st.session_state.zoom_image}")
    
    # Back button
    if st.button("← Back to Gallery"):
        st.session_state.zoom_mode = False
        st.rerun()
    
    # Display the zoomed image based on zoom_image value
    if st.session_state.zoom_image == "Input":
        st.image(img, use_container_width=True, clamp=True)
    elif st.session_state.zoom_image == "Brightness/Contrast":
        st.image(img_br, use_container_width=True, clamp=True)
    elif st.session_state.zoom_image == "Erosion":
        st.image(img_erosion, use_container_width=True, clamp=True)
    elif st.session_state.zoom_image == "Brightness/Contrast":
        st.image(img_br, use_container_width=True, clamp=True)
    elif st.session_state.zoom_image == "Dilation":
        st.image(img_dilation, use_container_width=True, clamp=True)
    elif st.session_state.zoom_image == "Threshold":
        st.image(img_thr, use_container_width=True, clamp=True)
    elif st.session_state.zoom_image == "Output":
        st.image(img_out, use_container_width=True, clamp=True)
