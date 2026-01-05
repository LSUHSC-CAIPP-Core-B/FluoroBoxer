import os
import csv
import random
from datetime import datetime
import streamlit as st

from config import *
from CellProcessor import read_image, process_image, draw_contours

# Page config
st.set_page_config(layout="wide", page_title="Cell Image Processor")

# Initialize session state
if 'curr_image_num' not in st.session_state:
    st.session_state.curr_image_num = 0
    st.session_state.images = os.listdir(GREEN_PATH)
    st.session_state.len_images = len(st.session_state.images)

# Initialize reset counter
if 'reset_counter' not in st.session_state:
    st.session_state.reset_counter = 0

# Sidebar controls
st.sidebar.header("Image Processing Parameters")

# Create unique keys using reset counter
reset_key = st.session_state.reset_counter

brightness_val = st.sidebar.slider('Brightness', -35.0, 50.0, float(brightness), key=f'brightness_{reset_key}')
contrast_val = st.sidebar.slider('Contrast', -5.0, 25.0, float(contrast), key=f'contrast_{reset_key}')
threshold_val = st.sidebar.slider('Threshold', 0, 255, int(threshold), key=f'threshold_{reset_key}')
i_erode_val = st.sidebar.slider('Erosion', 0, 15, int(i_erode), key=f'erosion_{reset_key}')
i_dialate_val = st.sidebar.slider('Dilation', 0, 15, int(i_dialate), key=f'dilation_{reset_key}')

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

# Main display
st.title("Cell Image Processor")

# Display images in a grid
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Input")
    st.image(img, use_container_width=True, clamp=True)
    
    st.subheader("Erosion")
    st.image(img_erosion, use_container_width=True, clamp=True)

with col2:
    st.subheader("Brightness/Contrast")
    st.image(img_br, use_container_width=True, clamp=True)
    
    st.subheader("Dilation")
    st.image(img_dilation, use_container_width=True, clamp=True)

with col3:
    st.subheader("Threshold")
    st.image(img_thr, use_container_width=True, clamp=True)
    
    st.subheader("Output")
    st.image(img_out, use_container_width=True, clamp=True)
