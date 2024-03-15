import streamlit as st
from Segment_Ingr import getobjects
from classify_ingr import classify_ingr
from PIL import Image
import cv2
import numpy as np

def barcode_scanner_page():
    st.header("Renewable Recipes")
    # Add the content or functionality for this page


def veg_classifier_page():
    st.header("Vegetable Classifier")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    camera_image = st.camera_input("Take a picture")

    if uploaded_file is not None:
        # Convert the uploaded file to an PIL Image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        process_and_classify_image(image)
    elif camera_image is not None:
        # Camera image is already a PIL Image
        st.image(camera_image, caption='Captured Image.', use_column_width=True)
        process_and_classify_image(camera_image)

def process_and_classify_image(image):
    # Convert PIL Image to a numpy array for OpenCV
    image_array = np.array(image)
    image_cv = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
    processed_image = getobjects(image_cv)

    ingredients = classify_ingr(processed_image)
    st.write("Identified Ingredients:", ingredients)

# Include the rest of your pages and the main() function here, without changes needed for this fix

def rotten_classifier_page():
    st.header("Rotten Classifier")
    # Add the content or functionality for this page

def generate_recipe_page():
    st.header("Generate Recipe")
    # Add the content or functionality for this page

def barcode_scanner_page():
    st.header("Barcode Scanner")
    # Add the content or functionality for this page

# Main app
def main():
    # Set the background color to light green and text color to black
    st.markdown("""
    <style>
    .stApp {
      background-color: #e3ffad;
    }
    /* This targets all text within the app to make it black */
    h1, h2, h3, h4, h5, h6, p, .stTextInput>div>div>input, .css-1x0y31y, .css-145kmo2 {
      color: #000000 !important;
    }
    </style>
    """, unsafe_allow_html=True)
    # Main content section
    st.image("Fonallogopng.png", use_column_width=True, caption="Fonallo Logo")

    # Sidebar navigation
    st.sidebar.header("Main Navigation")
    page = st.sidebar.radio("Go to", ("Alex: Veg Classifier", "Husain: Rotten Classifier", "James: Generate Recipe", "George: Barcode Scanner", "Group: Renewable Recipes"))

    if page == "Alex: Veg Classifier":
        veg_classifier_page()
    elif page == "Husain: Rotten Classifier":
        rotten_classifier_page()
    elif page == "James: Generate Recipe":
        generate_recipe_page()
    elif page == "George: Barcode Scanner":
        barcode_scanner_page()
    elif page == "Group: Renewable Recipes":
        rotten_classifier_page()


    # Example button to run rottenCNN and display its result (place this inside the appropriate page function as needed)
    if st.button('Run RottenCNN'):
        st.write("results from RottenCNN will go here...")

if __name__ == "__main__":
    main()
