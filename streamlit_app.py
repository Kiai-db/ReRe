import streamlit as st # Import the rottenCNN function from rottendetector.py
from Segment_Ingr import getobjects
from classify_ingr import classify_ingr
from PIL import Image
import cv2
def barcode_scanner_page():
    st.header("Renewable Recipes")
    # Add the content or functionality for this page


def veg_classifier_page():
    st.header("Vegetable Classifier")

    # Upload image from files
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    # Take a photo with the system's camera
    camera_image = st.camera_input("Take a picture")

    if uploaded_file is not None:
        # Convert the uploaded file to an PIL Image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        process_and_classify_image(image)
    elif camera_image is not None:
        # Convert the camera image to an PIL Image
        image = Image.open(camera_image)
        st.image(image, caption='Captured Image.', use_column_width=True)
        process_and_classify_image(image)

def process_and_classify_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  
    pil_image_array = getobjects(image)
    
    ingredients = classify_ingr(pil_image_array)
    st.write("Identified Ingredients:", ingredients)

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
    # Set the background color to light green
    st.markdown("""
    <style>
    .stApp {
      background-color: #e3ffad;
    }
    </style>
    """, unsafe_allow_html=True)

    # Sidebar navigation
    st.sidebar.header("Main Navigation")
    page = st.sidebar.radio("Go to", ("Alex: Veg Classifier", "Husain: Rotten Classifier", "James: Generate Recipe", "George: Barcode Scanner","Group: Renewable Recipes"))

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

    # Main content section
    st.image("Fonallogopng.png", use_column_width=True, caption="Fonallo Logo")

    # Example button to run rottenCNN and display its result (place this inside the appropriate page function as needed)
    if st.button('Run RottenCNN'):
        st.write("results from RottenCNN will go here...")

if __name__ == "__main__":
    main()
