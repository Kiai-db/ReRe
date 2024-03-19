import streamlit as st
import time
from Segment_Ingr import getobjects
from rottendetector import rottenCNN
from classify_ingr import classify_ingr
from PIL import Image
import cv2
import numpy as np
import io
from collections import Counter
from generator_basic import recipe_generator
def veg_classifier_page():
    st.header("Vegetable Classifier")
    camera_image = st.camera_input("Take a picture")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None or camera_image is not None:
        # Convert the uploaded file or camera image to a PIL Image
        image = Image.open(uploaded_file) if uploaded_file else camera_image
        # Call the function to process and classify the image
        process_and_classify_image(image)

def process_and_classify_image(image):
    with st.spinner('Processing Ingredients...'):
        image = image.convert('RGB')
        # Convert PIL Image to a numpy array for OpenCV
        image_array = np.array(image)
        image_cv = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        processed_image = getobjects(image_cv)
        ingredients = classify_ingr(processed_image)
    display_classifications(ingredients)

def group_veg_classifier_page():
    st.session_state.ingr_img_classifications = []
    camera_image = st.camera_input("Take a picture")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    individual = st.file_uploader("Upload individual ingredient...", type=["jpg", "jpeg", "png"])
    st.header("Ingredients: ")
    # Initialize the ingredient list in the session state if it doesn't exist
    if 'ingredients_list' not in st.session_state:
        st.session_state.ingredients_list = Counter()

    if uploaded_file is not None or camera_image is not None:
        image = Image.open(uploaded_file) if uploaded_file else camera_image
        ingredients = group_process_and_classify_image(image)
        st.session_state.ingr_img_classifications.append(ingredients)

        # Update the session state with new classifications
        for _, classification in ingredients:
            st.session_state.ingredients_list[classification] += 1

        # Display updated ingredient list
        for ingredient, count in st.session_state.ingredients_list.items():
            st.write(f"{ingredient}: {count}")

    elif individual is not None:
        image = Image.open(individual)
        ingredients = classify_ingr([image])
        for ingredient in ingredients:
            st.session_state.ingr_img_classifications.append(ingredient)

        # Update the session state with new classifications
        for _, classification in ingredients:
            st.session_state.ingredients_list[classification] += 1
        
        # Display updated ingredient list
        for ingredient, count in st.session_state.ingredients_list.items():
            st.write(f"{ingredient}: {count}")
            
    if st.button("Clear Ingredients"):
        st.session_state.ingredients_list = Counter()
    

def group_process_and_classify_image(image):
    with st.spinner('Processing Ingredients...'):
        image = image.convert('RGB')
        # Convert PIL Image to a numpy array for OpenCV
        image_array = np.array(image)
        image_cv = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        processed_image = getobjects(image_cv)
        ingredients = classify_ingr(processed_image)
    
    return ingredients

# Your existing imports and code
def group():
    st.header("Renewable Recipes")
    group_veg_classifier_page()

    if st.button("Process Ingredients"):
        ingr_list = st.session_state.ingr_img_classifications
        print("value: ", ingr_list)
        images = []
        for ingr in ingr_list[0]:
            print("ingr: ", ingr)
            images.append(ingr)
        print("images: ", images)
        result = rottenCNN(images) # Assuming rotten_detector is defined elsewhere
        itera = 0
        for rot_class in result:
            print("rot_class: ", rot_class, "amount: ", len(result))
            print("loop: ", ingr_list[itera][0])
            st.image(ingr_list[itera][0])
            html_str = f"""
            <style>
            .rotten-text {{
                color: black;
                font-size: 30px;
            }}
            </style>
            <div class='rotten-text'>{ingr_list[itera][1], rot_class}</div>
            """
            st.markdown(html_str, unsafe_allow_html=True)
            itera += 1

        difficulty = st.selectbox(
            "Select Difficulty",
            ["Easy", "Medium", "Hard"],
            index=0  # Default to first option
        )

        # User selects the cuisine type
        cuisine = st.selectbox(
            "Select Cuisine Type",
            ["Italian", "Mexican", "Indian", "Chinese", "American", "Others"],
            index=0  # Default to first option
        )
        print("rresult: ", result)
        if st.button("Generate Recipe"):
            recipe = recipe_generator(result, difficulty, cuisine)
            st.write(recipe)


def display_classifications(classified_objects):
    for pil_image, classification in classified_objects:
        if classification == "uncertain":
            continue
        # Convert PIL Image to bytes for display in Streamlit
        buf = io.BytesIO()
        pil_image.save(buf, format="PNG")
        byte_im = buf.getvalue()

        # Display the image
        st.image(byte_im, use_column_width=True)

        # Customize the display of the classification text
        html_str = f"""
        <style>
        .classification-text {{
            color: black;
            font-size: 30px;
        }}
        </style>
        <div class='classification-text'>{classification}</div>
        """
        st.markdown(html_str, unsafe_allow_html=True)


# Include the rest of your pages and the main() function here, without changes needed for this fix

def rotten_classifier_page():
    st.header("Rotten Classifier")
    camera_image = st.camera_input("Take a picture")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None or camera_image is not None:
        image = Image.open(uploaded_file) if uploaded_file else camera_image
        result = rottenCNN([image])
        html_str = f"""
        <style>
        .rotten-text {{
            color: black;
            font-size: 30px;
        }}
        </style>
        <div class='rotten-text'>{result}</div>
        """
        st.markdown(html_str, unsafe_allow_html=True)

def generate_recipe_page():
    st.header("Generate Recipe")

    # User selects the difficulty level
    difficulty = st.selectbox(
        "Select Difficulty",
        ["Easy", "Medium", "Hard"],
        index=0  # Default to first option
    )

    # User selects the cuisine type
    cuisine = st.selectbox(
        "Select Cuisine Type",
        ["Italian", "Mexican", "Indian", "Chinese", "American", "Others"],
        index=0  # Default to first option
    )

    # User inputs ingredients they want to include
    ingredients = st.text_area(
        "Enter Ingredients",
        "List each ingredient on a new line."
    )

    # Button to trigger recipe generation
    if st.button("Generate Recipe"):
        # Split ingredients into a list, assuming each ingredient is on a new line
        ingredients_list = ingredients.split("\n")
        # Remove any empty strings that might result from splitting
        ingredients_list = [ingredient.strip() for ingredient in ingredients_list if ingredient.strip()]
        ingr_ripe = []
        for ingre in ingredients_list:
            ingr_ripe.append((ingre, "ripe"))
        # Call the recipe generation function with the user's inputs
        recipe = recipe_generator(ingr_ripe, difficulty, cuisine)
        
        # Display the generated recipe
        st.write(recipe)

def barcode_scanner_page():
    st.header("Barcode Scanner")
    # Add the content or functionality for this page

# Main app
def main():
    # Set the background color to light green and text color to black
    st.markdown("""
    <style>
        /* Set the app background color */
        .stApp {
            background-color: #e3ffad;
        }
        /* Set the sidebar background color */
        .sidebar .sidebar-content {
            background-color: #7bb82f; 
        }
        /* Set the color of all text within the app */
        h1, h2, h3, h4, h5, h6, p, .stTextInput>div>div>input, .css-1x0y31y, .css-145kmo2 {
            color: #000000 !important;
        }
        /* Set button colors */
        .stButton>button {
            border: 2px solid #4CAF50;
            color: white;
            background-color: #7bb82f;
        }
        /* Modify button hover effect */
        .stButton>button:hover {
            background-color: #4CAF50; /* Darker shade when hovering */
        }
        /* Set color for text input, select, and similar widgets */
        .st-ed, .st-dr, .css-10trblm {
            background-color: #7bb82f;
            color: #ffffff;
        }
        /* Text color for input and select widgets */
        .stTextInput>div>div>input, .stSelectbox>div>div>select {
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
            # Example button to run rottenCNN and display its result (place this inside the appropriate page function as needed)
    elif page == "Husain: Rotten Classifier":
        rotten_classifier_page()
    elif page == "James: Generate Recipe":
        generate_recipe_page()
    elif page == "George: Barcode Scanner":
        barcode_scanner_page()
    elif page == "Group: Renewable Recipes":
        group()





if __name__ == "__main__":
    main()
