import streamlit as st
from rottendetector import rottenCNN  # Import the rottenCNN function from rottendetector.py

# Define page content functions
def veg_classifier_page():
    st.header("Vegetable Classifier")
    # Add the content or functionality for this page

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
    page = st.sidebar.radio("Go to", ("Veg Classifier", "Rotten Classifier", "Generate Recipe", "Barcode Scanner"))

    if page == "Veg Classifier":
        veg_classifier_page()
    elif page == "Rotten Classifier":
        rotten_classifier_page()
    elif page == "Generate Recipe":
        generate_recipe_page()
    elif page == "Barcode Scanner":
        barcode_scanner_page()

    # Main content section (if you want a specific action/button here, add it)
    st.image("rere.png", width=100, caption="Navigate using the sidebar.")

    # Example button to run rottenCNN and display its result (place this inside the appropriate page function as needed)
    if st.button('Run RottenCNN'):
        result = rottenCNN()
        st.write(result)

if __name__ == "__main__":
    main()
