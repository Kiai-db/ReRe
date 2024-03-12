import streamlit as st
import base64

def get_base64_of_bin_file(bin_file):
    """Converts binary file to base64 encoded string."""
    with open(bin_file, 'rb') as file:
        content = file.read()
    return base64.b64encode(content).decode()

def set_background_image_from_local_file(path_to_file):
    """Creates CSS to set the background image from a local file converted to base64."""
    bin_str = get_base64_of_bin_file(path_to_file)
    background_image_style = f"""
    <style>
    .stApp {{
      background-image: url("data:image/png;base64,{bin_str}");
      background-size: cover;
    }}
    </style>
    """
    st.markdown(background_image_style, unsafe_allow_html=True)

# Convert the local file to a base64 string and set it as the background
set_background_image_from_local_file("rere.png")

# Adding welcome text with some styling
st.markdown("""
    <style>
    .welcome-text {
        color: white;
        font-size: 48px;
        position: absolute;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
    }
    </style>
    <div class="welcome-text">Welcome</div>
    """, unsafe_allow_html=True)
