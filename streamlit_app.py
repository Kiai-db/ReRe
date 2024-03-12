import streamlit as st

# CSS to inject contained in a multiline string
background_image_style = """
<style>
.stApp {
  background-image: url("rere.png");
  background-size: cover;
}
</style>
"""

st.markdown(background_image_style, unsafe_allow_html=True)  # Using the background image

# Centering the text over the image
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
