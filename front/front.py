import streamlit as st
import requests
import numpy as np
from PIL import Image

# Display logo
logo_path = 'front/logo_deepbook.png'
st.image(logo_path, width=140)

# Front-end for book suggestion
st.title('DeepBook')
st.markdown("## Scan your book and get a new suggestion")

# Option to take a photo or manually add ISBN
option = st.selectbox('How would you like to provide the book information?', ('Take a photo', 'Manually add ISBN'))

uploaded_file = None
isbn = None

if option == 'Take a photo':
    uploaded_file = st.file_uploader("Upload a photo of your book", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        st.image(uploaded_file, caption='Uploaded Image.', width=200)
        image = Image.open(uploaded_file)
        img_array_rgb = np.array(image)  # Convert image to NumPy array
    else:
        st.write("Please upload a photo to proceed.")
else:
    isbn = st.text_input('Enter the ISBN of the book')

# Ask for curiosity level
curiosity_level = st.slider('Select your curiosity level', 1, 3)

API_URL = "http://127.0.0.1:8000" # needs to be changed based on API!!

if st.button('Start'):
    st.session_state["curiosity_level"] = curiosity_level
    st.session_state["option"] = option

    # API Call
    if option == "Take a photo" and uploaded_file is not None:
        payload = {"image_array": img_array_rgb.tolist(), "curiosity_level": curiosity_level}
        response = requests.post(API_URL, json=payload)
    elif option == "Manually add ISBN" and isbn:
        payload = {"isbn": isbn, "curiosity_level": curiosity_level}
        response = requests.post(API_URL, json=payload)
    else:
        st.error("Please provide an image or an ISBN.")
        st.stop()

    # Process API Response
    if response.status_code == 200:
        output_API = response.json()
        input_book = output_API.get("input_book", {})
        output_book = output_API.get("output_books", [])[0]  # Taking only the first recommendation

        st.session_state["input_book"] = input_book
        st.session_state["output_book"] = output_book
        st.session_state["page"] = "show_recommendation"
        st.rerun()
    else:
        st.error("Error while calling the API.")
        st.stop()

# Show book recommendation immediately after clicking "Start"
if "page" in st.session_state and st.session_state["page"] == "show_recommendation":
    input_book = st.session_state.get("input_book", {})
    output_book = st.session_state.get("output_book", {})

    st.markdown(
        f"You have uploaded the book **{input_book.get('title', 'Unknown')}** by **{input_book.get('authors', 'Unknown')}**."
    )

    # Button 1: "Rerun - I did not upload the book" (Placed next to the sentence)
    if st.button("Rerun - I did not upload the book"):
        st.session_state.clear()
        st.rerun()

    st.markdown("### Based on your upload, we recommend the following book:")


    if output_book:
        url = output_book.get("image")+ '&fife=w1080'
        st.image(url, caption=output_book.get("title"), width=200)

        st.write(f"**Title:** {output_book.get('title')}")
        st.write(f"**Author:** {output_book.get('authors')}")
        st.write(f"**Description:** {output_book.get('description')}")

    # Button 2: "Suggest Another Book"
    if st.button("Suggest Another Book"):
        st.session_state["page"] = "show_recommendation"
        st.rerun()
