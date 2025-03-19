import streamlit as st
import requests
import numpy as np
from PIL import Image

st.set_page_config(page_title="DeepBook", layout="wide")

# --- HEADER ---
col1, col2 = st.columns([0.15, 0.85])

with col1:
    st.image('front/logo_deepbook.png', width=300)

with col2:
    st.markdown(
        """
        <div style='display: flex; flex-direction: column; justify-content: center; height: 100%;'>
            <h1 style='color:#2B6CB0; font-size: 36px; margin: 0;'>La vie est trop courte pour lire seulement ce qu’on attend de vous.</h1>
        """,
        unsafe_allow_html=True
    )



# --- SECTION 1️⃣ INTRO ---
st.markdown(
    """
    ### 1️⃣ **SCANNEZ VOS LIVRES**
    """,
    unsafe_allow_html=True
)

# Option to take a photo or manually add ISBN
option = st.selectbox('Comment souhaitez-vous fournir les informations ?', ('📷Prendre une photo', '📝Ajouter manuellement un ISBN'))
# Initialisation session state pour proceed_to_step_2
if "proceed_to_step_2" not in st.session_state:
    st.session_state["proceed_to_step_2"] = False

# --- UPLOAD OR ENTER ISBN ---
uploaded_file = None
isbn = None
photo_type = None

# ✅ nouvelle variable

if option == '📷Prendre une photo':
    # ✅ Choix entre un livre ou plusieurs
    photo_type_display = st.selectbox(
        "Quel type de photo souhaitez-vous importer ?",
        ('📖 Un seul livre', '🏛️ Plusieurs livres / Bibliothèque')
    )

    if photo_type_display  == '📖 Un seul livre':
        photo_type = "single"
    else :
        photo_type = "multiple"

    # ✅ Création des colonnes
    col_left, col_right = st.columns([0.6, 0.4])  # Tu peux jouer sur la proportion

    with col_left:
        uploaded_file = st.file_uploader("Téléchargez une photo de votre livre", type=["jpg", "jpeg", "png"])

    with col_right:
        if uploaded_file is not None:
            st.markdown("<div style='text-align: right;'>", unsafe_allow_html=True)
            st.image(uploaded_file, caption='Image téléchargée.', width=150)
            st.markdown("</div>", unsafe_allow_html=True)

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        img_array_rgb = np.array(image)  # Convert image to NumPy array
        st.session_state["proceed_to_step_2"] = True  # ✅ Active dans le bon cas !

elif option == '📝Ajouter manuellement un ISBN':
    isbn_input = st.text_input("Entrez l'ISBN du livre")
    submit_isbn = st.button("✅ Valider l'ISBN")
    photo_type = "isbn"
    if submit_isbn and isbn_input:
        st.session_state["isbn"] = isbn_input
        st.session_state["proceed_to_step_2"] = True

if st.session_state["proceed_to_step_2"]:

    st.markdown(
        """
        ### 2️⃣ EXPLOREZ DE NOUVEAUX HORIZONS
        """,
        unsafe_allow_html=True
    )
    curiosity_level = st.slider(
        "A quel point êtes vous curieux ?",
        min_value=1,
        max_value=4,
        value=1
        )


    API_URL = st.secrets["API_URL"] # needs to be changed based on API!!

    if st.button('Démarrer'):
        st.session_state["curiosity_level"] = curiosity_level
        st.session_state["option"] = option
        st.session_state["photo_type"] = photo_type

        # API Call
        if option == '📷Prendre une photo' and uploaded_file is not None:
            uploaded_file.seek(0)
            file = {"image_array" : (uploaded_file.name, uploaded_file,uploaded_file.type)}
            payload = {"curiosity_level": curiosity_level, "photo_type" : photo_type}  # ✅ Ajout de photo_type
            response = requests.post(API_URL, data =payload, files = file)

        elif option == '📝Ajouter manuellement un ISBN' and "isbn" in st.session_state:
            payload = {"isbn": st.session_state["isbn"], "curiosity_level": curiosity_level, "photo_type" : photo_type}
            response = requests.post(API_URL, data=payload)
        else:
            st.error("Veuillez fournir une image ou un ISBN")
            st.stop()

        # Process API Response
        if response.status_code == 200:
            output_API = response.json()
            input_book = output_API.get("input_book", {})
            output_books = output_API.get("output_books", [])  # Taking only the first recommendation

            st.session_state["input_book"] = input_book
            st.session_state["output_books"] = output_books
            st.session_state["page"] = "show_recommendation"
            st.rerun()
        else:
            st.error("Error while calling the API.")
            st.stop()


# --- PARTIE 3️⃣ ---
    # Show book recommendation immediately after clicking "Start"
    if "page" in st.session_state and st.session_state["page"] == "show_recommendation":
        input_book = st.session_state.get("input_book", {})
        output_books = st.session_state.get("output_books", [])
        option = st.session_state.get("option")
        photo_type= st.session_state.get("photo_type")

        if (option == '📷Prendre une photo' and photo_type == "single") or option == '📝Ajouter manuellement un ISBN' :
            st.markdown("#### Vous avez téléchargé le livre... ")
            col1, col2 = st.columns([0.4, 0.6])

            with col1:
                if input_book.get("image_link", "") != "Non disponible" :
                    url = input_book.get("image_link", "") + '&fife=w1080'
                    st.image(url, width=200)
                else :
                    pass

            with col2:
                st.write(f"**Title:** {input_book.get('title', 'Unknown')}")
                st.write(f"**Author:** {input_book.get('authors', 'Unknown')}")

                # Button 1: "Rerun - I did not upload the book" (Placed next to the sentence)
                if st.button("Relancer - Ce n'est pas le bon livre"):
                    st.session_state.clear()
                    st.rerun()

        st.markdown("### ✨ NOUS VOUS RECOMMANDONS...", unsafe_allow_html=True)

        # ✅ Création de 3 colonnes
        col1, col2, col3 = st.columns(3)

        # ✅ On boucle sur les 3 premières recommandations (ou moins s'il y en a moins)
        columns = [col1, col2, col3]
        for idx, output_book in enumerate(output_books[:3]):  # Limite à 3 recommandations
            with columns[idx]:
                image_link = output_book.get("image_link")

                if image_link and image_link != "Non disponible":
                    url = image_link + '&fife=w1080'
                    st.markdown(
                    f"""
                    <div style="height: 300px; display: flex; justify-content: center; align-items: center; overflow: hidden;">
                        <img src="{url}" style="height: 100%; object-fit: contain;" />
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                title = output_book.get("title", "Unknown")
                author = output_book.get("authors", "Unknown")
                description = output_book.get("description", "No description available")

                st.markdown(f"<strong>{title}</strong>", unsafe_allow_html=True)
                st.markdown(f"<em>{author}</em>", unsafe_allow_html=True)

                # Description avec retour à la ligne et taille réduite
                st.markdown(
                f"""
                <div style="font-size: 12px; margin-top: 8px;">
                    <strong>Description:</strong><br/>
                    <em>{description}</em>
                </div>
                """,
                unsafe_allow_html=True
                )
