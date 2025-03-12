import streamlit as st
import csv

# Display logo
logo_path = '/Users/niklasfriese/code/nikbd/Roxanelaigle/DeepBook/logo/logo_deepbook.png'
st.image(logo_path, width=140)

# Front end for book suggestion
st.title('DeepBook')

st.markdown(
    """
    ## Scan your book and get a new suggestion
    """
)

# Option to take a photo or manually add ISBN
option = st.selectbox('How would you like to provide the book information?', ('Take a photo', 'Manually add ISBN'))

if option == 'Take a photo':
    uploaded_file = st.file_uploader("Upload a photo of your book", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        st.image(uploaded_file, caption='Uploaded Image.', use_container_width=True)
        st.write("Processing the image...")
        # Here you would add the code to process the image and extract the ISBN
    else:
        if st.button('Take a photo'):
            st.write("This feature is not yet implemented. Please upload a photo instead.")
else:
    isbn = st.text_input('Enter the ISBN of the book')

# Ask for curiosity level
curiosity_level = st.slider('Select your curiosity level', 1, 3)

# Load books from CSV (only for prototype for now!)
def load_books(csv_file):
    books = []
    with open(csv_file, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            books.append({
                'title': row[0],
                'authors': row[1],
                'image': row[14],
                'isbn': row[9]
            })
    return books

books = load_books('/Users/niklasfriese/code/nikbd/Roxanelaigle/DeepBook/data/google_books_consolide_total_final.csv')

# Find book by ISBN
def find_book_by_isbn(isbn, books):
    for book in books:
        if book['isbn'] == isbn:
            return book
    return None

# HIGHLIGHTED CHANGES: Handling navigation without `st.switch_page()`
if st.button('Start'):
    book = find_book_by_isbn(isbn, books) if option == 'Manually add ISBN' else None
    st.session_state['book'] = book
    st.session_state['option'] = option

    # NEW: Set a session state variable to track page navigation
    st.session_state["page"] = "confirm_book"
    st.rerun()  # Force refresh

# NEW: Instead of using `st.switch_page()`, we manually execute `confirm_book.py`
if "page" in st.session_state and st.session_state["page"] == "confirm_book":
    exec(open("front/confirm_book.py").read())  # Runs the confirm_book.py script
    st.stop()  # Prevents any further execution of front.py

elif "page" in st.session_state and st.session_state["page"] == "display_suggestion":
    exec(open("front/final_recommendation.py").read())  # NEW: Ensure final_recommendation runs
    st.stop()  # Prevents execution of front.py after switching pages
