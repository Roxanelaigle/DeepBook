import streamlit as st

if "book" not in st.session_state:
    st.error("No book selected. Please go back and enter an ISBN.")
    if st.button("Go Back"):
        st.session_state["page"] = "front"
        st.rerun()
    st.stop()

book = st.session_state["book"]

st.title("Confirm Your Book Selection")

if book:
    st.image(book["image"], caption=book["title"], use_column_width=True)
    st.write(f"**Title:** {book['title']}")
    st.write(f"**Author:** {book['authors']}")

if st.button("Confirm Book"):
    st.session_state["page"] = "display_suggestion"
    st.session_state["confirm_book"] = True  # Use session state to trigger in final_recommendation
    st.rerun()

if st.button("Try Again"):
    st.session_state.clear()
    st.session_state["page"] = "front"
    st.rerun()
