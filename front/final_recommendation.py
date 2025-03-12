import streamlit as st

def recommend_book(curiosity_level):
    # Your recommendation logic here
    return f"Recommended book based on curiosity level {curiosity_level}"

def display_suggestion():
    curiosity_level = 1  # Adjust dynamically if needed
    st.write(recommend_book(curiosity_level))  # Display the recommendation

def main():
    if "page" in st.session_state and st.session_state["page"] == "display_suggestion":
        st.title("Your Final Recommendation")

        # Check if the confirmation flag is set
        if "confirm_book" in st.session_state and st.session_state["confirm_book"]:
            display_suggestion()
            del st.session_state["confirm_book"]  # Prevent re-execution on refresh

if __name__ == "__main__":
    main()
