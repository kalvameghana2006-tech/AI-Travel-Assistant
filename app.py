import streamlit as st

st.set_page_config(page_title="AI Travel Assistant")

st.title("🌍 AI Travel Concierge Assistant")

st.write("Plan your trips with AI!")

uploaded_file = st.file_uploader(
    "Upload travel document",
    type=["pdf","txt"]
)

user_question = st.text_input("Ask a travel question")

if user_question:
    st.write("You asked:", user_question)