import streamlit as st

st.title("AI Travel Concierge Assistant")

st.write("Welcome! Ask travel-related questions and upload travel documents.")

uploaded_file = st.file_uploader("Upload a travel document (PDF or TXT)", type=["pdf", "txt"])

user_input = st.text_input("Ask your travel question:")

if user_input:
    st.write("You asked:", user_input)