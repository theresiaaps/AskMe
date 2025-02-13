import streamlit as st
import requests

st.title("Ask Me")

url = st.text_input("Enter URL:")
question = st.chat_input("Ask a question:")

if question and url:
    st.chat_message("user").write(question)

    response = requests.post(
        "http://127.0.0.1:8000/ask",  # API FastAPI
        json={"url": url, "question": question}
    )

    if response.status_code == 200:
        answer = response.json().get("answer", "No answer available.")
    else:
        answer = "Error fetching answer from API."

    st.chat_message("assistant").write(answer)
