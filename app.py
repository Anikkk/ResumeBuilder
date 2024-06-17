import streamlit as st
from streamlit_chat import message
from core.core import *

st.title("Resume Modification Tool")

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'modified_resume' not in st.session_state:
    st.session_state.modified_resume = ""

user_input = st.text_input("Enter your query:")

if st.button("Submit"):
    if user_input:
        with st.spinner("Generating response"):
            modified_resume = run_modification_agent(user_input, st.session_state.chat_history)
            st.session_state.chat_history.append((user_input, modified_resume))
            st.session_state.modified_resume = modified_resume
            st.write(modified_resume)

if st.button("Save Resume as DOCX"):
    if st.session_state.modified_resume:
        with st.spinner("Saving the Resume"):
            save_result = run_python_agent(st.session_state.modified_resume, st.session_state.chat_history)
            st.success("Resume saved")
            st.write(save_result)
