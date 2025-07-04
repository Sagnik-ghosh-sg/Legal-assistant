# test_app.py
import sys
print("Python executable:", sys.executable)

import streamlit as st
from llama_cpp import Llama

llm = Llama(model_path="models\gemma\gemma-2-2b-it-Q4_K_M.gguf", n_ctx=512)

st.title("🦙 Local LLM Chatbot")

prompt = st.text_input("Enter your prompt")

if st.button("Generate") and prompt:
    output = llm(prompt, max_tokens=100, stop=["</s>"])
    st.write(output["choices"][0]["text"])
