# legal_assistant_app.py

import streamlit as st
import re
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from llama_cpp import Llama

# === Load Local LLM ===
llm = Llama(
    model_path="models/gemma/gemma-2-2b-it-Q4_K_M.gguf",
    n_ctx=4096,
    n_batch=512,
    n_gpu_layers=35,
    f16_kv=True,
    temperature=0.2,
    top_k=40,
    top_p=0.95,
    repeat_penalty=1.15,
    max_tokens=2048,
    chat_format="chatml",
    verbose=False
)

# === Prompt Template ===
template = """
You are a highly knowledgeable, articulate, and responsible **AI Legal Assistant** trained on Indian law. Your role is to interpret complex legal matters in a way that is **legally accurate**, **contextually relevant**, and **understandable** to both legal professionals and laypersons.

---

🧑‍⚖️ **INDIAN LEGAL ASSISTANT RESPONSE STRUCTURE**

### 🔹 1. For Laypersons:
- **Purpose**: Provide a plain-language summary of the legal situation and how the law applies.

---

### 🔹 2. Legal Reasoning:
- Explain the **applicable sections of law**, how courts interpret them, and examples.

---

### 🔹 3. Relevant Laws and Provisions:
- 📜 **Section [number]**, *[Act]*: Description

---

### 🔹 4. Authorities and Jurisdiction:
- 🏛️ Courts, 👮 Police, ⚖️ Legal Aid

---

### 🔹 5. Official Contacts & Legal Resources:
- 🖥️ Websites, 📱 Helplines, 🗂️ Portals

---

⚠️ **Disclaimer**: This is general guidance and not legal advice.

---

**Context:**
{context}

**User's Legal Question:**
{question}

---

Answer in the structure above:
"""
prompt = PromptTemplate(input_variables=["context", "question"], template=template)

# === Domain Detection ===
def detect_legal_domain(query):
    query = query.lower()
    if any(w in query for w in ["search", "arrest", "bail", "warrant"]):
        return "CrPC"
    elif any(w in query for w in ["murder", "theft", "assault", "cheating"]):
        return "IPC"
    elif any(w in query for w in ["privacy", "liberty", "equality", "speech"]):
        return "Constitution"
    elif "evidence" in query:
        return "EvidenceAct"
    return "General"

# === Load Vectorstore ===
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L12-v2")
vectorstore = Chroma(persist_directory="chroma_db", embedding_function=embedding)

# === Streamlit App ===
st.set_page_config(page_title="Indian Legal AI", layout="centered")
st.title("🧑‍⚖️ Indian Legal AI Assistant")
st.caption("Powered by Gemma-2B + LangChain + Chroma")

query = st.text_area("Enter your legal question below:")

if st.button("Get Answer"):
    if not query.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Analyzing and generating answer..."):
            domain = detect_legal_domain(query)
            retriever = vectorstore.as_retriever(
                search_kwargs={"k": 5, "filter": {"law": domain}}
            )
            docs = retriever.get_relevant_documents(query)
            context = "\n\n".join(doc.page_content for doc in docs)

            system_prompt = prompt.format(context=context, question=query)

            messages = [{"role": "user", "content": system_prompt}]
            result = llm.create_chat_completion(messages)
            answer = result["choices"][0]["message"]["content"]

            st.subheader("📘 Answer:")
            st.markdown(answer)
            st.success(f"Domain detected: {domain}")
