from llama_cpp import Llama
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# === Step 1: Load and preprocess documents ===
loader_pdf = DirectoryLoader("data/legal_docs", glob="*.pdf", loader_cls=PyPDFLoader)
documents = loader_pdf.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
splits = text_splitter.split_documents(documents)

def clean_text(text):
    return re.sub(r'\s+', ' ', text.strip().lower())

unique_chunks = {}
for chunk in splits:
    cleaned = clean_text(chunk.page_content)
    if cleaned not in unique_chunks:
        unique_chunks[cleaned] = chunk
unique_chunks = list(unique_chunks.values())

# === Step 2: Embedding & Vector Store ===
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L12-v2")
vectorstore = Chroma.from_documents(unique_chunks, embedding=embedding, persist_directory="chroma_db")
vectorstore.persist()
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# === Step 3: Load GGUF Gemma-2B model correctly ===
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
    chat_format="chatml",  # CORRECTED!
    verbose=True
)

# === Step 4: Define Prompt ===
prompt_template = """
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

prompt = PromptTemplate(input_variables=["context", "question"], template=prompt_template)

# === Step 5: Similarity Check Functions ===
def is_redundant_sentence(sentence, seen):
    norm = clean_text(sentence)
    return not norm or norm in seen or len(norm) < 5

def score_sentence_similarity(sentence, sentences, threshold=0.9):
    if not sentences:
        return False
    try:
        new_embed = embedding.embed_documents([sentence])[0]
        existing_embeds = embedding.embed_documents(sentences)
        sim_scores = cosine_similarity([new_embed], existing_embeds)[0]
        if len(sentence.strip().split()) <= 3:
            return True
        return any(score > threshold for score in sim_scores)
    except Exception:
        return False

# === Step 6: Stateless Query Loop ===
while True:
    query = input("Ask a legal question (or type 'exit'): ")
    if query.lower() == 'exit':
        break

    docs = retriever.get_relevant_documents(query)
    context = "\n\n".join([doc.page_content for doc in docs])

    system_prompt = prompt.format(context=context, question=query)

    messages = [
        {"role": "user", "content": system_prompt}
    ]

    result = llm.create_chat_completion(messages)
    answer = result["choices"][0]["message"]["content"]

    seen = set()
    final_sentences = []
    sentences = re.split(r'(?<=[.!?])\s+', answer)

    for sentence in sentences:
        if is_redundant_sentence(sentence, seen):
            continue
        if not score_sentence_similarity(sentence, final_sentences, threshold=0.9):
            seen.add(clean_text(sentence))
            final_sentences.append(sentence.strip())

    final_answer = " ".join(final_sentences)
    print("\nAnswer:\n", final_answer)
    print("\n---\n")
