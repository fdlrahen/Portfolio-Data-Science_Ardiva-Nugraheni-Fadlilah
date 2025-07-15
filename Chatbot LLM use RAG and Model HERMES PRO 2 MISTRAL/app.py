import streamlit as st
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from ctransformers import AutoModelForCausalLM
import json
import textwrap
import re

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)  # hilangkan spasi berlebihan
    text = re.sub(r'[^\x00-\x7F\u00A0-\uFFFF]+', '', text)  # hapus karakter non-Unicode normal
    return text.strip()

    
def clean_redundancy(text, max_repeats=3):
    # Hapus kata yang diulang lebih dari max_repeats kali berturut-turut
    def replacer(match):
        word = match.group(1)
        return ' '.join([word] * max_repeats)

    pattern = r'\b(\w+)( \1){' + str(max_repeats) + r',}'
    cleaned = re.sub(pattern, replacer, text)
    return cleaned
# sebelum encode:

# --- Load chunked docs ---
def load_chunks(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

docs = load_chunks("uksw_chunks.json")

# --- Build embeddings + index ---
st.write("⏳ Memproses embeddings dan index.....")

embedder = SentenceTransformer("all-MiniLM-L6-v2")
texts = [doc["text"] for doc in docs]
vectors = embedder.encode(texts, show_progress_bar=True, batch_size=64)

dim = vectors.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(np.array(vectors))

id_map = {i: docs[i] for i in range(len(docs))}

st.write(f"✅ Index siap. Total chunks: {len(docs)}")

# --- Load GGUF model ---
st.write("⏳ Memuat model Hermes 2 Pro...")
model = AutoModelForCausalLM.from_pretrained(
    model_path_or_repo_id="models/Hermes-2-Pro-Mistral-7B.Q4_K_M.gguf",
    model_type="mistral",
    gpu_layers=0
)
st.write("✅ Model siap digunakan!")

# --- Search ---
def search(query, index, id_map, embedder, top_k=5):
    q_emb = embedder.encode([query])
    distances, indices = index.search(q_emb, top_k)
    results = []
    for idx in indices[0]:
        results.append(id_map[idx])
    return results

def find_most_relevant_paragraphs(embedder, query, docs, top_n=3):
    paragraphs = []
    for doc in docs:
        chunks = doc["text"].split("\n\n")
        paragraphs.extend([p.strip() for p in chunks if p.strip()])

    para_embeddings = embedder.encode(paragraphs)
    query_emb = embedder.encode([query])[0]

    similarities = np.dot(para_embeddings, query_emb) / (
        np.linalg.norm(para_embeddings, axis=1) * np.linalg.norm(query_emb)
    )

    top_indices = np.argsort(similarities)[-top_n:][::-1]

    return [paragraphs[i] for i in top_indices]


# --- Generate Answer ---
def generate_answer(prompt, model, window_size=512, default_max=300):
    # Perkiraan jumlah token
    approx_tokens = int(len(prompt) / 4)
    available = window_size - approx_tokens
    max_new_tokens = max(16, min(default_max, available))

    if max_new_tokens < 16:
        return "⚠️ Maaf, konteks dan pertanyaan terlalu panjang untuk dijawab oleh model."

    output = model(
        prompt,
        max_new_tokens=max_new_tokens,
        temperature=0.3,
        top_p=0.9
    )
    return output
def build_prompt(query, top_paragraphs):
    context = "\n\n".join(top_paragraphs)
    if len(context) > 1500:
        context = context[:1500] + "..."

    return f"""Jawablah pertanyaan berikut berdasarkan konteks yang diberikan.

Konteks:
{context}

Pertanyaan:
{query}

Jawaban:"""



st.title("🤖 Chatbot UKSW - Hermes RAG")
query = st.text_area("Tulis pertanyaanmu tentang UKSW di sini:", height=100)
if st.button("Tanya"):
    if query.strip() == "":
        st.warning("Silakan masukkan pertanyaan terlebih dahulu.")
    else:
        # 1. Ambil dokumen top-k berdasarkan FAISS
        top_docs = search(query, index, id_map, embedder, top_k=5)

        # 2. Ambil top-n paragraf paling relevan dari dokumen
        top_paragraphs = find_most_relevant_paragraphs(embedder, query, top_docs, top_n=1)
        context = "\n\n".join(top_paragraphs)

        # 3. Bangun prompt untuk model
        prompt = build_prompt(query, top_paragraphs)

        # 4. Jawaban dari model
        model_answer = generate_answer(prompt, model)
        model_answer = clean_redundancy(model_answer)
        
        # 5. Gabungkan jawaban dengan paragraf teratas
        final_answer = f"{model_answer.strip()}\n\n{top_paragraphs[0].strip()}"
        # 6. Tampilkan ke pengguna
        st.subheader("💡 Jawaban:")
        st.write(final_answer)

        st.markdown("---")
        st.subheader("🔎 Sumber asli:")
        st.info(textwrap.fill(context, width=100))

