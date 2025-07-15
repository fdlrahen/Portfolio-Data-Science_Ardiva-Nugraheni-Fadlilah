# 🤖 Chatbot LLM with RAG & Hermes Pro 2 Mistral

This project implements a retrieval-augmented generation (RAG) chatbot powered by Hermes Pro 2 Mistral, a transformer-based language model fine-tuned for contextual understanding. The system combines semantic document retrieval with generative response synthesis, enabling dynamic conversations grounded in knowledge sources.

## 📚 Objective
Build a chatbot capable of providing context-rich answers by retrieving relevant chunks from a scraped document corpus, embedding them, and generating responses based on selected documents.

## 🧰 Tech Stack
- SentenceTransformers (`all-MiniLM-L6-v2`) for embedding text chunks
- FAISS for vector similarity search
- LangChain for retrieval orchestration
- Transformers (Hermes Pro 2 via Hugging Face)
- JSON for metadata management

## 📦 Workflow
1. Scrape & clean documents (custom pipeline)
2. Split and chunk text using `RecursiveCharacterTextSplitter`
3. Generate semantic embeddings for each chunk
4. Index vectors using FAISS
5. On user query:
   - Embed query
   - Retrieve top-k similar chunks
   - Feed into Hermes/Mistral for response generation

## 📈 Results
- Fast retrieval across large document corpora (~thousands of chunks)
- High contextual relevance in generated responses
- Ideal for document-based QA or internal assistants

## 🚀 Usage
