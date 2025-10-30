# 🤖 Streamlit Chatbot

An intelligent **Streamlit-based chatbot** built using **LangChain** and **Groq API**.  
This chatbot allows users to interact with their documents in natural language, enabling context-aware question answering with accurate, document-grounded responses.

---

## 🖼️ Project Preview

<img width="1919" height="904" alt="image" src="https://github.com/user-attachments/assets/5336f364-96dc-4445-ac27-7296899627e7" />


## 🚀 Features

- 🧠 Contextual Q&A from uploaded PDFs or documents  
- 💬 Clean, interactive Streamlit interface  
- ⚙️ Powered by **LangChain** and **Groq** for fast and accurate LLM responses  
- 🧩 Integrated with **FAISS** / **ChromaDB** for semantic vector search  
- 📊 Adjustable parameters like temperature, token limit, and number of documents  
- 🔐 `.env` file support for secure API key management  

---

## 🏗️ Project Structure

streamlit-chatbot/
│
├── src/
│ ├── app.py # Streamlit main application
│ ├── ingest.py # Document ingestion and vector storage
│ └── utils/ # Utility functions
│
├── vector_store/ # Stores FAISS/ChromaDB indexes
├── .env # Environment variables (API keys)
├── requirements.txt # Project dependencies
├── README.md # Documentation file



## ⚙️ Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/LakshSharma16/streamlit-chatbot.git
   cd streamlit-chatbot
Create and activate a virtual environment:

python -m venv venv
venv\Scripts\activate       # On Windows
source venv/bin/activate    # On macOS/Linux
Install the dependencies:


pip install -r requirements.txt
Create a .env file in the root directory and add your Groq API key:

GROQ_API_KEY=your_api_key_here

▶️ Run the Chatbot
To start the application, run:

streamlit run src/app.py
Then open the local URL shown in the terminal (usually http://localhost:8501).

📘 Example Usage
Upload a PDF or document.

Ask any question about the content (e.g., "What is pipelining in COA?").

Get accurate and context-based answers instantly.

🧩 Tech Stack
Streamlit – Interactive UI framework

LangChain – For conversational LLM orchestration

Groq API – Fast and efficient model inference

FAISS / ChromaDB – Vector database for embeddings

Sentence Transformers – Embedding generation

Python Dotenv – Secure environment variable management
