import os
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader
from langchain_community.document_loaders import TextLoader, Docx2txtLoader

# Function to process documents in a specified directory
def process_documents(directory):
    text = []

    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        file_extension = os.path.splitext(filename)[1]
        
        loader = None
        if file_extension == ".pdf":
            loader = PyPDFLoader(file_path)
        elif file_extension in [".docx", ".doc"]:
            loader = Docx2txtLoader(file_path)
        elif file_extension == ".txt":
            loader = TextLoader(file_path)

        if loader:
            text.extend(loader.load())

    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=100, length_function=len)
    text_chunks = text_splitter.split_documents(text)

    # Create embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )

    # Create vector store
    vector_store = FAISS.from_documents(text_chunks, embedding=embeddings)

    return vector_store

# Main entry point for the ingestion script
if __name__ == '__main__':
    documents_directory = './documents'
    vector_store = process_documents(documents_directory)
    # Persist FAISS index and metadata to local folder
    vector_store.save_local("vector_store")
