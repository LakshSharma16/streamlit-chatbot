# Streamlit Chatbot

This project is a Streamlit-based chatbot that utilizes LangChain components for handling user queries and managing conversation flow. The chatbot is designed to answer questions related to Computer Organization and Architecture by retrieving relevant information from ingested documents.

## Project Structure

```
streamlit-chatbot
├── src
│   ├── app.py          # Main application file for the Streamlit chatbot
│   ├── ingest.py       # Responsible for ingesting documents and creating the FAISS vector store
│   └── utils
│       └── __init__.py # Initialization file for the utils module
├── vector_store         # Directory containing FAISS vector store files
├── .env                 # Environment variables (API keys, etc.)
├── .gitignore           # Files and directories to be ignored by Git
├── requirements.txt     # Python dependencies required for the project
└── README.md            # Documentation for the project
```

## Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd streamlit-chatbot
   ```

2. **Create a virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install the required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables:**
   Create a `.env` file in the root directory and add your API keys and other necessary environment variables.

5. **Run the application:**
   ```bash
   streamlit run src/app.py
   ```

## Usage

- Open the Streamlit app in your web browser.
- Ask questions related to Computer Organization and Architecture.
- The chatbot will retrieve relevant information from the ingested documents and provide answers.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.