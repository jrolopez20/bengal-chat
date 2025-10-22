# --- Dependencies ---
# pip install -r requirements.txt
#
# --- To Run ---
# 1. Create a .env file with your GEMINI_API_KEY (see README.md)
# 2. Run the command: flask run
# 3. Navigate to http://127.0.0.1:5000 in your browser

import os
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv

# LangChain v1.0+ imports
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
script_dir = os.path.dirname(os.path.abspath(__file__))
CHROMA_DB_PATH = os.path.join(script_dir, "chroma_db")
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

# --- Initialization ---
# Serve the static frontend files
app = Flask(__name__, static_folder='static')
CORS(app)

# Configure the Gemini API client
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in environment variables. Please create a .env file.")

print("Loading embedding model...")
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
print("Embedding model loaded.")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

vectorstore = Chroma(
    persist_directory=CHROMA_DB_PATH,
    embedding_function=embeddings
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# --- LLM and RAG Chain (LCEL) ---
print("Initializing LLM and RAG chain...")

# 1. Define the Prompt Template
template = """
You are a helpful assistant for the university's finance department.
Answer the user's question based *only* on the following context provided from the department's website.
If the context does not contain the answer, say "I'm sorry, I don't have that information based on the resources I can access. Please contact the finance department directly."
Do not make up information.

Context:
---
{context}
---

Question: {question}

Answer:
"""
prompt = ChatPromptTemplate.from_template(template)

# 2. Initialize the LLM
model = ChatGoogleGenerativeAI(model="gemini-pro",
                             temperature=0.3)

# 3. Define the Output Parser
output_parser = StrOutputParser()

# 4. Create the RAG chain using LCEL
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# The chain will:
# 1. Pass the user's question to the retriever and as a standalone question.
# 2. The retriever's output (docs) is formatted into a string.
# 3. The formatted context and the question are passed to the prompt.
# 4. The prompt is passed to the model.
# 5. The model's output is parsed into a string.
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | model
    | output_parser
)

print("RAG chain initialized successfully.")

# --- API Endpoints ---

@app.route('/scrape', methods=['POST'])
def scrape_and_embed():
    """
    Scrapes a list of URLs, splits the content, creates embeddings, and stores them.
    Expects JSON payload: {"urls": ["url1", "url2", ...]}
    """
    data = request.get_json()
    if not data or 'urls' not in data:
        return jsonify({"error": "Missing 'urls' in request body"}), 400

    urls = data['urls']
    try:
        print(f"Loading documents from URLs: {urls}")
        loader = WebBaseLoader(urls)
        documents = loader.load()

        print("Splitting documents into chunks...")
        docs_split = text_splitter.split_documents(documents)

        print(f"Adding {len(docs_split)} document chunks to the vector store...")
        vectorstore.add_documents(docs_split)

        print("Scraping and embedding complete.")
        return jsonify({"status": "success", "message": f"Successfully scraped and embedded content from {len(urls)} URLs."})
    except Exception as e:
        print(f"An error occurred during scraping: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/chat', methods=['POST'])
def chat():
    """
    Handles a user's chat message.
    Expects JSON payload: {"query": "user's question"}
    """
    data = request.get_json()
    if not data or 'query' not in data:
        return jsonify({"error": "Missing 'query' in request body"}), 400

    user_query = data['query'].lower()

    # --- Special Keyword Handling ---
    payment_keywords = ["payment", "pay bill", "tuition", "make a payment"]
    if any(keyword in user_query for keyword in payment_keywords):
        payment_link = "https://www.example.com/finance/payment-portal"
        response_text = f"It looks like you're asking about making a payment. You can access the secure payment portal here: <a href='{payment_link}' target='_blank' class='text-blue-400 underline'>{payment_link}</a>"
        return jsonify({"response": response_text})

    # --- RAG Process (now using LCEL) ---
    try:
        print(f"\nInvoking RAG chain for query: '{user_query}'")
        
        # The chain.invoke() method handles the full retrieval, context stuffing, and generation
        llm_answer = rag_chain.invoke(user_query)

        print(f"LLM Answer: {llm_answer}")
        return jsonify({"response": llm_answer})

    except Exception as e:
        print(f"An error occurred during chat processing: {e}")
        return jsonify({"error": str(e)}), 500

# --- Frontend Serving ---
@app.route('/')
def serve_index():
    """Serves the frontend HTML file."""
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    """Serves other static files."""
    return send_from_directory(app.static_folder, path)


if __name__ == '__main__':
    app.run(debug=True, port=5000)

