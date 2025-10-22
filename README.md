# Bengal Chat

Finance Department RAG Chatbot (LangChain v1.0)

This project is a fully functional, self-contained chatbot for a university's finance department. It uses a Retrieval-Augmented Generation (RAG) architecture built with the modern LangChain Expression Language (LCEL) to answer user questions, ensuring accurate and contextually relevant responses.

Architecture Overview

The chatbot operates on a RAG pipeline:

Data Ingestion: A web scraper (WebBaseLoader) fetches content from specified finance department URLs.

Indexing: The scraped text is split (RecursiveCharacterTextSplitter), converted into vector embeddings (HuggingFaceEmbeddings), and stored in a local ChromaDB vector store.

Retrieval & Generation (LCEL): When a user asks a question, an LCEL chain:

Retrieves the most relevant text chunks from the database.

Formats these chunks and the user's question into a prompt.

Passes the prompt to the Google Gemini Pro model via langchain-google-genai.

Parses the output into a clean string.

Features

Modern RAG Pipeline (LCEL): Provides grounded answers using the latest LangChain standards.

Web-Based UI: A clean, modern chat interface built with HTML and Tailwind CSS.

Easy Data Management: Scrape and update the chatbot's knowledge base with a single API call.

Keyword Detection: Instantly provides a payment link for payment-related queries.

Ready for Deployment: Includes configuration for zero-cost deployment platforms like Render.

Technology Stack

Backend: Python, Flask

Environment Management: python-dotenv

AI Orchestration: LangChain (LCEL)

langchain

langchain-core

langchain-community

langchain-google-genai

LLM: Google Gemini Pro

Vector Database: ChromaDB (runs locally)

Embedding Model: all-MiniLM-L6-v2

Frontend: HTML, Tailwind CSS, JavaScript

Setup and Installation

1. Clone the Repository

git clone <your-repository-url>
cd <repository-name>


2. Create a Python Virtual Environment
It's highly recommended to use a virtual environment to manage dependencies.

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate

# For Windows
python -m venv venv
.\venv\Scripts\activate


3. Install Dependencies

pip install -r requirements.txt


4. Create an Environment File
This project uses a .env file to manage environment variables.

First, get your free Google API key from Google AI Studio.

In the root directory of the project, create a new file named .env.

Add the following lines to the .env file, replacing "YOUR_API_KEY" with your actual key:

FLASK_APP="backend.py"
GOOGLE_API_KEY="YOUR_API_KEY"


Note: The variable name is GOOGLE_API_KEY, which is the default key used by the langchain-google-genai library.

The .gitignore file is already configured to prevent this file from being committed to your repository.

How to Run the Application

1. Start the Backend Server
The .env file will be loaded automatically by Flask. Now, simply run the application:

flask run


The backend server will start on http://127.0.0.1:5000.

2. Populate the Knowledge Base
Before you can chat, you must scrape your website's content. Use a tool like curl or Postman to send a POST request to the /scrape endpoint with the URLs you want to index.

Example using curl:

curl -X POST -H "Content-Type: application/json" \
-d '{"urls": ["[https://www.your-university.edu/finance](https://www.your-university.edu/finance)", "[https://www.your-university.edu/finance/faq](https://www.your-university.edu/finance/faq)"]}' \
[http://127.0.0.1:5000/scrape](http://127.0.0.1:5000/scrape)


This will create a chroma_db directory in your project containing the vectorized data.

3. Launch the Chatbot
Open your web browser and navigate to the backend URL:
http://127.0.0.1:5000

The chat interface will load, and you can start asking questions!