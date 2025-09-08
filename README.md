DocuMentor: A Conversational RAG Chatbot
DocuMentor is an intelligent web application that allows you to have a conversation with your documents. Upload a PDF or TXT file, and our AI-powered assistant will answer your questions based on its content, complete with conversational memory for follow-up questions.

Live Application URL: https://mindx-rag-chatbot.onrender.com

📸 Application Preview
This is a placeholder image. Replace with a screenshot of your application.

✨ Key Features
Interactive Chat Interface: A modern and intuitive UI built with Streamlit for a seamless user experience.

Multiple Document Formats: Supports both PDF and TXT file uploads.

Conversational Memory: The chatbot understands the context of the conversation and can answer follow-up questions.

Accurate, Sourced Answers: Built on a Retrieval-Augmented Generation (RAG) architecture to provide answers grounded in the document's content, minimizing inaccuracies.

Cloud-Deployed: Fully deployed and publicly accessible via Render.

🛠️ Technology Stack
Backend: Python, LangChain, Google Gemini API

Frontend: Streamlit

Embedding Model: Hugging Face Sentence Transformers (all-MiniLM-L6-v2)

Vector Store: ChromaDB

Deployment: Render

🏗️ Architecture and Flowchart
The application uses a Retrieval-Augmented Generation (RAG) architecture to ensure that the AI's answers are directly based on the content of the uploaded document.
![alt text](<Workflow Diagram.png>)

System Architecture Diagram
![alt text](<Architecture Diagram.png>)

Data Flowchart
The user interaction is divided into two main phases: Document Processing and Conversational Querying.
This is a placeholder image. Replace with your data flowchart.

🚀 Getting Started
Follow these instructions to set up and run the project locally.

Prerequisites
Python 3.9 or higher

A Google API Key for the Gemini model

1. Clone the Repository
git clone [https://github.com/1603-SUHAIB/Mindx-rag_chatbot.git]
cd Mindx-rag_chatbot

2. Create a Virtual Environment
It's recommended to use a virtual environment to manage dependencies.

# For Windows
python -m venv rag_env
rag_env\Scripts\activate

# For macOS/Linux
python3 -m venv rag_env
source rag_env/bin/activate

3. Install Dependencies
Install all the required packages from the requirements.txt file.

pip install -r requirements.txt

4. Configure Your API Key
Create a new file named .env in the root of the project folder.

Add your Google API key to this file as follows:

GOOGLE_API_KEY="your-google-api-key-goes-here"

5. Run the Application
Launch the Streamlit application from your terminal.

streamlit run app.py

The application should now be running and accessible in your web browser at http://localhost:8501.
