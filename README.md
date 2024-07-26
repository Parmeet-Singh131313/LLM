# PDF Chatbot with Gemini Pro
This project creates a chatbot capable of interacting with multiple PDF files using Google Gemini Pro's API. The chatbot can understand user queries, extract relevant information from the uploaded PDFs, and provide accurate and coherent responses.
# Table of Contents
1. Features
2. Installation
3. Usage
4. Configuration
5. File Structure
6. Contributing
7. License
# Features
1. Extracts text from multiple PDF files.
2. Processes and splits text into manageable chunks.
3. Creates vector embeddings of the text using Google Generative AI.
4. Uses FAISS for efficient similarity search.
5. Answers user queries based on the content of the uploaded PDFs.
6. Simple and intuitive web interface built with Streamlit.
# Installation
1. Clone the repository:
git clone https://github.com/yourusername/pdf-chatbot.git
cd pdf-chatbot

2. Create and activate a virtual environment:
python -m venv venv
source venv/bin/activate

3. Install the dependencies:
pip install -r requirements.txt

4. Set up environment variables:
Create a .env file in the root directory of the project and add your Google API key:
GOOGLE_API_KEY=your_google_api_key

# Usage
1. Run the Streamlit app:
streamlit run app.py

2. Upload PDF files:
Go to the sidebar, click on "Upload the file", and select multiple PDF files to upload.

3. Submit the PDFs:
Click the "Submit" button to process and store the PDFs in the vector database.

4. Ask a Question:
   1. Type your question in the input field and press Enter.
   2. The chatbot will display the answer based on the content of the uploaded PDFs.

# Configuration
Google API Key: This is required to use Google Generative AI services. Obtain the API key from the Google Cloud Console and add it to the .env file as shown above.

# File Structure
pdf-chatbot/
├── app.py
├── requirements.txt
├── .env.example
├── templates/
│   └── index.html
└── README.md

# Contributing
Contributions are welcome! Please read the contributing guidelines first.

# License
This project is licensed under the MIT License
