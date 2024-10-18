# Streamlit-RAG-App
#### **Setup Instructions**

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/salushaikh/Streamlit-RAG-App.git
   cd Streamlit-RAG-App

2. **Install Required Libraries**:
   Ensure you have Python 3.8+ installed. Install the required libraries by running:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set Up Pinecone**:
   - Sign up for a Pinecone account at [Pinecone.io](https://www.pinecone.io/).
   - Create a new index in Pinecone and obtain your API key and environment region.
   - Update the `api-key` parameter in the 'app.py' file with your Pinecone API key:
     ```plaintext
     api-key=<your_api_key>
     ```

4. **Run the Application**:
   Launch the Streamlit application by running:
   ```bash
   streamlit run app.py
   ```

#### **Usage Instructions**

1. **Uploading a PDF**:
   - Use the Streamlit UI to upload a PDF file. The contents of the PDF will be displayed in a text area once processed.

2. **Asking a Question**:
   - After uploading the document, input a query related to the document. The system will process the query and return an answer based on the document's content.

3. **Real-Time Response**:
   - The answer to the query will be displayed below the input box, allowing for dynamic interaction with the uploaded document.

#### **Pipeline and Deployment Instructions**

1. **Document Ingestion Pipeline**:
   - When a PDF is uploaded, the text is extracted and split into chunks.
   - Each chunk is embedded using the Hugging Face embedding model and stored in Pinecone with metadata.

2. **Query and Retrieval Pipeline**:
   - A query is embedded and used to search for relevant chunks in Pinecone.
   - The retrieved chunks are used as context for generating a response with the LLaMA model.

3. **Deployment**:
   - The application is deployed using Streamlit and can be hosted on any platform that supports Python. For example, you can deploy it to **Heroku**, **Streamlit Cloud**, or any cloud VM that supports Python and Streamlit.
