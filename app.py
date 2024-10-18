import streamlit as st
from PyPDF2 import PdfReader
import os
import tqdm
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from lightrag.core.generator import Generator
from lightrag.core.component import Component
from lightrag.core.model_client import ModelClient
from lightrag.components.model_client import OllamaClient

# Hugging Face embedding model for document embeddings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Hugging Face embedding model initialization
embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    device=0
)

# Pinecone setup
pc = Pinecone(api_key="your-pinecone-api-key")
if 'my-index' not in pc.list_indexes().names():
    pc.create_index(
      name='my-index',
      dimension=384,
      metric='euclidean',
      spec=ServerlessSpec(
          cloud='aws',
          region='us-east-1'
        )
    )

# Device setup for model inference
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Function to ingest PDF and add to Pinecone
def ingest_pdf_to_pinecone(pdf_content, chunk_size=512, chunk_overlap=50):
    # Split PDF content into chunks
    text_chunks = [pdf_content[i:i + chunk_size] for i in range(0, len(pdf_content), chunk_size)]

    # Embed and upsert to Pinecone
    pinecone_index = pc.Index("my-index", host="your-pinecone-index-host-url")
    for i, chunk in enumerate(text_chunks):
        embedding = embed_model.encode(chunk)
        metadata = {
            "text": chunk,
        }
        # Upsert into Pinecone with unique IDs for each chunk
        pinecone_index.upsert([(f"pdf_chunk_{i}", embedding.tolist(), metadata)])

# Function to generate text response
def generate_text(prompt):
    qa_template = r"""<SYS>
    You are a Q&A assistant. Your main goal is to provide accurate answers based on the provided context.
    </SYS>
    User: {{input_str}}
    You:"""

    class SimpleQA(Component):
        def __init__(self, model_client: ModelClient, model_kwargs: dict):
            super().__init__()
            self.generator = Generator(
                model_client=model_client,
                model_kwargs=model_kwargs,
                template=qa_template,
            )

        def call(self, input: dict) -> str:
            return self.generator.call({"input_str": str(input)})

        async def acall(self, input: dict) -> str:
            return await self.generator.acall({"input_str": str(input)})

    model = {
        "model_client": OllamaClient(),
        "model_kwargs": {"model": "llama3.1:8b"},
    }
    qa = SimpleQA(**model)
    output = qa(f"{prompt}")
    response = output.data

    return response

# Load a pre-trained model for embeddings
embed_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Function to handle query and retrieve relevant document chunks
def handle_query(query):
    # Generate embedding for the query using the sentence transformer model
    query_embedding = embed_model.encode(query, convert_to_tensor=True)

    # Query Pinecone for relevant document chunks
    pinecone_index = pc.Index("my-index", host="your-pinecone-index-host-url")
    pinecone_response = pinecone_index.query(
        vector=query_embedding.tolist(),
        top_k=5,
        include_metadata=True
    )

    # Extract the most relevant text chunks from the response
    context_strs = []
    for match in pinecone_response["matches"]:
        context_strs.append(match["metadata"]["text"])

    # Combine the retrieved contexts into one string
    combined_context_str = "\n\n".join(context_strs)

    # Format the prompt for answer generation
    full_prompt = f"""<SYS>
    You are a Q&A assistant. Your main goal is to provide accurate answers based on the provided context.
    </SYS>
    Context: {combined_context_str}
    Question: {query}
    You:"""

    # Generate the answer using your text generation function
    answer = generate_text(full_prompt)

    return answer, context_strs

# Streamlit App
st.sidebar.title("RAG QA Bot Parameters")
uploaded_file = st.sidebar.file_uploader("Upload your PDF", type="pdf")
chunk_size = st.sidebar.slider("Chunk Size", 100, 2000, 512)
chunk_overlap = st.sidebar.slider("Chunk Overlap", 0, 500, 50)

st.title("Retrieval-Augmented QA Bot")
st.markdown("Upload a document and ask questions. The app will retrieve relevant document segments and generate answers.")

# PDF processing
if uploaded_file:
    reader = PdfReader(uploaded_file)
    text = "\n".join([page.extract_text() for page in reader.pages])
    st.text_area("Uploaded Document Content", value=text, height=300)

    # Ingest the uploaded PDF into Pinecone
    ingest_pdf_to_pinecone(text, chunk_size, chunk_overlap)

    # Query input
    query = st.text_input("Ask a question about the document:")
    if query:
        answer, relevant_segments = handle_query(query)

        # Display the answer and retrieved document segments
        st.subheader("Answer:")
        st.text_area(label="Answer", value=answer, height=200)
        st.subheader("Retrieved Document Segments:")
        for i, segment in enumerate(relevant_segments):
            st.text_area(f"Segment {i + 1}:", value=segment, height=150)
