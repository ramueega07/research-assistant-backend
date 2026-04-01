from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpointEmbeddings
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from config import PINECONE_API_KEY, PINECONE_INDEX, HUGGINGFACE_API_KEY
import os

def load_and_split_pdf(file_path):
    # Load the PDF document
    loader = PyPDFLoader(file_path)
    docs = loader.load()

    # REQUIREMENT: Max 10 pages per document
    if len(docs) > 10:
        if os.path.exists(file_path):
            os.remove(file_path)
        return f"File {os.path.basename(file_path)} exceeds the 10-page limit."
    
    for doc in docs:
        doc.metadata["source"] = os.path.basename(file_path)
        # 🛑 FIX: Increment by 1 so the user sees "Page 1" instead of "Page 0"
        doc.metadata["page"] = doc.metadata.get("page", 0) + 1

    # Split the document into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=120)
    chunks = text_splitter.split_documents(docs)

    return chunks

embedding = HuggingFaceEndpointEmbeddings(huggingfacehub_api_token=HUGGINGFACE_API_KEY,model="sentence-transformers/all-MiniLM-L6-v2")

pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "researchassistant1"
def store_documents(docs,namespace_name="default"):
    PineconeVectorStore.from_documents(docs, embedding, index_name=index_name, namespace=namespace_name)