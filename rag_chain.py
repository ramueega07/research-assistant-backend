from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from config import GROQ_API_KEY, PINECONE_INDEX, GEMINI_API_KEY

llm = ChatGroq(groq_api_key=GROQ_API_KEY,model="llama-3.3-70b-versatile", temperature=0)
#llm = ChatGoogleGenerativeAI(google_api_key=GEMINI_API_KEY, model="gemini-2.5-flash",temperature=0, convert_system_message_to_human=True)

prompt = ChatPromptTemplate.from_template("""
You are an expert Research Analyst. Your goal is to provide a highly technical and data-driven summary based ONLY on the provided context.

CRITICAL INSTRUCTIONS:
1. Always prioritize quantitative data (percentages, metrics, accuracy rates).
2. Identify and mention specific Frameworks, Figures, or Tables by name if they appear.
3. If the context mentions specific results (like '98% accuracy'), you MUST include them.
                                          
Context:
{context}

Question:
{question}

Answer with clear citations:
(Document Name, Page Number)
""")

output_parser = StrOutputParser()

rag_chain = prompt | llm | output_parser

embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
index_name = "researchassistant1"
vector_store = PineconeVectorStore(embedding=embedding, index_name=index_name)

retriever = vector_store.as_retriever(search_kwargs={"k": 5})