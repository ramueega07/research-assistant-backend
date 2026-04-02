# Smart Research Assistant - Backend

A powerful AI-powered research assistant backend built with FastAPI, LangChain, and Groq. This system allows users to upload PDF documents, process them into a searchable knowledge base, and query the information using advanced RAG (Retrieval-Augmented Generation) techniques.

## Features

- **PDF Document Processing**: Upload and process PDF files (max 5 files, 10 pages each)
- **Vector Database Storage**: Store document chunks in Pinecone with namespace organization
- **Intelligent Agent**: Multi-tool agent that can search documents, web, or handle general conversation
- **RAG Pipeline**: Retrieval-Augmented Generation using Groq's LLaMA models
- **Web Search Integration**: Fallback to web search when document knowledge is insufficient
- **Source Citation**: Provides structured sources with page numbers and content snippets
- **CORS Support**: Configured for frontend integration

## Tech Stack

- **Framework**: FastAPI
- **LLM**: Groq (LLaMA 3.3 70B)
- **Embeddings**: HuggingFace (sentence-transformers/all-MiniLM-L6-v2)
- **Vector Database**: Pinecone
- **Web Search**: SerpAPI (Google Search)
- **Document Processing**: PyPDF, LangChain text splitters

## Setup Instructions

### Prerequisites

- Python 3.10+
- Virtual environment (recommended)
- API keys for required services

### Installation

1. **Clone the repository and navigate to backend directory:**
   ```bash
   cd backend
   ```

2. **Create and activate virtual environment:**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables:**

   Create a `.env` file in the backend directory with the following variables:

   ```env
   GROQ_API_KEY=your_groq_api_key_here
   PINECONE_API_KEY=your_pinecone_api_key_here
   PINECONE_INDEX=your_pinecone_index_name
   SERPAPI_API_KEY=your_serpapi_key_here
   GEMINI_API_KEY=your_gemini_api_key_here 
   HUGGINGFACE_API_KEY=your_huggingface_api_key_here
   ```

### API Key Setup

- **Groq API Key**: Get from [console.groq.com](https://console.groq.com)
- **Pinecone API Key**: Get from [app.pinecone.io](https://app.pinecone.io)
- **SerpAPI Key**: Get from [serpapi.com](https://serpapi.com)
- **HuggingFace API Key**: Get from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

### Pinecone Setup

1. Create a Pinecone account and index
2. Set the index name in your `.env` file
3. The system will automatically create namespaces for uploaded documents

## Running the Application

1. **Start the FastAPI server:**
   ```bash
   python main.py
   ```

2. **The API will be available at:** `http://0.0.0.0:8000`

3. **API Documentation:** Visit `http://0.0.0.0:8000/docs` for interactive Swagger UI

## API Endpoints

### POST `/upload/`
Upload PDF files for processing and indexing.

**Parameters:**
- `files`: List of PDF files (max 5 files, 10 pages each)

**Response:**
```json
{
  "message": "Processed X documents successfully.",
  "shelves": ["document_name_1", "document_name_2"]
}
```

### POST `/query/`
Query the research assistant.

**Parameters:**
- `q`: Question string

**Response:**
```json
{
  "answer": "The answer to your question...",
  "sources": [
    {
      "type": "pdf",
      "filename": "document.pdf",
      "page": "5",
      "content": "Relevant content snippet..."
    }
  ]
}
```

### GET `/view-pdf/{filename}`
Serve uploaded PDF files for viewing.

## Agent Tools

The system uses a ReAct agent with three main tools:

1. **DocumentSearch**: Searches the uploaded document knowledge base
2. **WebSearch**: Performs web searches using SerpAPI
3. **GeneralChat**: Handles greetings and general conversation

## Architecture

- **Ingestion Pipeline**: PDF loading → text splitting → embedding → vector storage
- **Query Pipeline**: Agent reasoning → tool selection → retrieval → generation
- **Storage**: Namespaced vector storage in Pinecone for multi-document support

## Configuration

Key configuration options in `config.py`:
- LLM model selection (Groq LLaMA vs Gemini)
- Embedding model configuration
- Vector search parameters (k=5 documents)
- Text splitting parameters (chunk_size=700, overlap=120)

## Error Handling

- Rate limit handling for API keys
- File size and page limit validation
- Graceful fallback between tools
- Structured error responses

## Development

- **Memory**: Conversation memory using LangChain's memory system
- **Logging**: Tool usage logging for debugging
- **Validation**: Input validation and error handling

## License

See LICENSE file for details.