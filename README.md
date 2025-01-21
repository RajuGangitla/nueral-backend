# FastAPI Backend for Chat Bot

This is a FastAPI backend for a RAG Application with File Uploads. It integrates with Pinecone for vector storage and Azure OpenAI for embeddings and chat functionality.

## Environment Variables

Create a `.env` file in the root directory and add the following environment variables:

```plaintext
PINECONE_API_KEY=your_pinecone_api_key
AZURE_OPENAI_ENDPOINT=your_azure_openai_endpoint
AZURE_OPENAI_API_KEY=your_azure_openai_api_key
EMBEDDING_MODEL=your_embedding_model
PINECONE_INDEX_NAME=your_pinecone_index_name
AZURE_API_VERSION=your_azure_api_version
```

1. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

2. **Run the project**:
   ```bash
   uvicorn main:app --reload
   ```
