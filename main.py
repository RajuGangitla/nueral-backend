import os
from typing import List
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, File, HTTPException, Request, Response, UploadFile
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from dotenv import load_dotenv
from db.vector_store import DocumentRetreiveSystem
from langchain.prompts import ChatPromptTemplate



load_dotenv()
app = FastAPI()
doc_system = DocumentRetreiveSystem()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://neural-frontend-alpha.vercel.app/", "https://neural-frontend-alpha.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

async def create_embeddings():
    return AzureOpenAIEmbeddings(
        azure_deployment="text-embedding-ada-002",
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version="2023-05-15"
    )

async def create_llm():
   return AzureChatOpenAI(
    azure_deployment="gpt-4o",  # or your deployment
    api_version="2023-06-01-preview",  # or your api version
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
)

@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/upload")
async def upload_document(file: UploadFile):
    """Upload and process a PDF document"""
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    try:
         # Save the uploaded file temporarily
        temp_file_path = f"temp_{file.filename}"
        with open(temp_file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Process the document
        documents = await doc_system.load_and_split_documents(temp_file_path, file.filename)
        num_chunks = await doc_system.insert_documents(documents)

        os.remove(temp_file_path)
        
        return {
            "message": f"Successfully processed {file.filename}",
            "chunks": num_chunks
        }        
    except Exception as e:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        raise HTTPException(status_code=500, detail=str(e))
    

def create_context_from_docs(processed_results: List[dict]) -> str:
    # Extract relevant text from the processed results
    context_parts = []
    for doc in processed_results:
        if 'metadata' in doc and 'text' in doc['metadata']:
            context_parts.append(doc['metadata']['text'])
    
    return "\n\n".join(context_parts)


@app.route("/agent", methods=["POST"])
async def answer_user_query(request: Request):
    try:
        data = await request.json()
        query = data.get("query")
        if not query:
            raise HTTPException(status_code=400, detail="Query is required")
        
        # Create instances asynchronously
        llm = await create_llm()
        
        # Search documents
        documents = await doc_system.retrieve_documents(query=query)
        
        processed_results = [
            {
                'id': str(idx),
                'score': 1.0,  # Default score since we don't have it
                'metadata': {
                    'text': doc,  # The page_content is now in the text field
                    'source': 'document'
                }
            }
            for idx, doc in enumerate(documents)
        ]
        
        # Create context from retrieved documents
        context = create_context_from_docs(processed_results)
        
        # Create prompt template
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful AI assistant. Use the following context to answer the user's question. 
                If the context is provided, base your answer on it. If the context is empty or irrelevant, respond in a helpful and informative way based on your general knowledge.            
            Context: {context}"""),
            ("human", "{question}")
        ])

        inputs = {
                "context": context,
                "question":query
            }
        
        messages = await prompt.ainvoke(inputs)
        response = await llm.ainvoke(messages)
        
        return Response(content=response.content, media_type="text/plain")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))