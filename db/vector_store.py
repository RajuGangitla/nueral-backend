from ast import List
import os 
from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
from langchain_openai import AzureOpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from io import BytesIO
from pypdf import PdfReader

# Load environment variables
load_dotenv()


class DocumentRetreiveSystem:
    def __init__(self, text_chunk_size: int = 1000, text_chunk_overlap: int = 200):
        """
        Initializes the Document Retrieval System using environment variables.

        Args:
            text_chunk_size (int): Chunk size for text splitting. Default is 1000.
            text_chunk_overlap (int): Overlap size for text splitting. Default is 200.
        """
        self.text_chunk_size = text_chunk_size
        self.text_chunk_overlap = text_chunk_overlap

        # Load configuration from environment variables
        self.embedding_model = os.getenv("EMBEDDING_MODEL")
        self.azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.azure_api_version = os.getenv("AZURE_API_VERSION")
        self.azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
        self.pinecone_index_name = os.getenv("PINECONE_INDEX_NAME")

                # Initialize Azure OpenAI embeddings
        self.embeddings = AzureOpenAIEmbeddings(
            deployment=self.embedding_model,
            azure_endpoint=self.azure_endpoint,
            openai_api_version=self.azure_api_version,
            openai_api_key=self.azure_api_key
        )

    async def load_and_split_documents(self, file_content: bytes, file_name: str):
        """
        Loads and splits the documents into chunks directly from in-memory file content.
        
        Args:
            file_content (bytes): The content of the file as bytes.
            file_name (str): Original filename for metadata.
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.text_chunk_size,
            chunk_overlap=self.text_chunk_overlap,
            separators=["\n\n", "\n", ".", "!", "?", " ", ""],
            length_function=len
        )

        # Load and process the PDF from in-memory bytes
        pdf_file = BytesIO(file_content)
        reader = PdfReader(pdf_file)
        documents = []

        # Extract text from each page and create documents
        for page_num, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                documents.append({
                    "page_content": text,
                    "metadata": {
                        "source": file_name,
                        "page": page_num + 1
                    }
                })

        # Split documents
        return text_splitter.split_documents(documents)

    async def insert_documents(self, docs):
        """
        Inserts documents into the Pinecone vector store.
        
        Args:
            docs: List of document chunks to insert
        """
        vector_store = PineconeVectorStore.from_documents(
            docs,
            index_name=self.pinecone_index_name,
            embedding=self.embeddings
        )
        return len(docs)  # Return number of chunks inserted
        
    async def retrieve_documents(self, query: str, k: int = 10):
        """
        Retrieves relevant documents based on a query.

        Args:
            query (str): The query to search for relevant documents.
            k (int): The number of top matches to retrieve. Default is 10.

        Returns:
            List[str]: List of matched document contents.
        """
        vector_store = PineconeVectorStore(
            index_name=self.pinecone_index_name,
            embedding=self.embeddings
        )
        retriever = vector_store.as_retriever(
            search_type="mmr", 
            search_kwargs={'k': k}
        )
        
        # Use the new invoke method instead of get_relevant_documents
        matched_docs = await retriever.ainvoke(query)
        print(matched_docs, "matched_docs")
        # Extract content from matched documents
        return [doc.page_content for doc in matched_docs]

        pass