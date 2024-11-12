import logging
import uuid
from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel
from classes.PDFPreprocess import PDFProcessor
from classes.LangchainManager import LangchainManager
from classes.history_chains import MessageHistoryStore, ChainWithHistory  # Ensure this import is correct
from classes.DBManager import DBManager  # Importing the DBManager class
from langchain.retrievers import ElasticSearchBM25Retriever
from dotenv import load_dotenv
import os

# Initialize the logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Explicitly provide the path to your .env file
load_dotenv(dotenv_path="C:/Users/jbulgare/VS_project/Academy/RAG/OpenAI.env")

# FastAPI app initialization
app = FastAPI()

# Pydantic model for request validation
class QueryRequest(BaseModel):
    query: str
    stream: bool = False  # Optional, default False

# Initialize MongoDB Manager
db_manager = DBManager(
    db_name="pdf_database",
    collection_name="text_chunks",
    host=os.getenv("MONGO_URI", "mongodb://mongo:27017"),
    port=27017
)

# Initialize PDF Processor (this could be moved to a background task if needed)
pdf_path = "./files_pdf/thinkpython2.pdf"  # Modify this path as needed
pdf_processor = PDFProcessor(
    pdf_path=pdf_path,
    mongo_uri="mongodb://localhost:27017/",
    db_name="pdf_database",
    collection_name="text_chunks"
)
pdf_processor.process_and_store()

# Initialize LangchainManager for Q&A interaction
openai_api_base = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("API Key for OpenAI is missing. Set it as an environment variable 'OPENAI_API_KEY'.")

langchain_manager = LangchainManager(openai_api_base=openai_api_base, api_key=api_key)

# Set up message history management
message_history_store = MessageHistoryStore()

# Example setup for ElasticSearchRetriever, replace it with your chosen method
retriever = ElasticSearchBM25Retriever(
    index_name="your_index_name", 
    embedding_function=pdf_processor.embedding_model.encode  # This should match your embedding setup
)

qa_chain = langchain_manager.create_chain(retriever=retriever)

# Chain with history
chain_with_history = ChainWithHistory(
    qa_chain=qa_chain,
    message_history_store=message_history_store
)

@app.get("/")
def read_root():
    logger.info("Root endpoint accessed")
    return {"message": "Hello World"}

@app.post("/start_conversation")
async def start_conversation():
    """Start a new conversation and generate a new conversation ID."""
    conversation_id = str(uuid.uuid4())  # Generate a new UUID for the conversation
    logger.info(f"New conversation started with ID: {conversation_id}")
    return {"conversation_id": conversation_id}

@app.post("/chat/{conversation_id}")
async def chat(conversation_id: str, request: QueryRequest):
    """Handle the chat interaction with history chains."""
    query = request.query
    stream = request.stream

    # Prepare input data for the chain (for example, just the query)
    input_data = {"question": query}

    # Set session or conversation ID to track history
    config = {"configurable": {"session_id": conversation_id}}  # Use the correct session_id from conversation

    # Get response from the Q&A chain with history
    response_with_history = chain_with_history.invoke(input_data=input_data, config=config)

    # Store the conversation history in MongoDB
    conversation_data = {
        "conversation_id": conversation_id,
        "conversation_history": message_history_store.get_by_session_id(conversation_id).get_conversation()
    }
    db_manager.insert_document(conversation_data)

    # Return response and conversation history
    return {
        "response": response_with_history,
        "conversation_id": conversation_id,
        "history": message_history_store.get_by_session_id(conversation_id).get_conversation()
    }

@app.get("/conversation/{conversation_id}")
def get_conversation_history(conversation_id: str):
    """Retrieve the conversation history for a given session."""
    logger.info(f"Retrieving conversation history for ID: {conversation_id}")
    conversation = db_manager.read_document({"conversation_id": conversation_id})
    if conversation:
        logger.info(f"Conversation history found for ID: {conversation_id}")
        return {"history": conversation["conversation_history"]}
    else:
        logger.warning(f"Conversation history not found for ID: {conversation_id}")
        raise HTTPException(status_code=404, detail="Conversation not found.")

@app.post("/clear-conversation/{conversation_id}")
def clear_conversation(conversation_id: str):
    """Clear the conversation history for a given session."""
    # Clear the conversation history in MongoDB
    result = db_manager.delete_document({"conversation_id": conversation_id})
    if result:
        return {"message": "Conversation history cleared."}
    else:
        raise HTTPException(status_code=404, detail="Conversation not found.")

@app.get("/documents")
def get_documents():
    """Retrieve documents from MongoDB."""
    documents = db_manager.read_documents({}, limit=5)
    return {"documents": documents}

@app.post("/documents")
def insert_document(document: dict):
    """Insert a new document into MongoDB."""
    inserted_id = db_manager.insert_document(document)
    if inserted_id:
        return {"message": f"Inserted new document with ID: {inserted_id}"}
    else:
        raise HTTPException(status_code=400, detail="Error inserting document.")

@app.on_event("shutdown")
def shutdown():
    """Close the MongoDB connection when the server shuts down."""
    db_manager.close_connection()
