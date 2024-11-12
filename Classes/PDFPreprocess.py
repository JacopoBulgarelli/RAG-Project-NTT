from sentence_transformers import SentenceTransformer
from pymongo import MongoClient
import re
from typing import List
import fitz  # PyMuPDF for PDF parsing


class PDFProcessor:
    def __init__(self, pdf_path: str, mongo_uri: str, db_name: str, collection_name: str, embedding_model_name='all-MiniLM-L6-v2'):
        """
        Initializes the PDF processor with MongoDB connection and PDF file path.
        
        Parameters:
        pdf_path (str): Path to the PDF file.
        mongo_uri (str): MongoDB URI for connection.
        db_name (str): MongoDB database name.
        collection_name (str): MongoDB collection name for storing chunks.
        embedding_model_name (str): The name of the SentenceTransformer model to use for embeddings.
        """
        self.pdf_path = pdf_path
        self.mongo_uri = mongo_uri
        self.db_name = db_name
        self.collection_name = collection_name
        self.client = MongoClient(mongo_uri)
        self.db = self.client[db_name]
        self.collection = self.db[collection_name]
        self.embedding_model = SentenceTransformer(embedding_model_name)  # Load the embedding model

    def load_pdf(self) -> str:
        """
        Loads and extracts text from a PDF file.
        
        Returns:
        str: Extracted text from the PDF.
        """
        text = ""
        with fitz.open(self.pdf_path) as pdf:
            for page in pdf:
                text += page.get_text()
        return text

    def normalize_text(self, text: str) -> str:
        """
        Normalizes the text by converting it to lowercase, removing punctuation, 
        and eliminating stop words.
        
        Parameters:
        text (str): Raw text to be normalized.
        
        Returns:
        str: Normalized text.
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation
        text = re.sub(r'[^\w\s]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Define stop words (using a fixed list instead of nltk's stop words)
        stop_words = {
            'the', 'is', 'in', 'and', 'to', 'of', 'that', 'it', 'for', 'on', 'with', 
            'as', 'by', 'at', 'from', 'this', 'or', 'an', 'be', 'was', 'are', 'not', 
            'have', 'had', 'but', 'has', 'they', 'which', 'a', 'their', 'we', 'you', 
            'your', 'more', 'can', 'about', 'so', 'my', 'there', 'some', 'what', 'if',
            'when', 'all', 'one', 'also', 'out', 'who', 'were', 'will', 'other', 'do'
        }
        
        # Remove stop words
        text = " ".join(word for word in text.split() if word not in stop_words)
        
        return text

    def split_text(self, text: str, max_length: int = 200) -> List[str]:
        """
        Splits the text into chunks of specified maximum length.
        
        Parameters:
        text (str): Text to be split into chunks.
        max_length (int): Maximum token count for each chunk.
        
        Returns:
        List[str]: List of text chunks.
        """
        # Use regex to split the text into sentences
        sentences = re.split(r'(?<=[.!?]) +', text)
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            # Split sentences to ensure the chunk size doesn't exceed `max_length`
            if len(current_chunk.split()) + len(sentence.split()) <= max_length:
                current_chunk += " " + sentence
            else:
                # Add the current chunk to chunks and start a new chunk
                chunks.append(current_chunk.strip())
                current_chunk = sentence

        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks

    def generate_embeddings(self, chunks: List[str]) -> List[List[float]]:
        """
        Generates embeddings for each chunk using the SentenceTransformer model.
        
        Parameters:
        chunks (List[str]): List of text chunks to be embedded.
        
        Returns:
        List[List[float]]: List of embeddings for the chunks.
        """
        return self.embedding_model.encode(chunks).tolist()

    def save_to_mongo(self, chunks: List[str], embeddings: List[List[float]]):
        """
        Saves chunks and their embeddings to MongoDB.
        
        Parameters:
        chunks (List[str]): List of text chunks to be stored.
        embeddings (List[List[float]]): Corresponding embeddings for the chunks.
        """
        documents = [{"chunk_text": chunk, "embedding": embedding} for chunk, embedding in zip(chunks, embeddings)]
        self.collection.insert_many(documents)

    def process_and_store(self):
        """
        Loads, processes, generates embeddings, and stores text chunks from PDF into MongoDB.
        """
        # Load and process text
        raw_text = self.load_pdf()
        normalized_text = self.normalize_text(raw_text)  # Apply normalization
        chunks = self.split_text(normalized_text)

        # Generate embeddings
        embeddings = self.generate_embeddings(chunks)

        # Save to MongoDB
        self.save_to_mongo(chunks, embeddings)

        print(f"Successfully saved {len(chunks)} chunks with embeddings to MongoDB.")

# Example usage:
pdf_processor = PDFProcessor(
    pdf_path="./files_pdf/thinkpython2.pdf",
    mongo_uri="mongodb://localhost:27017/",
    db_name="pdf_database",
    collection_name="text_chunks"
)

pdf_processor.process_and_store()
