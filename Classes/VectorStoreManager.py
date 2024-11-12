from langchain.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.docstore import InMemoryDocstore
import faiss
import numpy as np

class VectorStoreManager:
    def __init__(self, embedding_model_name="BAAI/bge-small-en-v1.5"):
        # Crea embeddings usando HuggingFace
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
        
        # Prova a ottenere la dimensione dell'embedding
        sample_embedding = self.embeddings.embed_query("test")
        if isinstance(sample_embedding, list):
            embedding_dimension = len(sample_embedding[0]) if isinstance(sample_embedding[0], list) else len(sample_embedding)
        elif isinstance(sample_embedding, np.ndarray):
            embedding_dimension = sample_embedding.shape[0]
        else:
            raise ValueError(f"Formato embedding non supportato: {type(sample_embedding)}")
        
        # Inizializza l'indice FAISS con la dimensione calcolata
        faiss_index = faiss.IndexFlatL2(embedding_dimension)
        self.docstore = InMemoryDocstore({})
        self.index_to_docstore_id = {}
        self.vector_store = FAISS(
            index=faiss_index,
            docstore=self.docstore,
            index_to_docstore_id=self.index_to_docstore_id,
            embedding_function=self.embeddings.embed_query
        )

    def add_document(self, document, document_id):
        """Aggiunge un documento all'indice FAISS."""
        embedding = self.embeddings.embed_query(document)
        if isinstance(embedding, list):
            embedding = embedding[0] if isinstance(embedding[0], list) else embedding
        self.vector_store.add_texts([document], [document_id], embeddings=[embedding])

    def search(self, query, k=5):
        """Effettua una ricerca tra i documenti indicizzati."""
        return self.vector_store.similarity_search(query, k=k)