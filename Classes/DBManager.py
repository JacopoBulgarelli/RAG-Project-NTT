import logging
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, PyMongoError

# Configure logging for this module
logger = logging.getLogger(__name__)

class DBManager:
    def __init__(self, db_name, collection_name, host='mongodb://localhost:27017', port=None):
        """
        Initialize the DBManager class to handle MongoDB operations.
        
        :param db_name: Name of the MongoDB database.
        :param collection_name: Name of the MongoDB collection.
        :param host: MongoDB URI (default is 'mongodb://localhost:27017').
        :param port: Port number for MongoDB (optional if included in URI).
        """
        self.db_name = db_name
        self.collection_name = collection_name
        self.host = host
        self.port = port
        
        try:
            # Connection to MongoDB
            if self.port:
                self.client = MongoClient(self.host, self.port, serverSelectionTimeoutMS=5000)
            else:
                self.client = MongoClient(self.host, serverSelectionTimeoutMS=5000)
                
            # Check the connection
            self.client.server_info()  # Forces connection to validate
            self.db = self.client[self.db_name]
            self.collection = self.db[self.collection_name]
            logger.info(f"Connected to MongoDB at {self.host}, database: {self.db_name}, collection: {self.collection_name}")
            
        except ConnectionFailure as e:
            logger.error("Failed to connect to MongoDB", exc_info=e)
            self.client = None
        except PyMongoError as e:
            logger.error(f"MongoDB error during initialization: {e}")
            self.client = None

    def insert_document(self, document):
        """Insert a document and return the inserted ID."""
        if not self.client:
            logger.warning("Cannot insert document; MongoDB connection not available")
            return None
        try:
            result = self.collection.insert_one(document)
            logger.info(f"Document inserted with ID: {result.inserted_id}")
            return result.inserted_id
        except PyMongoError as e:
            logger.error(f"Error inserting document: {e}")
            return None

    def read_document(self, filter):
        """Read a single document that matches the filter."""
        if not self.client:
            logger.warning("Cannot read document; MongoDB connection not available")
            return None
        try:
            document = self.collection.find_one(filter)
            logger.info(f"Document read: {document}")
            return document
        except PyMongoError as e:
            logger.error(f"Error reading document: {e}")
            return None

    def read_documents(self, filter={}, limit=0):
        """Read all documents that match the filter with an optional limit."""
        if not self.client:
            logger.warning("Cannot read documents; MongoDB connection not available")
            return []
        try:
            documents = list(self.collection.find(filter).limit(limit))
            logger.info(f"{len(documents)} documents read")
            return documents
        except PyMongoError as e:
            logger.error(f"Error reading documents: {e}")
            return []

    def update_document(self, filter, new_data):
        """Update a document that matches the filter with new data."""
        if not self.client:
            logger.warning("Cannot update document; MongoDB connection not available")
            return 0
        try:
            result = self.collection.update_one(filter, {"$set": new_data})
            logger.info(f"Document updated, modified count: {result.modified_count}")
            return result.modified_count
        except PyMongoError as e:
            logger.error(f"Error updating document: {e}")
            return 0

    def delete_document(self, filter):
        """Delete a single document that matches the filter."""
        if not self.client:
            logger.warning("Cannot delete document; MongoDB connection not available")
            return 0
        try:
            result = self.collection.delete_one(filter)
            logger.info(f"Document deleted, deleted count: {result.deleted_count}")
            return result.deleted_count
        except PyMongoError as e:
            logger.error(f"Error deleting document: {e}")
            return 0

    def close_connection(self):
        """Close the MongoDB connection."""
        if self.client:
            self.client.close()
            logger.info("MongoDB connection closed")
