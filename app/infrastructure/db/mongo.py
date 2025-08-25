import os
from pymongo import MongoClient
from pymongo.errors import ServerSelectionTimeoutError


class ConfigureDatabase:
    def __init__(self,
                 uri=None,
                 database_name=None):
        self.client = None
        self.database = None
        try:
            # Read from env with safe defaults to preserve existing behavior
            uri = uri or os.getenv("MONGODB_URI", "mongodb+srv://husni:husni@cluster0.7kwciz1.mongodb.net/")
            database_name = database_name or os.getenv("MONGODB_DB", "cognitive_load_detection")

            self.client = MongoClient(uri, serverSelectionTimeoutMS=5000)  # 5 sec timeout
            self.client.admin.command('ping')
            print("MongoDB connection successful")
            self.database = self.client[database_name]
        except ServerSelectionTimeoutError:
            print("Failed to connect to MongoDB")
            self.client = None
            self.database = None

    def get_cognitive_states_collection(self, collection_name="cognitive_states"):
        if self.database is None:
            raise RuntimeError("Database is not initialized. Check MongoDB connection and configuration.")
        return self.database[collection_name]
