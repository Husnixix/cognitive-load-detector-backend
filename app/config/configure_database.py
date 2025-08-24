from pymongo import MongoClient
from pymongo.errors import ServerSelectionTimeoutError


class ConfigureDatabase:
    def __init__(self,
                 uri="mongodb+srv://husni:husni@cluster0.7kwciz1.mongodb.net/",
                 database_name="cognitive_load_detection"):
        self.client = None
        self.database = None
        try:
            self.client = MongoClient(uri, serverSelectionTimeoutMS=5000)  # 5 sec timeout
            self.client.admin.command('ping')
            print("MongoDB connection successful")
            self.database = self.client[database_name]
        except ServerSelectionTimeoutError:
            print("Failed to connect to MongoDB")
            self.client = None
            self.database = None

    def get_cognitive_states_collection(self, collection_name="cognitive_states"):
        return self.database[collection_name]



