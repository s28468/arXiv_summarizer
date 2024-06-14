from pymongo import MongoClient
from pymongo.errors import PyMongoError
class MongoDBSaver:
   def __init__(self, uri, db_name, collection_name):
       self.uri = uri
       self.db_name = db_name
       self.collection_name = collection_name
       self.client = None
       self.db = None
       self.collection = None
       self.connect()
   def connect(self):
       try:
           self.client = MongoClient(self.uri)
           self.db = self.client[self.db_name]
           self.collection = self.db[self.collection_name]
           print("Connected to MongoDB successfully")
       except PyMongoError as e:
           print(f"Could not connect to MongoDB: {e}")
   def save_data(self, data):
       if self.collection is None:
           print("No collection available to insert data")
           return
       try:
           result = self.collection.insert_one(data)
           print(f"Data inserted with id: {result.inserted_id}")
       except Exception as e:
           print(f"Error inserting data: {e}")
# Example usage
if __name__ == "__main__":
   mongo_saver = MongoDBSaver("mongodb://localhost:27017/", "arxiv_db", "documents")
   test_data = {"title": "Sample PDF", "summary": "This is a summary"}
   mongo_saver.save_data(test_data)