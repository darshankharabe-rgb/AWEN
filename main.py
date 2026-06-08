import os
import chromadb
from fastapi import FastAPI

app = FastAPI()

#Creating a path to folder chromaDB
DB_PATH = os.path.join(os.path.dirname(__file__), "chromadb")

#ChromaDB save data to that folder
chroma_client = chromadb.PersistentClient(path = DB_PATH)

# Create a collection (our AI table)
collection = chroma_client.get_or_create_collection(name="awen_collection")

@app.get("/")
def root():
    return {"message" : "Awen is Alive"}

@app.post("/add")
def add_document(doc_id: str, text: str):
    # ChromaDB automatically converts this text into vector math
    collection.add(
        documents=[text],
        ids=[doc_id]
    )
    return {"message": f"Successfully added document '{doc_id}' to the database!"}
# Search for a document in the colleciom
@app.get("/search")
def search_document(query: str):
    results = collection.query(
        query_texts=[query],
        n_results=1 
    )
    return {"results": results}
# Delete a document from the collection
@app.delete("/delete")
def delete_document(doc_id: str):
    collection.delete(ids=[doc_id])
    return {"message": f"Successfully deleted document '{doc_id}' from the database!"}  

# Update the text of an existing document using its ID and Endpoint
@app.put("/Update")
def update_document(doc_id: str, new_text: str):
    # This overwrites the existing document with the new text
    collection.update(
        ids=[doc_id],
        documents=[new_text]
    )
    return {"message": f"Successfully updated document '{doc_id}' in the database!"}