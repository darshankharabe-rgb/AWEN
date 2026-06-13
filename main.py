import os
import chromadb
from fastapi import FastAPI, HTTPException, UploadFile, File
from google import genai
from dotenv import load_dotenv
load_dotenv()

app = FastAPI()

# 1. Initialize the Gemini Client
# Make sure you have set your environment variable: export GEMINI_API_KEY="your-key"
client = genai.Client()

DB_PATH = os.path.join(os.path.dirname(__file__), "chromadb")
chroma_client = chromadb.PersistentClient(path=DB_PATH)
collection = chroma_client.get_or_create_collection(name="awen_collection")

@app.get("/")
def root():
    return {"message": "Awen is Alive"}

@app.post("/add")
def add_document(doc_id: str, text: str):
    collection.add(
        documents=[text],
        ids=[doc_id]
    )
    return {"message": f"Successfully added document '{doc_id}' to the database!"}

# START OF NEW CODE: FILE UPLOAD & CHUNKING
# =========================================
def chunk_text(text: str, chunk_size: int = 500, chunk_overlap: int = 50):
    """
    Helper function to slice a large text into smaller overlapping chunks.
    Overlap ensures we don't cut a sentence in half and lose the context.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        # Move the start forward, but step back slightly to create an overlap
        start += chunk_size - chunk_overlap 
    return chunks

#UPLOAD ENDPOINT for file upload
#===============

# Allowed file types map
ALLOWED_CONTENT_TYPES = {
    "text/plain",
    "application/pdf",
    "image/jpeg",
    "image/png"
}

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    #1. Guard against unsupported file types
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file.content_type}. Please upload a text, PDF, or image file."
        )
    
    #2. Read the raw bytes from the uploaded file
    content = await file.read()
    
    # 3. Extract text from the file based on its type
    text = ""
    
    if file.content_type == "text/plain":
        try:
            text = content.decode("utf-8")
        except Exception:
            raise HTTPException(status_code=400, detail="Failed to decode text file. Ensure it is UTF-8 formatted.")
            
    elif file.content_type == "application/pdf":
        # TODO: Add PDF text extraction logic here
        raise HTTPException(
            status_code=501, 
            detail="PDF accepted by server, but PDF text extraction code is not yet written."
        )
        
    elif file.content_type in ["image/jpeg", "image/png", "image/webp"]:
        # TODO: Add Image Optical Character Recognition (OCR) logic here
        raise HTTPException(
            status_code=501, 
            detail="Image accepted by server, but Image reading code is not yet written."
        )
    
    # 4. Slice the large text into manageable chunks
    chunks = chunk_text(text)
    
    # 5. Generate unique IDs for each chunk (e.g., "my_file.txt_chunk_0")
    chunk_ids = [f"{file.filename}_chunk_{i}" for i in range(len(chunks))]
    
    # 6. Save all chunks into ChromaDB simultaneously
    collection.add(
        documents=chunks,
        ids=chunk_ids
    )
    
    return {
        "message": f"Successfully processed '{file.filename}'",
        "total_chunks_created": len(chunks)
    }






@app.get("/search")
def search_document(query: str):
    results = collection.query(
        query_texts=[query],
        n_results=1 
    )
    return {"results": results}

# --- THE NEW STEP: THE RAG QUERY ENDPOINT ---
@app.post("/query")
def query_rag(query: str):
    # Step A: Retrieve relevant context from ChromaDB
    search_results = collection.query(
        query_texts=[query],
        n_results=2  # Let's pull the top 2 closest matches for better context
    )
    
    # Verify if we actually found any documents
    if not search_results or not search_results['documents'] or len(search_results['documents'][0]) == 0:
        raise HTTPException(status_code=404, detail="No relevant context found in the database. Please add documents first.")
    
    # Combine the retrieved chunks into a single text block
    retrieved_chunks = search_results['documents'][0]
    context = "\n".join(retrieved_chunks)
    
    #  Constructing the RAG prompt with strict instructions
    #  to only use the provided context

    prompt = f"""
    You are a precise AI assistant named Awen. 
    Answer the user's question using ONLY the provided context below. 
    If the context does not contain the answer, say "I don't have enough information in my database".
    
    Context:
    {context}
    
    User Question: {query}
    Answer:
    """
    
    #Generating the answer using Gemini Flash Lite LLM
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash-lite',
            contents=prompt,
        )
        
        return {
            "query": query,
            "retrieved_context": retrieved_chunks,
            "ai_response": response.text
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"API Key Error for AI LLM Please check the key : {str(e)}")

@app.delete("/delete")
def delete_document(doc_id: str):
    collection.delete(ids=[doc_id])
    return {"message": f"Successfully deleted document '{doc_id}' from the database!"}  

@app.put("/Update")
def update_document(doc_id: str, new_text: str):
    collection.update(
        ids=[doc_id],
        documents=[new_text]
    )
    return {"message": f"Successfully updated document '{doc_id}' in the database!"} 