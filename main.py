import os
import asyncio
import chromadb
import fitz  # PyMuPDF for handling PDFs efficiently
from fastapi import FastAPI, HTTPException, UploadFile, File
from google import genai
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

# 1. Initialize the Gemini Client
client = genai.Client()

# Set up ChromaDB locally
DB_PATH = os.path.join(os.path.dirname(__file__), "chromadb")
chroma_client = chromadb.PersistentClient(path=DB_PATH)
collection = chroma_client.get_or_create_collection(name="awen_collection")


@app.get("/")
async def root():
    # Made this async! FastAPI handles async naturally, which frees up the server 
    # to handle other requests while waiting for I/O operations.
    return {"message": "Awen is Alive and Async"}


@app.post("/add")
async def add_document(doc_id: str, text: str):
    # We use asyncio.to_thread to run synchronous ChromaDB operations 
    # without blocking the main FastAPI event loop.
    await asyncio.to_thread(
        collection.add,
        documents=[text],
        ids=[doc_id]
    )
    return {"message": f"Successfully added document '{doc_id}' to the database!"}


# =========================================
# TEXT CHUNKING HELPER
# =========================================
def chunk_text(text: str, chunk_size: int = 500, chunk_overlap: int = 50):
    """
    Slices large text into smaller overlapping chunks.
    Overlap ensures we don't cut a sentence in half and lose context.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - chunk_overlap 
    return chunks


# =========================================
# FILE UPLOAD ENDPOINT (Only Text & PDF)
# =========================================
ALLOWED_CONTENT_TYPES = {
    "text/plain",
    "application/pdf"
}

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    # 1. Guard against unsupported file types (Images removed as requested)
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file.content_type}. Please upload a text or PDF file."
        )
    
    # 2. Read the raw bytes asynchronously
    content = await file.read()
    text = ""
    
    # 3. Extract text based on file type
    if file.content_type == "text/plain":
        try:
            text = content.decode("utf-8")
        except Exception:
            raise HTTPException(status_code=400, detail="Failed to decode text file. Ensure it is UTF-8.")
            
    elif file.content_type == "application/pdf":
        try:
            # PyMuPDF (fitz) needs a document object. We pass the raw bytes directly.
            pdf_document = fitz.open(stream=content, filetype="pdf")
            
            # Iterate through every page and extract the text
            for page_num in range(len(pdf_document)):
                page = pdf_document.load_page(page_num)
                text += page.get_text()
                
            pdf_document.close()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to parse PDF: {str(e)}")
    
    # 4. Slice the extracted text into manageable chunks
    chunks = chunk_text(text)
    if not chunks:
        raise HTTPException(status_code=400, detail="No readable text found in the document.")
    
    # 5. Generate unique IDs for each chunk
    chunk_ids = [f"{file.filename}_chunk_{i}" for i in range(len(chunks))]
    
    # 6. Save chunks to ChromaDB without blocking the server
    await asyncio.to_thread(
        collection.add,
        documents=chunks,
        ids=chunk_ids
    )
    
    return {
        "message": f"Successfully processed '{file.filename}'",
        "total_chunks_created": len(chunks)
    }


# =========================================
# THE RAG QUERY ENDPOINT
# =========================================
@app.post("/query")
async def query_rag(query: str):
    # Retrieve relevant context from ChromaDB
    search_results = await asyncio.to_thread(
        collection.query,
        query_texts=[query],
        n_results=2 
    )
    
    # Verify if we found any context
    if not search_results or not search_results['documents'] or len(search_results['documents'][0]) == 0:
        raise HTTPException(status_code=404, detail="No relevant context found. Add documents first.")
    
    # Combine chunks into a single text block
    retrieved_chunks = search_results['documents'][0]
    context = "\n".join(retrieved_chunks)
    
    prompt = f"""
    You are a precise AI assistant named Awen. 
    Answer the user's question using ONLY the provided context below. 
    If the context does not contain the answer, say "I don't have enough information in my database".
    
    Context:
    {context}
    
    User Question: {query}
    Answer:
    """
    
    # Call Gemini LLM asynchronously to prevent holding up other users
    try:
        response = await asyncio.to_thread(
            client.models.generate_content,
            model='gemini-2.5-flash-lite',
            contents=prompt,
        )
        
        return {
            "query": query,
            "ai_response": response.text
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI API Error: {str(e)}")


@app.delete("/delete")
async def delete_document(doc_id: str):
    await asyncio.to_thread(collection.delete, ids=[doc_id])
    return {"message": f"Successfully deleted document '{doc_id}'!"}  

@app.put("/update")
async def update_document(doc_id: str, new_text: str):
    await asyncio.to_thread(
        collection.update,
        ids=[doc_id],
        documents=[new_text]
    )
    return {"message": f"Successfully updated document '{doc_id}'!"}