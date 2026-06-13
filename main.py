import os
import asyncio
import uuid
import chromadb
import fitz
import bcrypt
import jwt
from datetime import datetime, timedelta
from fastapi import FastAPI, HTTPException, UploadFile, File, Depends, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from google import genai
from dotenv import load_dotenv
from sqlalchemy import create_engine, Column, String, Integer, DateTime
from sqlalchemy.orm import sessionmaker, declarative_base, Session

load_dotenv()

app = FastAPI()

# =========================================
# 1. AUTHENTICATION CONFIGURATION
# =========================================
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "my-super-secret-development-key-awen")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

def get_password_hash(password: str) -> str:
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return bcrypt.checkpw(plain_password.encode('utf-8'), hashed_password.encode('utf-8'))

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


# =========================================
# 2. POSTGRESQL DATABASE SETUP (WITH USERS)
# =========================================
SQLALCHEMY_DATABASE_URL = os.getenv("DATABASE_URL")
if not SQLALCHEMY_DATABASE_URL:
    raise RuntimeError("DATABASE_URL is not set in the .env file!")

engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class UserRecord(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    username = Column(String, unique=True, index=True)
    hashed_password = Column(String)

class DocumentRecord(Base):
    __tablename__ = "documents"
    id = Column(String, primary_key=True, index=True)
    filename = Column(String, index=True)
    chunk_count = Column(Integer)
    owner_id = Column(Integer) # Tracks WHICH user owns this file
    upload_date = Column(DateTime, default=datetime.utcnow)

# --- EPHEMERAL MODE (CLEAN SLATE) ---
# Wipes Supabase clean every time the server restarts. 
# NOTE: Once you move to production, delete the `drop_all` line so users aren't deleted!
Base.metadata.drop_all(bind=engine)
Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# =========================================
# 3. CURRENT USER DEPENDENCY (THE SECURITY WALL)
# =========================================
def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except jwt.PyJWTError:
        raise credentials_exception
        
    user = db.query(UserRecord).filter(UserRecord.username == username).first()
    if user is None:
        raise credentials_exception
    return user


# =========================================
# 4. AI & EPHEMERAL VECTOR DB INITIALIZATION
# =========================================
client = genai.Client()

# Uses computer RAM. Wipes completely clean when server stops.
chroma_client = chromadb.EphemeralClient() 
collection = chroma_client.get_or_create_collection(name="awen_collection")

@app.get("/")
async def root():
    return {"message": "Awen is SECURE, Full-Stack, and running in Clean Slate Mode!"}


# =========================================
# 5. AUTHENTICATION ENDPOINTS
# =========================================
@app.post("/register")
async def register_user(username: str, password: str, db: Session = Depends(get_db)):
    existing_user = db.query(UserRecord).filter(UserRecord.username == username).first()
    if existing_user:
        raise HTTPException(status_code=400, detail="Username already registered")
    
    hashed_pwd = get_password_hash(password)
    new_user = UserRecord(username=username, hashed_password=hashed_pwd)
    
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    
    return {"message": f"User {username} created successfully! You can now log in."}

@app.post("/login")
async def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(UserRecord).filter(UserRecord.username == form_data.username).first()
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token = create_access_token(data={"sub": user.username})
    return {"access_token": access_token, "token_type": "bearer"}


# =========================================
# TEXT CHUNKING HELPER
# =========================================
def chunk_text(text: str, chunk_size: int = 500, chunk_overlap: int = 50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - chunk_overlap 
    return chunks


# =========================================
# 6. SECURED CORE ENDPOINTS
# =========================================
ALLOWED_CONTENT_TYPES = {"text/plain", "application/pdf"}

@app.post("/upload")
async def upload_document(
    file: UploadFile = File(...), 
    db: Session = Depends(get_db),
    current_user: UserRecord = Depends(get_current_user) # Locks the endpoint
):
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=400, detail="Please upload a text or PDF file.")
    
    content = await file.read()
    text = ""
    
    if file.content_type == "text/plain":
        try:
            text = content.decode("utf-8")
        except Exception:
            raise HTTPException(status_code=400, detail="Failed to decode text.")
    elif file.content_type == "application/pdf":
        try:
            pdf_document = fitz.open(stream=content, filetype="pdf")
            for page_num in range(len(pdf_document)):
                page = pdf_document.load_page(page_num)
                text += str(page.get_text()) # Pylance fix
            pdf_document.close()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to parse PDF: {str(e)}")
    
    chunks = chunk_text(text)
    if not chunks:
        raise HTTPException(status_code=400, detail="No readable text found.")
    
    # UUID + User ID prevents crashes and isolates data
    unique_file_id = f"user_{current_user.id}_{uuid.uuid4().hex[:8]}_{file.filename}"
    
    chunk_ids = [f"{unique_file_id}_chunk_{i}" for i in range(len(chunks))]
    
    await asyncio.to_thread(collection.add, documents=chunks, ids=chunk_ids)
    
    db_document = DocumentRecord(
        id=unique_file_id, 
        filename=file.filename,
        chunk_count=len(chunks),
        owner_id=current_user.id 
    )
    db.add(db_document)
    db.commit()
    db.refresh(db_document)
    
    return {
        "message": f"Successfully processed '{file.filename}' for user '{current_user.username}'",
        "total_chunks": len(chunks)
    }


@app.post("/query")
async def query_rag(
    query: str, 
    current_user: UserRecord = Depends(get_current_user) # Locks the endpoint
):
    search_results = await asyncio.to_thread(
        collection.query,
        query_texts=[query],
        n_results=2 
    )
    
    if not search_results or not search_results['documents'] or len(search_results['documents'][0]) == 0:
        raise HTTPException(status_code=404, detail="No relevant context found.")
    
    retrieved_chunks = search_results['documents'][0]
    context = "\n".join(retrieved_chunks)
    
    prompt = f"""
    You are a precise AI assistant named Awen. 
    Answer the user's question using ONLY the provided context below. 
    If the context does not contain the answer, say "I don't have enough information in my database but i would still try to answer your question" and then give a decent answer based on that context.
    
    Context:
    {context}
    
    User Question: {query}
    Answer:
    """
    
    try:
        response = await asyncio.to_thread(
            client.models.generate_content,
            model='gemini-2.5-flash-lite',
            contents=prompt,
        )
        return {
            "query": query, 
            "ai_response": response.text, 
            "requested_by": current_user.username
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI API Error: {str(e)}")