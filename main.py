import os
import asyncio
import uuid 
import chromadb
import fitz 
import bcrypt 
import jwt 
from datetime import datetime, timedelta 
from fastapi import FastAPI, HTTPException, UploadFile, File, Depends, status, Request 
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from google import genai
from dotenv import load_dotenv
from sqlalchemy import create_engine, Column, String, Integer, DateTime
from sqlalchemy.orm import sessionmaker, declarative_base, Session
import json
import redis.asyncio as aioredis # BACK TO STANDARD REDIS CACHING

# IMPORTS FOR RATE LIMITING & CORS
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

load_dotenv()

# APP Instance
app = FastAPI()

# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# 1. CORS & RATE LIMITER SETUP
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

# Allows the HTML frontend to talk to this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
    "http://localhost:5173",  
    "http://127.0.0.1:5173",
    "http://localhost:3000",
], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Limits users to prevent API spam Using their IP address
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# 2. AUTHENTICATION CONFIGURATION
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
SECRET_KEY = os.getenv("JWT_SECRET_KEY")
if not SECRET_KEY:
    raise RuntimeError("JWT_SECRET_KEY is not set!")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/login")

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


# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# 3. POSTGRESQL & REDIS DATABASE SETUP
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

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
    owner_id = Column(Integer) 
    upload_date = Column(DateTime, default=datetime.utcnow)

# Wipe Supabase clean every time the server restarts (Ephemeral Mode)
Base.metadata.drop_all(bind=engine)
Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# $$$$$$$ REDIS SETUP $$$$$$$
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
redis_client = aioredis.from_url(REDIS_URL, decode_responses=True)


# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# 4. CURRENT USER DEPENDENCY
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
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


# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# 5. AI & EPHEMERAL VECTOR DB INITIALIZATION
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
client = genai.Client()

chroma_client = chromadb.EphemeralClient() 
collection = chroma_client.get_or_create_collection(name="awen_collection")

@app.get("/")
async def root():
    return {"message": "Awen is SECURE, Full-Stack, Rate-Limited, Cached (Redis), and running in Clean Slate Mode!"}


# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# 6. AUTHENTICATION ENDPOINTS
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
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


# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# TEXT CHUNKING HELPER
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
def chunk_text(text: str, chunk_size: int = 500, chunk_overlap: int = 50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - chunk_overlap 
    return chunks


# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# 7. SECURED CORE ENDPOINTS
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
ALLOWED_CONTENT_TYPES = {"text/plain", "application/pdf"}

@app.post("/upload")
async def upload_document(
    file: UploadFile = File(...), 
    db: Session = Depends(get_db),
    current_user: UserRecord = Depends(get_current_user) 
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
                text += str(page.get_text()) 
            pdf_document.close()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to parse PDF: {str(e)}")
    
    chunks = chunk_text(text)
    if not chunks:
        raise HTTPException(status_code=400, detail="No readable text found.")
    
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
@limiter.limit("5/minute")
async def query_rag(request: Request, query: str, current_user: UserRecord = Depends(get_current_user)):
    
    # $$$$$$$ REDIS EXACT MATCH CACHE CHECK $$$$$$$
    normalized_query = query.strip().lower()
    redis_key = f"cache:user_{current_user.id}:query_{normalized_query}"
    
    cached_redis_result = await redis_client.get(redis_key)
    
    if cached_redis_result:
        print("⚡ REDIS CACHE HIT!")
        return json.loads(cached_redis_result)

    print("🐢 CACHE MISS! Asking Gemini...")
    
    # $$$$$$$ IF NOT CACHED, PROCEED TO VECTOR SEARCH $$$$$$$
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
        
        final_response = {
            "query": query, 
            "ai_response": response.text, 
            "requested_by": current_user.username,
            "source": "Gemini AI"
        }
        
        # --- SAVE TO REDIS CACHE ---
        await redis_client.set(redis_key, json.dumps(final_response), ex=3600)
        
        return final_response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI API Error: {str(e)}")
