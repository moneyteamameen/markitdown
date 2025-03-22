from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
import tempfile
import os
import mimetypes
import uvicorn
from pydantic import BaseModel
import logging
from typing import Optional
import markdown
from bs4 import BeautifulSoup
import docx2txt
import pypandoc
from markitdown import MarkItDown
import re
from dotenv import load_dotenv
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from fastapi.middleware.trustedhost import TrustedHostMiddleware

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Document to Markdown API")

# Add CORS middleware to allow requests from your Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Add your Next.js app URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# To handle large files, we'll use FastAPI's built-in configuration
# Add these lines after the CORS middleware
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["*"])

# Initialize MarkItDown
markitdown = MarkItDown()

# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(429, _rate_limit_exceeded_handler)

# Function to clean markdown
def clean_markdown(markdown_text):
    """
    Clean up markdown text by removing redundant whitespace, unwanted elements,
    and fixing inconsistent formatting.
    """
    # Remove redundant whitespace
    cleaned = re.sub(r'\n{3,}', '\n\n', markdown_text)
    
    # Remove any unwanted elements (like HTML comments)
    cleaned = re.sub(r'<!--.*?-->', '', cleaned, flags=re.DOTALL)
    
    # Fix inconsistent heading formatting
    cleaned = re.sub(r'###+\s+', '### ', cleaned)
    
    # Remove excessive spaces
    cleaned = re.sub(r' {2,}', ' ', cleaned)
    
    # Fix list formatting
    cleaned = re.sub(r'\n\s*[-*+]\s', '\n- ', cleaned)
    
    # Add handling for code blocks
    cleaned = re.sub(r'```(\w+)?\n\s+', r'```\1\n', cleaned)
    
    # Fix ordered list formatting
    cleaned = re.sub(r'\n\s*(\d+)\.\s+', r'\n\1. ', cleaned)
    
    # Normalize line endings
    cleaned = cleaned.replace('\r\n', '\n')
    
    return cleaned.strip()

class MarkdownResponse(BaseModel):
    markdown: str
    filename: str
    mimetype: Optional[str] = None
    original_size: int
    markdown_size: int

@app.get("/")
async def root():
    return {
        "message": "Document to Markdown API is running",
        "version": "1.0.0",
        "endpoints": {
            "/convert": "POST - Convert a file to markdown",
            "/": "GET - This information"
        }
    }

@app.get("/health")
async def health_check():
    """
    Health check endpoint
    """
    return {"status": "healthy"}

MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB instead of 10MB

@app.post("/convert", response_model=MarkdownResponse)
@limiter.limit("50/minute")
async def convert_to_markdown(
    request: Request,
    file: UploadFile = File(...)
):
    try:
        # Create a temporary file and stream the content
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_file:
            chunk_size = 1024 * 1024  # 1MB chunks
            original_size = 0
            while chunk := await file.read(chunk_size):
                temp_file.write(chunk)
                original_size += len(chunk)
            temp_path = temp_file.name
        
        # Log file information
        logger.info(f"Processing file: {file.filename}, size: {original_size} bytes")
        
        # Get file extension and mimetype from the uploaded file
        filename = file.filename
        extension = os.path.splitext(filename)[1].lower() if filename else ""
        mimetype = file.content_type
        
        # If mimetype is not provided, try to guess it
        if not mimetype and extension:
            mimetype, _ = mimetypes.guess_type(filename)
        
        logger.info(f"Converting file with mimetype: {mimetype}, extension: {extension}")
        
        # Try using MarkItDown first
        try:
            result = markitdown.convert(temp_path)
            markdown_content = result.text_content
            logger.info("Conversion successful using MarkItDown")
        except Exception as e:
            logger.warning(f"MarkItDown conversion failed: {str(e)}")
            
            # Fall back to previous conversion methods
            try:
                # Try using pypandoc as fallback
                markdown_content = pypandoc.convert_file(temp_path, 'md')
            except Exception as e:
                logger.warning(f"Pypandoc conversion failed: {str(e)}")
                
                # Fall back to specific converters based on file type
                if extension in ['.docx', '.doc']:
                    # Convert Word documents
                    text = docx2txt.process(temp_path)
                    markdown_content = text
                elif extension in ['.html', '.htm']:
                    # Convert HTML
                    with open(temp_path, 'r', encoding='utf-8', errors='ignore') as f:
                        html_content = f.read()
                    soup = BeautifulSoup(html_content, 'html.parser')
                    markdown_content = str(soup.get_text())
                elif extension == '.md':
                    # Already markdown
                    with open(temp_path, 'r', encoding='utf-8', errors='ignore') as f:
                        markdown_content = f.read()
                elif extension in ['.txt', '.text']:
                    # Plain text
                    with open(temp_path, 'r', encoding='utf-8', errors='ignore') as f:
                        markdown_content = f.read()
                else:
                    # Unsupported format
                    raise HTTPException(
                        status_code=400,
                        detail=f"Unsupported file format: {extension}"
                    )
        
        # Clean the markdown content
        markdown_content = clean_markdown(markdown_content)
        logger.info("Markdown content cleaned")
        
        # Gemini enhancement has been removed per request.
        
        markdown_size = len(markdown_content)
        logger.info(f"Conversion successful. Original size: {original_size} bytes, Markdown size: {markdown_size} bytes")
        
        # Return the markdown content and file details
        return {
            "markdown": markdown_content,
            "filename": filename,
            "mimetype": mimetype,
            "original_size": original_size,
            "markdown_size": markdown_size
        }
    finally:
        # Clean up the temporary file
        os.unlink(temp_path)
        logger.info(f"Temporary file removed: {temp_path}")

if __name__ == "__main__":
    config = uvicorn.Config(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        limit_concurrency=1000,
        limit_max_requests=1000,
        timeout_keep_alive=5,
        ws_ping_interval=20,
        ws_ping_timeout=20,
        http={'h11_max_incomplete_size': 50_000_000}  # 50MB
    )
    server = uvicorn.Server(config)
    server.run() 