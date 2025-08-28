import PyPDF2
import fitz  # PyMuPDF 
from typing import List, Dict
import hashlib
import os
from datetime import datetime

class PDFProcessor:
    def __init__(self):
        self.supported_formats = ['.pdf']
    
    def extract_text_pypdf2(self, file_path: str) -> str:
        """Extract text using PyPDF2"""
        try:
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n"
                return text
        except Exception as e:
            raise Exception(f"Error extracting text with PyPDF2: {str(e)}")
    
    def extract_text_pymupdf(self, file_path: str) -> str:
        """Extract text using PyMuPDF (fallback method)"""
        try:
            doc = fitz.open(file_path)
            text = ""
            for page in doc:
                text += page.get_text() + "\n"
            doc.close()
            return text
        except Exception as e:
            raise Exception(f"Error extracting text with PyMuPDF: {str(e)}")
    
    def process_pdf(self, file_path: str, filename: str) -> Dict:
        """Process PDF and extract text with metadata"""
        try:
            # Try PyPDF2 first, fallback to PyMuPDF
            try:
                text = self.extract_text_pypdf2(file_path)
            except:
                text = self.extract_text_pymupdf(file_path)
            
            # Clean text
            text = self.clean_text(text)
            
            # Generate document ID
            doc_id = self.generate_document_id(filename, text)
            
            # Get file stats
            file_stats = os.stat(file_path)
            
            return {
                "id": doc_id,
                "filename": filename,
                "text": text,
                "size": file_stats.st_size,
                "upload_date": datetime.now(),
                "word_count": len(text.split())
            }
        except Exception as e:
            raise Exception(f"Error processing PDF {filename}: {str(e)}")
    
    def clean_text(self, text: str) -> str:
        """Clean extracted text"""
        # Remove excessive whitespace
        lines = text.split('\n')
        cleaned_lines = []
        for line in lines:
            line = line.strip()
            if line:
                cleaned_lines.append(line)
        return '\n'.join(cleaned_lines)
    
    def generate_document_id(self, filename: str, text: str) -> str:
        """Generate unique document ID"""
        content = f"{filename}_{text[:100]}"
        return hashlib.md5(content.encode()).hexdigest()
