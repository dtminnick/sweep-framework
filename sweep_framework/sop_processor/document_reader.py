"""
Document Reader Module

Reads documents from a folder structure, supporting multiple file formats:
- PDF files (.pdf)
- Word documents (.docx)
- Text files (.txt)
"""

import os
from pathlib import Path
from typing import List, Dict, Optional, Any
import logging

try:
    import PyPDF2
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

try:
    from docx import Document
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False

logger = logging.getLogger(__name__)


class DocumentReader:
    """
    Reads documents from a folder structure and extracts text content.
    
    Supports:
    - PDF files (.pdf)
    - Word documents (.docx)
    - Text files (.txt)
    """
    
    SUPPORTED_EXTENSIONS = {'.pdf', '.docx', '.txt'}
    
    def __init__(self, root_folder: str, recursive: bool = True):
        """
        Initialize the document reader.
        
        Args:
            root_folder: Root folder path containing documents
            recursive: If True, search subdirectories recursively
        """
        self.root_folder = Path(root_folder)
        if not self.root_folder.exists():
            raise ValueError(f"Folder does not exist: {root_folder}")
        self.recursive = recursive
        
        # Check dependencies
        if not PDF_AVAILABLE:
            logger.warning("PyPDF2 not available. PDF files will be skipped.")
        if not DOCX_AVAILABLE:
            logger.warning("python-docx not available. DOCX files will be skipped.")
    
    def find_documents(self) -> List[Path]:
        """
        Find all supported documents in the folder structure.
        
        Returns:
            List of file paths to documents
        """
        documents = []
        
        if self.recursive:
            pattern = "**/*"
        else:
            pattern = "*"
        
        for file_path in self.root_folder.glob(pattern):
            if file_path.is_file() and file_path.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                documents.append(file_path)
        
        logger.info(f"Found {len(documents)} documents in {self.root_folder}")
        return sorted(documents)
    
    def read_pdf(self, file_path: Path) -> str:
        """Extract text from a PDF file."""
        if not PDF_AVAILABLE:
            raise ImportError("PyPDF2 is required to read PDF files. Install with: pip install PyPDF2")
        
        text_content = []
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text_content.append(page.extract_text())
            return '\n'.join(text_content)
        except Exception as e:
            logger.error(f"Error reading PDF {file_path}: {e}")
            raise
    
    def read_docx(self, file_path: Path) -> str:
        """Extract text from a Word document."""
        if not DOCX_AVAILABLE:
            raise ImportError("python-docx is required to read DOCX files. Install with: pip install python-docx")
        
        try:
            doc = Document(file_path)
            paragraphs = [para.text for para in doc.paragraphs]
            return '\n'.join(paragraphs)
        except Exception as e:
            logger.error(f"Error reading DOCX {file_path}: {e}")
            raise
    
    def read_txt(self, file_path: Path) -> str:
        """Read text from a plain text file."""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                return file.read()
        except Exception as e:
            logger.error(f"Error reading TXT {file_path}: {e}")
            raise
    
    def read_document(self, file_path: Path) -> Dict[str, Any]:
        """
        Read a document and return its content with metadata.
        
        Args:
            file_path: Path to the document
            
        Returns:
            Dictionary with keys:
            - 'path': Original file path
            - 'name': File name
            - 'extension': File extension
            - 'content': Extracted text content
        """
        extension = file_path.suffix.lower()
        
        if extension == '.pdf':
            content = self.read_pdf(file_path)
        elif extension == '.docx':
            content = self.read_docx(file_path)
        elif extension == '.txt':
            content = self.read_txt(file_path)
        else:
            raise ValueError(f"Unsupported file type: {extension}")
        
        return {
            'path': str(file_path),
            'name': file_path.name,
            'extension': extension,
            'content': content,
            'relative_path': str(file_path.relative_to(self.root_folder))
        }
    
    def read_all_documents(self) -> List[Dict[str, Any]]:
        """
        Read all documents from the folder structure.
        
        Returns:
            List of document dictionaries
        """
        documents = []
        file_paths = self.find_documents()
        
        for file_path in file_paths:
            try:
                doc_data = self.read_document(file_path)
                documents.append(doc_data)
                logger.info(f"Successfully read: {file_path.name}")
            except Exception as e:
                logger.error(f"Failed to read {file_path}: {e}")
                continue
        
        return documents

