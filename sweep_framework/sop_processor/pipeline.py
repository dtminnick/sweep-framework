"""
Main Pipeline for SOP Processing

Orchestrates the entire workflow:
1. Read documents from folder
2. Extract process flows using LLM
3. Store in database
"""

import logging
from typing import Optional, List, Dict, Any
from pathlib import Path

from .document_reader import DocumentReader
from .llm_extractor import ProcessFlowExtractor
from .database_storage import ProcessFlowDatabase

logger = logging.getLogger(__name__)


class SOPProcessorPipeline:
    """
    Main pipeline for processing SOP documents.
    """
    
    def __init__(
        self,
        documents_folder: str,
        db_path: str = "process_flows.db",
        llm_api_key: Optional[str] = None,
        llm_model: str = "gpt-4o-mini",
        llm_provider: str = "openai",
        llm_api_base_url: Optional[str] = None,
        recursive: bool = True
    ):
        """
        Initialize the pipeline.
        
        Args:
            documents_folder: Path to folder containing SOP documents
            db_path: Path to SQLite database file
            llm_api_key: API key for the LLM provider (or set via environment variable)
            llm_model: LLM model to use
            llm_provider: LLM provider - "openai", "anthropic", or "custom" (default: "openai")
            llm_api_base_url: Base URL for custom API provider (required if provider="custom")
            recursive: Whether to search subdirectories recursively
        """
        self.documents_folder = documents_folder
        self.db_path = db_path
        
        # Initialize components
        self.document_reader = DocumentReader(documents_folder, recursive=recursive)
        self.llm_extractor = ProcessFlowExtractor(
            api_key=llm_api_key,
            model=llm_model,
            provider=llm_provider,
            api_base_url=llm_api_base_url
        )
        self.database = ProcessFlowDatabase(db_path)
    
    def process_all(self) -> Dict[str, Any]:
        """
        Process all documents in the folder.
        
        Returns:
            Dictionary with processing results and statistics
        """
        logger.info(f"Starting SOP processing pipeline for: {self.documents_folder}")
        
        # Step 1: Read documents
        logger.info("Step 1: Reading documents...")
        documents = self.document_reader.read_all_documents()
        logger.info(f"Read {len(documents)} documents")
        
        if not documents:
            logger.warning("No documents found to process")
            return {
                'status': 'no_documents',
                'documents_read': 0,
                'process_flows_extracted': 0,
                'process_flows_stored': 0
            }
        
        # Step 2: Extract process flows
        logger.info("Step 2: Extracting process flows using LLM...")
        process_flows = self.llm_extractor.extract_from_documents(documents)
        logger.info(f"Extracted {len(process_flows)} process flows")
        
        # Step 3: Store in database
        logger.info("Step 3: Storing process flows in database...")
        stored_ids = self.database.insert_multiple(process_flows)
        logger.info(f"Stored {len(stored_ids)} process flows in database")
        
        # Summary
        result = {
            'status': 'success',
            'documents_read': len(documents),
            'process_flows_extracted': len(process_flows),
            'process_flows_stored': len(stored_ids),
            'database_path': self.db_path,
            'process_flow_ids': stored_ids
        }
        
        logger.info("Pipeline completed successfully")
        return result
    
    def process_single_document(self, document_path: str) -> Optional[int]:
        """
        Process a single document.
        
        Args:
            document_path: Path to the document file
            
        Returns:
            Process flow ID if successful, None otherwise
        """
        logger.info(f"Processing single document: {document_path}")
        
        # Read document
        doc_data = self.document_reader.read_document(Path(document_path))
        
        # Extract process flow
        process_flow = self.llm_extractor.extract_process_flow(
            doc_data['content'],
            doc_data['name']
        )
        process_flow['document_path'] = doc_data['path']
        process_flow['document_relative_path'] = doc_data.get('relative_path', '')
        
        # Store in database
        flow_id = self.database.insert_process_flow(process_flow)
        
        logger.info(f"Successfully processed document, stored as process flow ID: {flow_id}")
        return flow_id
    
    def close(self):
        """Close database connection."""
        self.database.close()

