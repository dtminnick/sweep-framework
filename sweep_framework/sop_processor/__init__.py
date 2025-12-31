"""
SOP Processor Module

This module provides functionality to:
- Read documents from folder structures
- Extract process flow structures using LLMs
- Store structured data in a database
- Log errors with stack traces to SQLite database
"""

from .document_reader import DocumentReader
from .llm_extractor import ProcessFlowExtractor
from .database_storage import ProcessFlowDatabase
from .error_logger import ErrorLogger

__all__ = ['DocumentReader', 'ProcessFlowExtractor', 'ProcessFlowDatabase', 'ErrorLogger']

