"""
Main script to process SOP documents.

Usage:
    python process_sops.py --folder <documents_folder> [options]
"""

import argparse
import logging
import sys
from pathlib import Path

from sweep_framework.sop_processor.pipeline import SOPProcessorPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description='Process SOP documents and extract process flows'
    )
    parser.add_argument(
        '--folder',
        type=str,
        required=True,
        help='Path to folder containing SOP documents'
    )
    parser.add_argument(
        '--db',
        type=str,
        default='process_flows.db',
        help='Path to SQLite database file (default: process_flows.db)'
    )
    parser.add_argument(
        '--api-key',
        type=str,
        default=None,
        help='API key for LLM provider (or set via environment variable: OPENAI_API_KEY, ANTHROPIC_API_KEY)'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='gpt-4o-mini',
        help='LLM model to use (default: gpt-4o-mini)'
    )
    parser.add_argument(
        '--provider',
        type=str,
        default='openai',
        choices=['openai', 'anthropic', 'custom'],
        help='LLM provider to use: openai, anthropic, or custom (default: openai)'
    )
    parser.add_argument(
        '--api-base-url',
        type=str,
        default=None,
        help='Base URL for custom API provider (required if --provider=custom, e.g., http://localhost:11434/v1 for Ollama)'
    )
    parser.add_argument(
        '--no-recursive',
        action='store_true',
        help='Do not search subdirectories recursively'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate folder exists
    folder_path = Path(args.folder)
    if not folder_path.exists():
        logger.error(f"Folder does not exist: {args.folder}")
        sys.exit(1)
    
    if not folder_path.is_dir():
        logger.error(f"Path is not a directory: {args.folder}")
        sys.exit(1)
    
    # Initialize and run pipeline
    try:
        pipeline = SOPProcessorPipeline(
            documents_folder=str(folder_path),
            db_path=args.db,
            llm_api_key=args.api_key,
            llm_model=args.model,
            llm_provider=args.provider,
            llm_api_base_url=args.api_base_url,
            recursive=not args.no_recursive
        )
        
        result = pipeline.process_all()
        
        # Print summary
        print("\n" + "="*60)
        print("Processing Summary")
        print("="*60)
        print(f"Documents read: {result['documents_read']}")
        print(f"Process flows extracted: {result['process_flows_extracted']}")
        print(f"Process flows stored: {result['process_flows_stored']}")
        print(f"Database: {result['database_path']}")
        print("="*60)
        
        pipeline.close()
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()

