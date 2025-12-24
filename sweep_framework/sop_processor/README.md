# SOP Processor Module

This module provides functionality to read Standard Operating Procedure (SOP) documents from a folder structure, extract process flow information using Large Language Models (LLMs), and store the structured data in a database.

## Features

- **Document Reading**: Supports PDF, DOCX, and TXT files
- **LLM Extraction**: Uses OpenAI models to extract structured process flows
- **Database Storage**: Stores extracted data in SQLite with normalized schema
- **Pipeline Orchestration**: End-to-end processing workflow

## Installation

Install required dependencies:

```bash
pip install -r requirements.txt
```

Required packages:
- `PyPDF2` - For reading PDF files
- `python-docx` - For reading Word documents
- One of the following for LLM access:
  - `openai` - For OpenAI API access
  - `anthropic` - For Anthropic/Claude API access
  - `requests` - For custom/OpenAI-compatible APIs (e.g., Ollama)

## Setup

Set your LLM provider API key based on which provider you're using:

### OpenAI
```bash
export OPENAI_API_KEY="your-api-key-here"
```

### Anthropic (Claude)
```bash
export ANTHROPIC_API_KEY="your-api-key-here"
```

### Custom/OpenAI-compatible (e.g., Ollama)
No API key required for local instances, but you'll need to provide the base URL.
For Ollama, typically: `http://localhost:11434/v1`

You can also provide the API key via command-line argument.

## Usage

### Command Line

Process all documents in a folder:

```bash
python process_sops.py --folder /path/to/sop/documents --db process_flows.db
```

Options:
- `--folder`: Path to folder containing SOP documents (required)
- `--db`: Path to SQLite database file (default: `process_flows.db`)
- `--api-key`: API key for LLM provider (or use environment variable)
- `--model`: LLM model to use (default: `gpt-4o-mini`)
- `--provider`: LLM provider - `openai`, `anthropic`, or `custom` (default: `openai`)
- `--api-base-url`: Base URL for custom provider (required if `--provider=custom`)
- `--no-recursive`: Don't search subdirectories
- `--verbose`: Enable verbose logging

### Example: Using Anthropic (Claude)
```bash
python process_sops.py --folder /path/to/documents --provider anthropic --model claude-3-5-sonnet-20241022
```

### Example: Using Ollama (Local)
```bash
# Start Ollama and pull a model first:
# ollama pull llama3.1

python process_sops.py --folder /path/to/documents --provider custom --api-base-url http://localhost:11434/v1 --model llama3.1
```

### Example: Using OpenAI
```bash
python process_sops.py --folder /path/to/documents --provider openai --model gpt-4o-mini
```

### Python API

```python
from sweep_framework.sop_processor.pipeline import SOPProcessorPipeline

# Initialize pipeline with OpenAI
pipeline = SOPProcessorPipeline(
    documents_folder="/path/to/documents",
    db_path="process_flows.db",
    llm_api_key="your-api-key",  # or set OPENAI_API_KEY env var
    llm_model="gpt-4o-mini",
    llm_provider="openai"
)

# Or with Anthropic/Claude
pipeline = SOPProcessorPipeline(
    documents_folder="/path/to/documents",
    db_path="process_flows.db",
    llm_api_key="your-api-key",  # or set ANTHROPIC_API_KEY env var
    llm_model="claude-3-5-sonnet-20241022",
    llm_provider="anthropic"
)

# Or with custom/OpenAI-compatible API (e.g., Ollama)
pipeline = SOPProcessorPipeline(
    documents_folder="/path/to/documents",
    db_path="process_flows.db",
    llm_model="llama3.1",
    llm_provider="custom",
    llm_api_base_url="http://localhost:11434/v1"
)

# Process all documents
result = pipeline.process_all()

# Close database connection
pipeline.close()
```

### Individual Components

You can also use components individually:

```python
from sweep_framework.sop_processor import DocumentReader, ProcessFlowExtractor, ProcessFlowDatabase

# Read documents
reader = DocumentReader("/path/to/documents")
documents = reader.read_all_documents()

# Extract process flows with OpenAI
extractor = ProcessFlowExtractor(api_key="your-api-key", provider="openai")

# Or with Anthropic
extractor = ProcessFlowExtractor(api_key="your-api-key", provider="anthropic", model="claude-3-5-sonnet-20241022")

# Or with custom/OpenAI-compatible API
extractor = ProcessFlowExtractor(
    api_key="not-required-for-local",
    provider="custom",
    model="llama3.1",
    api_base_url="http://localhost:11434/v1"
)

process_flows = extractor.extract_from_documents(documents)

# Store in database
db = ProcessFlowDatabase("process_flows.db")
db.insert_multiple(process_flows)
db.close()
```

## Database Schema

The module creates the following tables:

### `process_flows`
Main table storing process flow metadata:
- `id`: Primary key
- `process_name`: Name of the process
- `process_description`: Description
- `source_document`: Original document name
- `document_path`: Full path to document
- `extraction_model`: LLM model used
- `raw_data`: Complete JSON data
- `created_at`: Timestamp

### `process_steps`
Individual steps in the process:
- `id`: Primary key
- `process_flow_id`: Foreign key to `process_flows`
- `step_number`: Step sequence number
- `step_name`: Name of the step
- `description`: Detailed description
- `responsible_role`: Role responsible
- `inputs`: JSON array of inputs
- `outputs`: JSON array of outputs
- `decision_points`: JSON array of decisions
- `next_steps`: JSON array of next step numbers

### `process_roles`
Roles involved in the process:
- `id`: Primary key
- `process_flow_id`: Foreign key
- `role_name`: Name of the role

### `process_tools`
Tools and systems used:
- `id`: Primary key
- `process_flow_id`: Foreign key
- `tool_name`: Name of tool/system

### `compliance_requirements`
Compliance requirements:
- `id`: Primary key
- `process_flow_id`: Foreign key
- `requirement`: Requirement text

## Process Flow Structure

The LLM extracts data in the following JSON structure:

```json
{
    "process_name": "Employee Onboarding Process",
    "process_description": "Complete process for onboarding new employees",
    "steps": [
        {
            "step_number": 1,
            "step_name": "Submit Application",
            "description": "Candidate submits job application",
            "responsible_role": "HR Manager",
            "inputs": ["Application Form", "Resume"],
            "outputs": ["Application Record"],
            "decision_points": ["Application Valid?"],
            "next_steps": [2, 3]
        }
    ],
    "roles": ["HR Manager", "Department Head"],
    "tools_systems": ["HRIS", "Email System"],
    "compliance_requirements": ["GDPR", "Labor Law"]
}
```

## Customization

### Custom Prompt

You can provide a custom prompt for extraction:

```python
custom_prompt = """Your custom prompt here..."""

extractor = ProcessFlowExtractor(
    api_key="your-api-key",
    custom_prompt=custom_prompt
)
```

### Different LLM Providers and Models

**OpenAI Models:**
- `gpt-4o-mini` (default, cost-effective)
- `gpt-4o`
- `gpt-4-turbo`
- `gpt-3.5-turbo`

**Anthropic/Claude Models:**
- `claude-3-5-sonnet-20241022`
- `claude-3-opus-20240229`
- `claude-3-sonnet-20240229`
- `claude-3-haiku-20240307`

**Custom/OpenAI-compatible APIs:**
Works with any OpenAI-compatible API endpoint, including:
- **Ollama** (local models): `http://localhost:11434/v1`
  - Models: `llama3.1`, `mistral`, `codellama`, etc.
- **vLLM** servers
- **Local inference servers**
- Any API that follows OpenAI's chat completion format

## Querying the Database

Example queries:

```python
import sqlite3

conn = sqlite3.connect("process_flows.db")
cursor = conn.cursor()

# List all processes
cursor.execute("SELECT * FROM process_flows")
processes = cursor.fetchall()

# Get steps for a process
cursor.execute("""
    SELECT * FROM process_steps 
    WHERE process_flow_id = ?
    ORDER BY step_number
""", (process_id,))
steps = cursor.fetchall()

conn.close()
```

## Error Handling

The pipeline handles errors gracefully:
- Documents that fail to read are logged and skipped
- LLM extraction failures are logged and skipped
- Database insertion failures are logged and skipped
- Processing continues for remaining documents

## Logging

Enable verbose logging to see detailed progress:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

Or use the `--verbose` flag in the command line.

