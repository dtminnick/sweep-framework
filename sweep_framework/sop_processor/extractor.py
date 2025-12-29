
"""
Process flow extraction with an LLM.

This module defines `FlowExtractor`, a utility for extracting structured process
flows from document content (e.g., SOPs) using a configured LLM client. It
supports initialization via direct client injection or through an `LLMFactory`,
and handles prompt loading, content truncation, and structured tabular output.

Key features:
    - Load a prompt template from file with flexible path resolution.
    - Build a system instruction + prompt for consistent extraction behavior.
    - Truncate overly long content (configurable threshold).
    - Output structured tabular data instead of JSON.
    - Batch extraction across multiple documents with error isolation.

Example:
    >>> extractor = FlowExtractor.from_factory(
    ...     provider="openai",
    ...     model="gpt-4o-mini",
    ...     prompt_file="prompts/process_flow.txt.tpl",
    ...     temperature=0.0,
    ...     max_tokens=20000
    ... )
    >>> flow = extractor.extract_process_flow(document_content="...text...", document_name="sop.pdf")
    >>> print(flow)  # Returns structured tabular string
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from stream.base_client import BaseClient
from stream.llm_factory import LLMFactory

logger = logging.getLogger(__name__)

class FlowExtractor:
    """
    Extract structured process flows using an LLM.

    Wraps a `BaseClient`-compatible LLM to convert raw document text into
    structured tabular data that represents the process flow (e.g., steps,
    roles, inputs/outputs, decision points, tools).

    Attributes:
        llm: The instantiated LLM client implementing `BaseClient`.
        temperature: Sampling temperature used when calling the LLM.
        max_tokens: Token cap for the LLM response (used if the client supports it).
        prompt_template: The prompt template text loaded from `prompt_file`.
    """
    
    DEFAULT_PROMPT = """You are an expert at analyzing Standard Operating Procedures (SOPs) and extracting structured process flow information.

Analyze the following SOP document and extract the process flow structure. Return the information in a structured tabular format:

PROCESS_NAME: [Name of the process]
PROCESS_DESCRIPTION: [Brief description of the overall process]

STEPS TABLE:
Step_Number|Step_Name|Description|Responsible_Role|Roles|Tools|Inputs|Outputs|Decision_Points|Next_Steps
[For each step, provide one row with pipe-delimited fields. Use semicolons (;) to separate multiple items within fields like Roles, Tools, Inputs, Outputs, Decision_Points, and Next_Steps.]

COMPLIANCE_REQUIREMENTS:
[One requirement per line]

Focus on:
- Sequential flow of steps
- Decision points and branching logic
- Roles and responsibilities (capture within each step)
- Tools and systems used (capture within each step)
- Inputs and outputs for each step

Use pipe (|) as the delimiter between columns in the STEPS TABLE.
For lists within cells (like roles, tools, inputs, outputs, decision_points, next_steps), use semicolons (;) to separate items.
Return ONLY the structured tabular data, no additional explanation."""

    def __init__(self, llm_client: BaseClient,
                 prompt_file: Optional[str] = None,
                 *,
                 temperature: float = 0.0,
                 max_tokens: int = 20000) -> None:
        """
        Initialize the extractor.

        Args:
            llm_client: A concrete `BaseClient` instance (e.g., OpenAIClient, AnthropicClient).
            prompt_file: Path to the prompt template file to load. If None, uses DEFAULT_PROMPT.
            temperature: Sampling temperature for the LLM (0.0 - 1.0 typically).
            max_tokens: Maximum tokens to request for the LLM output.

        Raises:
            FileNotFoundError: If the `prompt_file` cannot be resolved by `_load_prompt_from_file`.
        """

        self.llm: BaseClient = llm_client

        self.temperature: float = temperature

        self.max_tokens: int = max_tokens

        if prompt_file:
            self.prompt_template: str = self._load_prompt_from_file(prompt_file)
        else:
            self.prompt_template: str = self.DEFAULT_PROMPT

    @classmethod
    def from_factory(cls,
                     *,
                     provider: str,
                     model: str,
                     prompt_file: Optional[str] = None,
                     temperature: float = 0.0,
                     max_tokens: int = 20000) -> FlowExtractor:
        """
        Create an extractor using `LLMFactory`.

        Args:
            provider: LLM provider identifier (e.g., "openai", "anthropic").
            model: Model name to instantiate (e.g., "gpt-4o-mini", "claude-3-5-sonnet").
            prompt_file: Path to the prompt template file to load. If None, uses DEFAULT_PROMPT.
            temperature: Sampling temperature to use.
            max_tokens: Maximum tokens to request for the LLM output.

        Returns:
            A configured `FlowExtractor` instance.
        """

        llm = LLMFactory.create(provider = provider, model = model)

        return cls(llm_client = llm, 
                   prompt_file = prompt_file,
                   temperature = temperature,
                   max_tokens = max_tokens)

    def extract_process_flow(self, document_content: str, document_name: str = "") -> str:
        """
        Extract a structured process flow from raw document text.
        
        Returns:
            Structured tabular string containing the extracted process flow.
        """
        print(f"Passed to extractor: {document_name}")
        max_content_length = 50000
        if len(document_content) > max_content_length:
            logger.warning(
                "Document content too long (%d chars); truncating to %d chars.",
                len(document_content),
                max_content_length,
            )
            document_content = document_content[:max_content_length]

        prompt = (
            f"{self.prompt_template}"
            f"--- Document: {document_name} ---"
            f"{document_content}"
        )
        print(f"Document with content extracted: {prompt}")

        # Always keep the raw response so logging is safe even if processing fails
        raw_response: Optional[str] = None
        try:
            raw_response = self._call_llm_raw(prompt)
            
            # Add metadata header to the tabular output
            metadata_header = f"SOURCE_DOCUMENT: {document_name}\n"
            metadata_header += f"EXTRACTION_TEMPERATURE: {self.temperature}\n\n"
            formatted_output = metadata_header + raw_response
            
            logger.info("Successfully extracted process flow from %s", document_name)
            return formatted_output

        except Exception as e:
            logger.error("Extraction failed for %s: %s", document_name, e)
            # Optional: also preview raw response if available
            if raw_response:
                logger.error("Raw response (first 500 chars): %s", raw_response[:500])
            raise


    def extract_from_documents(self, documents: Optional[List[Dict[str, Any]]]) -> List[str]:
        """
        Extract process flows from a list of document records.

        Each document dict is expected to include:
            - "content" (str): raw text.
            - "name" (str, optional): document name.
            - "path" (str, optional): absolute or relative file path.
            - "relative_path" (str, optional): path relative to a root.

        The method continues on errors, logging per-document failures.

        Args:
            documents: A list of document dictionaries, or None/empty.

        Returns:
            A list of extracted process flow strings in structured tabular format.

        Notes:
            Failures are logged and do not halt the loop.
        """

        if not documents:
            logger.warning("No documents provided; returning empty list.")
            return []

        extracted: List[str] = []

        for doc in documents:
            try:
                flow = self.extract_process_flow(document_content = doc["content"],
                                                 document_name = doc.get("name", ""),)
                # Add document path metadata
                path_header = f"DOCUMENT_PATH: {doc.get('path', '')}\n"
                if doc.get('relative_path'):
                    path_header += f"DOCUMENT_RELATIVE_PATH: {doc.get('relative_path', '')}\n"
                path_header += "\n"
                
                formatted_output = path_header + flow
                extracted.append(formatted_output)
            except Exception as e:
                logger.error("Failed extraction for %s: %s", doc.get("name", "<unnamed>"), e)
                continue
        return extracted

    
    def _call_llm_raw(self, prompt: str) -> str:
        """
        Call the underlying LLM client and return raw text (no parsing).
        """
        try:
            return self.llm.chat(prompt, temperature=self.temperature, max_tokens=self.max_tokens)
        except TypeError:
            return self.llm.chat(prompt, temperature=self.temperature)



    def _load_prompt_from_file(self, prompt_file: str) -> str:
        """
        Load the prompt template from disk with flexible resolution.

        Resolution order:
            1) Use `prompt_file` as-is (absolute or current working directory).
            2) Relative to this module's directory.
            3) Relative to the parent directory (project root assumption).

        Args:
            prompt_file: Path or filename of the prompt template.

        Returns:
            The prompt template text as a string.

        Raises:
            FileNotFoundError: If the file cannot be found in any candidate location.
        """

        module_dir = Path(__file__).resolve().parent

        candidate_paths = [
            Path(prompt_file),                         # allow absolute or cwd-relative if caller wants
            module_dir / prompt_file,                  # relative to package/module dir
            module_dir.parent / prompt_file,           # relative to project root (if extractor is in stream/)
        ]

        print(Path(candidate_paths[0]))
        print(Path(candidate_paths[1]))
        print(Path(candidate_paths[2]))

        for p in candidate_paths:
            if p.exists():
                content = p.read_text(encoding="utf-8")
                print(content)
                return content

        raise FileNotFoundError(f"Prompt file not found: {prompt_file} (checked: {', '.join(str(p) for p in candidate_paths)})")


