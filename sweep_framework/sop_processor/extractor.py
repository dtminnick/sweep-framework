
"""
Process flow extraction with an LLM.

This module defines `FlowExtractor`, a utility for extracting structured process
flows from document content (e.g., SOPs) using a configured LLM client. It
supports initialization via direct client injection or through an `LLMFactory`,
and handles prompt loading, content truncation, and robust JSON extraction from
LLM responses.

Key features:
    - Load a prompt template from file with flexible path resolution.
    - Build a system instruction + prompt for consistent extraction behavior.
    - Truncate overly long content (configurable threshold).
    - Enforce that responses are a single valid JSON object (strip fences, clean).
    - Batch extraction across multiple documents with error isolation.

Example:
    >>> extractor = FlowExtractor.from_factory(
    ...     provider="openai",
    ...     model="gpt-4o-mini",
    ...     prompt_file="prompts/process_flow.json.tpl",
    ...     temperature=0.0,
    ...     max_tokens=20000
    ... )
    >>> flow = extractor.extract_process_flow(document_content="...text...", document_name="sop.pdf")
    >>> flow["process_name"]
    'Year-End Processing'
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
import re
from typing import Any, Dict, List, Optional

from stream.base_client import BaseClient
from stream.llm_factory import LLMFactory

logger = logging.getLogger(__name__)

class FlowExtractor:
    """
    Extract structured process flows using an LLM.

    Wraps a `BaseClient`-compatible LLM to convert raw document text into a
    normalized JSON object that represents the process flow (e.g., steps,
    roles, inputs/outputs, decision points).

    Attributes:
        llm: The instantiated LLM client implementing `BaseClient`.
        temperature: Sampling temperature used when calling the LLM.
        max_tokens: Token cap for the LLM response (used if the client supports it).
        prompt_template: The prompt template text loaded from `prompt_file`.
    """

    def __init__(self, llm_client: BaseClient,
                 prompt_file: str,
                 *,
                 temperature: float = 0.0,
                 max_tokens: int = 20000) -> None:
        """
        Initialize the extractor.

        Args:
            llm_client: A concrete `BaseClient` instance (e.g., OpenAIClient, AnthropicClient).
            prompt_file: Path to the prompt template file to load.
            temperature: Sampling temperature for the LLM (0.0 - 1.0 typically).
            max_tokens: Maximum tokens to request for the LLM output.

        Raises:
            FileNotFoundError: If the `prompt_file` cannot be resolved by `_load_prompt_from_file`.
        """

        self.llm: BaseClient = llm_client

        self.temperature: float = temperature

        self.max_tokens: int = max_tokens

        self.prompt_template: str = self._load_prompt_from_file(prompt_file)

    @classmethod
    def from_factory(cls,
                     *,
                     provider: str,
                     model: str,
                     prompt_file: str,
                     temperature: float = 0.0,
                     max_tokens: int = 20000) -> FlowExtractor:
        """
        Create an extractor using `LLMFactory`.

        Args:
            provider: LLM provider identifier (e.g., "openai", "anthropic").
            model: Model name to instantiate (e.g., "gpt-4o-mini", "claude-3-5-sonnet").
            prompt_file: Path to the prompt template file to load.
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

    def extract_process_flow(self, document_content: str, document_name: str = "") -> Dict[str, Any]:
        """
        Extract a structured process flow from raw document text.
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

        # Always keep the raw response so logging is safe even if parsing fails
        raw_response: Optional[str] = None
        try:
            raw_response = self._call_llm_raw(prompt)  # new helper returns raw text only
            response_text = self._ensure_json_object(raw_response)
            process_flow = json.loads(response_text)
            process_flow["source_document"] = document_name
            process_flow["extraction_temperature"] = self.temperature
            logger.info("Successfully extracted process flow from %s", document_name)
            return process_flow

        except json.JSONDecodeError as e:
            logger.error("Invalid JSON returned for %s: %s", document_name, e)
            preview = (raw_response or "")[:500]
            logger.error("Response preview: %s", preview)
            raise ValueError("LLM returned invalid JSON") from e

        except Exception as e:
            logger.error("Extraction failed for %s: %s", document_name, e)
            # Optional: also preview raw response if available
            if raw_response:
                logger.error("Raw response (first 500 chars): %s", raw_response[:500])
            raise


    def extract_from_documents(self, documents: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
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
            A list of extracted process flow dictionaries.

        Notes:
            Failures are logged and do not halt the loop.
        """

        if not documents:
            logger.warning("No documents provided; returning empty list.")
            return []

        extracted: List[Dict[str, Any]] = []

        for doc in documents:
            try:
                flow = self.extract_process_flow(document_content = doc["content"],
                                                 document_name = doc.get("name", ""),)
                flow["document_path"] = doc.get("path", "")
                flow["document_relative_path"] = doc.get("relative_path", "")
                extracted.append(flow)
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

    def _call_llm(self, prompt: str) -> str:
        """
        Backwards-compatible wrapper (kept if other callers still use it).
        """
        raw = self._call_llm_raw(prompt)
        return self._ensure_json_object(raw)


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
                return p.read_text(encoding="utf-8")
                print(p.read_text(encoding="utf-8"))

        raise FileNotFoundError(f"Prompt file not found: {prompt_file} (checked: {', '.join(str(p) for p in candidate_paths)})")

    

    def _ensure_json_object(self, text: str) -> str:
        """
        Ensure output is (or becomes) a single valid JSON object.

        - Strips markdown fences.
        - Attempts direct parse.
        - Scans for the longest balanced JSON prefix (handles strings/escapes).
        - Applies minimal repairs for truncation: close strings, brackets, and braces.
        - Normalizes artifacts (backticks, comments, trailing commas, smart quotes).
        """
        if not text:
            raise ValueError("LLM returned empty response")

        s = text.strip()

        # 1) Strip markdown fences: ```json ... ``` or ```
        fence_match = re.match(r"^\s*```(?:json)?\s*([\s\S]*?)\s*```\s*$", s, flags=re.DOTALL | re.IGNORECASE)
        if fence_match:
            s = fence_match.group(1).strip()

        # 2) Try direct parse first
        try:
            json.loads(s)
            return s
        except Exception:
            pass

        # Helper: scan and find longest balanced JSON prefix (top-level object or array)
        def longest_balanced_prefix(src: str) -> tuple[str, dict]:
            in_string = False
            escape = False
            brace_depth = 0  # {}
            bracket_depth = 0  # []
            start_idx = None
            last_complete_end = None

            for i, ch in enumerate(src):
                if in_string:
                    if escape:
                        escape = False
                    elif ch == "\\":
                        escape = True
                    elif ch == '"':
                        in_string = False
                    # raw newline inside a JSON string -> keep, we will normalize later
                else:
                    if ch == '"':
                        in_string = True
                    elif ch == "{":
                        if brace_depth == 0 and bracket_depth == 0 and start_idx is None:
                            start_idx = i
                        brace_depth += 1
                    elif ch == "}":
                        if brace_depth > 0:
                            brace_depth -= 1
                            if brace_depth == 0 and bracket_depth == 0 and start_idx is not None:
                                last_complete_end = i + 1  # inclusive end
                    elif ch == "[":
                        if brace_depth == 0 and bracket_depth == 0 and start_idx is None:
                            start_idx = i
                        bracket_depth += 1
                    elif ch == "]":
                        if bracket_depth > 0:
                            bracket_depth -= 1
                            if bracket_depth == 0 and brace_depth == 0 and start_idx is not None:
                                last_complete_end = i + 1

            meta = {
                "in_string": in_string,
                "brace_depth": brace_depth,
                "bracket_depth": bracket_depth,
                "start_idx": start_idx,
                "last_complete_end": last_complete_end,
            }
            if last_complete_end is not None and start_idx is not None:
                return src[start_idx:last_complete_end], meta
            # If no complete object found but a start exists, return from start to end (truncated)
            if start_idx is not None:
                return src[start_idx:], meta
            return "", meta

        candidate, meta = longest_balanced_prefix(s)
        if not candidate:
            raise ValueError("Could not locate a top-level JSON object in the LLM response")

        cleaned = candidate

        # 3) Normalize artifacts
        # Remove stray backticks
        cleaned = re.sub(r"[`]+", "", cleaned)
        # Remove JS/JSON5 comments
        cleaned = re.sub(r"//.*?$", "", cleaned, flags=re.MULTILINE)
        cleaned = re.sub(r"/\*[\s\S]*?\*/", "", cleaned)
        # Normalize smart quotes
        
        cleaned = (
            cleaned
            .replace("\u201C", '"')  # left double smart quote
            .replace("\u201D", '"')  # right double smart quote
            .replace("\u2018", "'")  # left single smart quote
            .replace("\u2019", "'")  # right single smart quote
        )


        # 4) Repair common issues if truncated
        # a) Close unterminated string
        if meta["in_string"]:
            cleaned += '"'

        # b) Close arrays/brackets
        if meta["bracket_depth"] > 0:
            cleaned += "]" * meta["bracket_depth"]

        # c) Close objects/braces
        if meta["brace_depth"] > 0:
            cleaned += "}" * meta["brace_depth"]

        # d) Remove trailing commas before } or ]
        cleaned = re.sub(r",\s*(\})", r"\1", cleaned)
        cleaned = re.sub(r",\s*(\])", r"\1", cleaned)

        # e) Replace raw newlines inside quoted strings with \n
        #    We walk through the text and if inside a string, convert literal newlines.
        def fix_newlines_in_strings(src: str) -> str:
            out = []
            in_string = False
            escape = False
            for ch in src:
                if in_string:
                    if escape:
                        out.append(ch)
                        escape = False
                    elif ch == "\\":
                        out.append(ch)
                        escape = True
                    elif ch == '"':
                        out.append(ch)
                        in_string = False
                    elif ch == "\n" or ch == "\r":
                        out.append("\\n")
                    else:
                        out.append(ch)
                else:
                    out.append(ch)
                    if ch == '"':
                        in_string = True
            return "".join(out)

        cleaned = fix_newlines_in_strings(cleaned)

        # 5) Final parse
        try:
            json.loads(cleaned)
            return cleaned
        except json.JSONDecodeError as e:
            snippet = cleaned[:500]
            raise json.JSONDecodeError(
                f"Sanitized JSON still invalid: {e.msg}. Snippet: {snippet}",
                cleaned,
                e.pos
            )


