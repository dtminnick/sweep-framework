"""
LLM-based Process Flow Extractor

Uses Large Language Models to extract process flow structures from SOP documents.
"""

import json
import logging
from typing import Dict, List, Optional, Any
import os
from pathlib import Path

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

logger = logging.getLogger(__name__)


class ProcessFlowExtractor:
    """
    Extracts process flow structures from SOP documents using LLM.
    """
    
    DEFAULT_PROMPT = """You are an expert at analyzing Standard Operating Procedures (SOPs) and extracting structured process flow information.

Analyze the following SOP document and extract the process flow structure. Return a JSON object with the following structure:

{{
    "process_name": "Name of the process",
    "process_description": "Brief description of the overall process",
    "steps": [
        {{
            "step_number": 1,
            "step_name": "Name of the step",
            "description": "Detailed description of what happens in this step",
            "responsible_role": "Role or person responsible for this step",
            "inputs": ["List of inputs required"],
            "outputs": ["List of outputs produced"],
            "decision_points": ["Any decision points or conditions"],
            "next_steps": [2, 3]  // Step numbers that follow this step
        }}
    ],
    "roles": ["List of all roles involved in the process"],
    "tools_systems": ["List of tools or systems used"],
    "compliance_requirements": ["Any compliance or regulatory requirements mentioned"]
}}

Focus on:
- Sequential flow of steps
- Decision points and branching logic
- Roles and responsibilities
- Inputs and outputs for each step
- Tools, systems, or resources used

Return ONLY valid JSON, no additional text or explanation."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        temperature: float = 0.1,
        max_tokens: int = 2000,
        custom_prompt: Optional[str] = None,
        prompt_file: Optional[str] = None,
        provider: str = "openai",
        api_base_url: Optional[str] = None
    ):
        """
        Initialize the process flow extractor.
        
        Args:
            api_key: API key for the LLM provider (or set via environment variable)
            model: Model to use (default: gpt-4o-mini)
            temperature: Sampling temperature (lower = more deterministic)
            max_tokens: Maximum tokens in response
            custom_prompt: Custom prompt template (optional, overridden by prompt_file)
            prompt_file: Path to text file containing the prompt template
            provider: LLM provider to use - "openai", "anthropic", or "custom" (default: "openai")
            api_base_url: Base URL for custom API provider (required if provider="custom")
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.provider = provider.lower()
        self.api_base_url = api_base_url
        
        # Load prompt from file if provided, otherwise use custom_prompt or default
        if prompt_file:
            self.prompt_template = self._load_prompt_from_file(prompt_file)
        else:
            self.prompt_template = custom_prompt or self.DEFAULT_PROMPT
        
        # Initialize provider-specific clients
        if self.provider == "openai":
            if not OPENAI_AVAILABLE:
                raise ImportError(
                    "OpenAI library is required. Install with: pip install openai"
                )
            
            self.api_key = api_key or os.getenv("OPENAI_API_KEY")
            if not self.api_key:
                raise ValueError(
                    "OpenAI API key is required. Provide via api_key parameter or "
                    "set OPENAI_API_KEY environment variable."
                )
            
            self.client = openai.OpenAI(api_key=self.api_key)
            
        elif self.provider == "anthropic":
            try:
                import anthropic
                self.anthropic_client = anthropic.Anthropic(api_key=api_key or os.getenv("ANTHROPIC_API_KEY"))
                self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
                if not self.api_key:
                    raise ValueError(
                        "Anthropic API key is required. Provide via api_key parameter or "
                        "set ANTHROPIC_API_KEY environment variable."
                    )
            except ImportError:
                raise ImportError(
                    "Anthropic library is required. Install with: pip install anthropic"
                )
                
        elif self.provider == "custom":
            if not REQUESTS_AVAILABLE:
                raise ImportError(
                    "Requests library is required for custom API. Install with: pip install requests"
                )
            if not api_base_url:
                raise ValueError(
                    "api_base_url is required when provider='custom'"
                )
            self.api_key = api_key
            if not self.api_key:
                raise ValueError(
                    "API key is required for custom provider. Provide via api_key parameter."
                )
        else:
            raise ValueError(
                f"Unsupported provider: {provider}. Supported providers: 'openai', 'anthropic', 'custom'"
            )
    
    def _load_prompt_from_file(self, prompt_file: str) -> str:
        """
        Load prompt template from a text file.
        
        Args:
            prompt_file: Path to the prompt text file
            
        Returns:
            Prompt template as string
        """
        prompt_path = Path(prompt_file)
        if not prompt_path.exists():
            raise FileNotFoundError(f"Prompt file not found: {prompt_file}")
        
        try:
            with open(prompt_path, 'r', encoding='utf-8') as f:
                prompt = f.read()
            logger.info(f"Loaded prompt from file: {prompt_file}")
            return prompt
        except Exception as e:
            logger.error(f"Error reading prompt file {prompt_file}: {e}")
            raise
    
    def extract_process_flow(self, document_content: str, document_name: str = "") -> Dict[str, Any]:
        """
        Extract process flow structure from document content.
        
        Args:
            document_content: Text content of the SOP document
            document_name: Name of the document (for context)
            
        Returns:
            Dictionary containing extracted process flow structure
        """
        # Truncate content if too long (most models have token limits)
        max_content_length = 100000  # Approximate character limit
        if len(document_content) > max_content_length:
            logger.warning(
                f"Document content too long ({len(document_content)} chars), "
                f"truncating to {max_content_length} chars"
            )
            document_content = document_content[:max_content_length]
        
        # Construct the prompt
        prompt = f"{self.prompt_template}\n\n--- Document: {document_name} ---\n\n{document_content}"
        
        try:
            # Call appropriate API based on provider
            if self.provider == "openai":
                response_text = self._call_openai_api(prompt)
            elif self.provider == "anthropic":
                response_text = self._call_anthropic_api(prompt)
            elif self.provider == "custom":
                response_text = self._call_custom_api(prompt)
            else:
                raise ValueError(f"Unsupported provider: {self.provider}")
            
            # Extract JSON from response
            process_flow = json.loads(response_text)
            
            # Add metadata
            process_flow['source_document'] = document_name
            process_flow['extraction_model'] = self.model
            
            logger.info(f"Successfully extracted process flow from {document_name}")
            return process_flow
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            logger.error(f"Response was: {response_text[:500]}")
            raise ValueError(f"Invalid JSON response from LLM: {e}")
        except Exception as e:
            logger.error(f"Error extracting process flow: {e}")
            raise
    
    def _call_openai_api(self, prompt: str) -> str:
        """Call OpenAI API and return response text."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are an expert at extracting structured process flows from SOP documents."},
                {"role": "user", "content": prompt}
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            response_format={"type": "json_object"}
        )
        return response.choices[0].message.content
    
    def _call_anthropic_api(self, prompt: str) -> str:
        """Call Anthropic API and return response text."""
        system_message = "You are an expert at extracting structured process flows from SOP documents."
        
        response = self.anthropic_client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            system=system_message,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        # Anthropic returns content as a list of text blocks
        response_text = ""
        for block in response.content:
            if hasattr(block, 'text'):
                response_text += block.text
            elif isinstance(block, str):
                response_text += block
        
        return response_text
    
    def _call_custom_api(self, prompt: str) -> str:
        """
        Call a custom LLM API via HTTP request.
        
        Expects the API to accept a POST request with JSON body containing:
        - model: model name
        - messages: list of message dicts with "role" and "content"
        - temperature: float
        - max_tokens: int
        
        And return JSON with response text in a standard format.
        """
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are an expert at extracting structured process flows from SOP documents."},
                {"role": "user", "content": prompt}
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }
        
        # Add JSON mode if supported (OpenAI-compatible format)
        payload["response_format"] = {"type": "json_object"}
        
        try:
            response = requests.post(
                self.api_base_url,
                headers=headers,
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            
            response_data = response.json()
            
            # Try to extract response text from common response formats
            if "choices" in response_data and len(response_data["choices"]) > 0:
                # OpenAI-compatible format
                return response_data["choices"][0]["message"]["content"]
            elif "content" in response_data:
                # Direct content field
                if isinstance(response_data["content"], list):
                    return "".join([item.get("text", "") for item in response_data["content"]])
                return response_data["content"]
            elif "text" in response_data:
                # Simple text field
                return response_data["text"]
            elif "message" in response_data:
                # Message field
                if isinstance(response_data["message"], dict) and "content" in response_data["message"]:
                    return response_data["message"]["content"]
                return str(response_data["message"])
            else:
                # Fallback: return the whole response as string
                logger.warning(f"Unexpected API response format: {list(response_data.keys())}")
                return json.dumps(response_data)
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Error calling custom API: {e}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Response status: {e.response.status_code}")
                logger.error(f"Response body: {e.response.text[:500]}")
            raise
    
    def extract_from_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract process flows from multiple documents.
        
        Args:
            documents: List of document dictionaries from DocumentReader
            
        Returns:
            List of extracted process flow dictionaries
        """
        extracted_flows = []
        
        for doc in documents:
            try:
                process_flow = self.extract_process_flow(
                    doc['content'],
                    doc['name']
                )
                # Add document metadata
                process_flow['document_path'] = doc['path']
                process_flow['document_relative_path'] = doc.get('relative_path', '')
                extracted_flows.append(process_flow)
            except Exception as e:
                logger.error(f"Failed to extract process flow from {doc['name']}: {e}")
                continue
        
        return extracted_flows

