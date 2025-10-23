from abc import ABC, abstractmethod
from typing import Dict, Any
import os
from mistralai import Mistral
from openai import OpenAI, AzureOpenAI
from anthropic import AnthropicBedrock
import time
import json
#import openai
# Load environment variables

class BaseLLM(ABC):
    """Abstract base class for LLM implementations"""
    
    @abstractmethod
    def __init__(self):
        self.client = None
        self.model = None

    def _process_response(self, response_text: str) -> Dict[str, str]:
        """Process raw LLM response text and handle JSON formatting"""
        # Handle markdown code block format
        if response_text.startswith('```json'):
            json_content = response_text.split('```json\n', 1)[1].rsplit('```', 1)[0]
            try:
                return json.loads(json_content)
            except json.JSONDecodeError:
                pass
        
        # Try to parse as JSON if it's a JSON response
        if response_text.startswith('{'):
            try:
                return json.loads(response_text)
            except json.JSONDecodeError:
                pass
        
        # Return simple text response
        return {"response": response_text}
    
    @abstractmethod
    def  query(self, prompt: str) -> Dict[str, str]:
        pass

class MistralLLM(BaseLLM):
    AVAILABLE_MODELS = [
        "mistral-large-latest",
        "mistral-small-latest",
        "mistral-medium"
    ]
    
    def __init__(self, model_name: str = "mistral-large-latest"):
        self.api_key = os.getenv("MISTRAL_API_KEY")
        if not self.api_key:
            raise ValueError("MISTRAL_API_KEY not found in environment variables")
        self.client = Mistral(api_key=self.api_key)
        if model_name not in self.AVAILABLE_MODELS:
            raise ValueError(f"Model must be one of: {self.AVAILABLE_MODELS}")
        self.model = model_name

    def query(self, prompt: str) -> Dict[str, str]:
        while True:
            try:
                completion = self.client.chat.complete(
                    model=self.model,
                    messages=[
                        {"role": "user", "content": prompt},
                    ]
                )

                return completion.choices[0].message.content.strip()
                
            except Exception as e:
                if hasattr(e, 'status_code') and e.status_code == 429:
                    time.sleep(1)  # Add a small delay for rate limiting
                    continue
                else:
                    raise

class OpenAILLM(BaseLLM):
    AVAILABLE_MODELS = [
        "gpt-4",
        "gpt-3.5-turbo",
        "gpt-4-turbo-preview"
    ]

    def __init__(self, model_name: str = "gpt-4"):
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        self.client = OpenAI(
            api_key=self.api_key,
        )
        if model_name not in self.AVAILABLE_MODELS:
            raise ValueError(f"Model must be one of: {self.AVAILABLE_MODELS}")
        self.model = model_name

    def query(self, prompt: str) -> Dict[str, str]:
        while True:
            try:
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )

                return completion.choices[0].message.content.strip()
                
            except Exception as e:
                if hasattr(e, 'status_code') and e.status_code == 429:
                    time.sleep(1)  # Add delay for rate limiting
                    continue
                else:
                    raise

class AnthropicLLM(BaseLLM):

    def __init__(self, model_name: str = "claude-3.7"):

        self.model = model_name # currently only supports "claude-3.7"
        self.model_id = "TODO...SET MODEL ID"
        self.region_name = "us-east-2"
        self.client = AnthropicBedrock(
            aws_access_key=os.environ["AWS_ACESS_KEY"],
            aws_secret_key=os.environ["AWS_SECRET_KEY"],
            aws_region="us-east-2",
        )



    def query(self, prompt: str) -> Dict[str, str]:
        while True:
            try:
                completion = self.client.messages.create(
                    model = self.model_id,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=4096
                )

                return completion.content[0].text.strip()
                
            except Exception as e:
                if hasattr(e, 'status_code') and e.status_code == 429:
                    time.sleep(1)  # Add delay for rate limiting
                    continue
                else:
                    raise

class AzureOpenAILLM(BaseLLM):
    AVAILABLE_MODELS = [
        "gpt-4o",
        "gpt-4o-mini",
        "o1",
        "o1-mini"
    ]

    def __init__(self, model_name: str = "gpt-4o-mini"):
        
        endpoint = "TODO: SET AZURE ENDPOINT"
        api_version = "2024-12-01-preview"

        self.api_key = os.getenv("AZURE_OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("AZURE_OPENAI_API_KEY not found in environment variables")
        
        self.client = AzureOpenAI(
            api_version=api_version,
            azure_endpoint=endpoint,
            api_key=self.api_key,
        )
        # openai.api_type = "azure"
        # openai.api_base = endpoint
        # openai.api_version = api_version
        # openai.api_key = self.api_key

        if model_name not in self.AVAILABLE_MODELS:
            raise ValueError(f"Model must be one of: {self.AVAILABLE_MODELS}")
        
        self.model = model_name


    def query(self, prompt: str) -> Dict[str, str]:
        while True:
            try:
                completion = self.client.chat.completions.create(
                #completion = openai.ChatCompletion.create(
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    # max_tokens=4096,
                    temperature=1.0,
                    top_p=1.0,
                    model = self.model
                )

                return completion.choices[0].message.content.strip()
                
            except Exception as e:
                if hasattr(e, 'status_code') and e.status_code == 429:
                    time.sleep(1)  # Add delay for rate limiting
                    continue
                else:
                    raise


class OpenRouterLLM(BaseLLM):
    AVAILABLE_MODELS = [
        "deepseek/deepseek-r1:free",
        "deepseek/deepseek-r1-distill-llama-70b:free"
    ]
    
    def __init__(self, model_name: str = "deepseek/deepseek-r1:free"):
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY not found in environment variables")
        self.base_url = "https://openrouter.ai/api/v1"
        self.client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
        )
        if model_name not in self.AVAILABLE_MODELS:
            raise ValueError(f"Model must be one of: {self.AVAILABLE_MODELS}")
        self.model = model_name

    def query(self, prompt: str) -> Dict[str, str]:
        while True:
            try:
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )

                return completion.choices[0].message.content.strip()

            except Exception as e:
                if hasattr(e, 'status_code') and e.status_code == 429:
                    time.sleep(1)  # Add delay for rate limiting
                    continue
                else:
                    raise

                    
class DeepSeekLLM(BaseLLM):
    AVAILABLE_MODELS = [
        "deepseek-chat",
        "deepseek-reasoner"
    ]
    
    def __init__(self, model_name: str = "deepseek-chat"):
        self.api_key = os.getenv("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise ValueError("DEEPSEEK_API_KEY not found in environment variables")
        self.base_url = "https://api.deepseek.com"
        self.client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
        )
        if model_name not in self.AVAILABLE_MODELS:
            raise ValueError(f"Model must be one of: {self.AVAILABLE_MODELS}")
        self.model = model_name

    def query(self, prompt: str) -> Dict[str, str]:
        while True:
            try:
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )

                return completion.choices[0].message.content.strip()

            except Exception as e:
                if hasattr(e, 'status_code') and e.status_code == 429:
                    time.sleep(1)  # Add delay for rate limiting
                    continue
                else:
                    raise