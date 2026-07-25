"""
Simple example for Azure OpenAI Chat Completions API using TRAPI .

This example demonstrates how to call the chat completions API using the TRAPI
endpoint with Azure AD authentication.

Documentation:
    https://learn.microsoft.com/en-us/azure/ai-foundry/openai/how-to/chatgpt

Usage:
    python chat_completions.py

Environment Variables:
    TRAPI_APIPATH - TRAPI API path (default: gcr/shared)
    TRAPI_MODEL - Model deployment name (default: gpt-4o_2024-11-20)
    TRAPI_API_VERSION - API version (default: 2025-04-01-preview)
"""

from openai import AzureOpenAI
from azure.identity import ChainedTokenCredential, AzureCliCredential, ManagedIdentityCredential, get_bearer_token_provider
import os

# Configuration
scope = os.environ.get("TRAPI_SCOPE", "api://trapi/.default")
apipath = os.environ.get("TRAPI_APIPATH", "gcr/shared")
deployment_name = os.environ.get("TRAPI_MODEL", "gpt-5.4_2026-03-05")
api_version = os.environ.get("TRAPI_API_VERSION", "2025-04-01-preview")
endpoint = os.environ.get("TRAPI_ENDPOINT", f"https://trapi.research.microsoft.com/{apipath}")

# Authentication
credential = get_bearer_token_provider(ChainedTokenCredential(
    AzureCliCredential(),
 ManagedIdentityCredential()
), scope)

print(f"Using credential: {credential}")

# 1. Access the cells in the closure
if credential.__closure__:
    for i, cell in enumerate(credential.__closure__):
        content = cell.cell_contents
        print(f"Cell {i} type: {type(content)}")
        print(f"Cell {i} value: {content}")
        print("-" * 20)
else:
    print("No closure variables found.")

# Client Initialization
client = AzureOpenAI(
    azure_endpoint=endpoint,
    azure_ad_token_provider=credential,
    api_version=api_version,
)


# 1. Manually get the token from the credential object
# Note: You need to provide the 'scope' you used earlier
token = credential()

# 2. Print the token string
print(f"Access Token: {token}")

response = client.chat.completions.create(
    model=deployment_name,
    messages=[
        {
            "role": "user",
            "content": "Give a one word answer, what is the capital of France?",
        },
    ]
)
response_content = response.choices[0].message.content
print(response_content)