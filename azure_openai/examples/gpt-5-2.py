import os
from openai import AzureOpenAI

endpoint = "https://e0271-miptdstj-eastus2.cognitiveservices.azure.com/"
model_name = "grok-4-1-fast-reasoning"
deployment_name = "grok-4-1-fast-reasoning"

# Never hardcode the key: export AZURE_OPENAI_API_KEY=... before running.
subscription_key = os.environ["AZURE_OPENAI_API_KEY"]
api_version = "2024-12-01-preview"

client = AzureOpenAI(
    api_version=api_version,
    azure_endpoint=endpoint,
    api_key=subscription_key,
)

response = client.chat.completions.create(
    messages=[
        {
            "role": "system",
            "content": "You are a helpful assistant.",
        },
        {
            "role": "user",
            "content": "I am going to Paris, what should I see?",
        }
    ],
    max_completion_tokens =12_000,
    model=deployment_name
)

print(response.choices[0].message.content)