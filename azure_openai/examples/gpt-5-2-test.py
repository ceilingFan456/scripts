import os
from openai import AzureOpenAI
 
endpoint = "https://e0271-miptdstj-eastus2.cognitiveservices.azure.com/"
#"https://e0271-miptdstj-eastus2.services.ai.azure.com/models/chat/completions?api-version=2024-05-01-preview%22
# "https://e0271-miptdstj-eastus2.cognitiveservices.azure.com/"
model_name = "gpt-5-chat"
 
# Never hardcode the key: export AZURE_OPENAI_API_KEY=... before running.
subscription_key = os.environ["AZURE_OPENAI_API_KEY"]
api_version = "2024-12-01-preview" # "2024-05-01-preview" # "2024-12-01-preview"
 
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
    max_completion_tokens=16384,
    model=model_name
)
 
print(response.choices[0].message.content)