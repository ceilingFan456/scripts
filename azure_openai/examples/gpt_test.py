from openai import AzureOpenAI
import os

# 1. TOKEN
# Never hardcode the token - it is a live credential. Load it from the env:
#   export TRAPI_ACCESS_TOKEN="$(az account get-access-token \
#       --resource api://trapi/.default --query accessToken -o tsv)"
MY_TOKEN = os.environ["TRAPI_ACCESS_TOKEN"]

# 2. CONFIGURATION (Must match your TRAPI settings)
apipath = "gcr/shared"
deployment_name = "gpt-4o_2024-11-20"
api_version = "2025-04-01-preview"
endpoint = f"https://trapi.research.microsoft.com/{apipath}"

# 3. INITIALIZE CLIENT
# We pass the Token into 'api_key'. The AzureOpenAI client handles 
# Entra ID tokens perfectly when passed here.
client = AzureOpenAI(
    azure_endpoint=endpoint,
    api_key=MY_TOKEN, 
    api_version=api_version,
)

# 4. EXECUTE
try:
    response = client.chat.completions.create(
        model=deployment_name,
        messages=[{"role": "user", "content": "What is the capital of France?"}]
    )
    print(response.choices[0].message.content)
except Exception as e:
    print(f"Error: {e}")
