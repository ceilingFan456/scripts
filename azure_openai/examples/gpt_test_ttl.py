import os
import sys
import datetime

import jwt  # pip install PyJWT

# Never hardcode a token here - it is a live credential.
# Pass it in instead:
#   export AZURE_ACCESS_TOKEN="$(az account get-access-token --query accessToken -o tsv)"
#   python gpt_test_ttl.py
# or:
#   python gpt_test_ttl.py <token>
token_string = os.getenv("AZURE_ACCESS_TOKEN") or (sys.argv[1] if len(sys.argv) > 1 else "")

if not token_string:
    sys.exit("No token provided. Set AZURE_ACCESS_TOKEN or pass the token as argv[1].")

# Decode without verification (since we just want to read the metadata)
decoded = jwt.decode(token_string, options={"verify_signature": False})

# Get expiration and issued-at times (Epoch timestamps)
exp_timestamp = decoded["exp"]
iat_timestamp = decoded["iat"]

# Convert to readable format
expiry_date = datetime.datetime.fromtimestamp(exp_timestamp)
issued_date = datetime.datetime.fromtimestamp(iat_timestamp)
remaining_time = expiry_date - datetime.datetime.now()

print(f"Token Issued At: {issued_date}")
print(f"Token Expires At: {expiry_date}")
print(f"Time Remaining: {remaining_time}")
