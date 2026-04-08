import os
from google import genai
from google.genai import types
import PIL.Image

# 1. Setup Client
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

# 2. Load the X-ray
main_image = PIL.Image.open('/home/t-qimhuang/code/grounded_medical_reasoning/datasets/MIMIC-CXR-decoded/p10/p10000032/s50414267/02aa804e-bde0afdd-112c0b34-7bc16630-4e384014.png')

# 3. Agentic Prompt
# Note: We explicitly ask the model to "verify" or "inspect" to trigger tool use.
prompt = """
You are an expert Radiologist. Analyze the attached Chest X-ray. 
If you notice any subtle or suspicious areas, use code execution to 
crop and inspect them at high resolution before finalizing your report.

Provide:
- Findings: (Detailed observations)
- Impression: (Conclusion)
"""

# 4. Run Agentic Inference
response = client.models.generate_content(
    model='gemini-3-flash-preview',
    contents=[prompt, main_image],
    config=types.GenerateContentConfig(
        # CRITICAL: This enables Agentic Vision
        tools=[types.Tool(code_execution=types.ToolCodeExecution())],
        
        # RECOMMENDED: High level gives it the "patience" to zoom/crop
        thinking_config=types.ThinkingConfig(
            include_thoughts=True,
            thinking_level="HIGH" 
        ),
        temperature=0.0,
    )
)

# 5. Output Results
print("--- AGENTIC RADIOLOGY REPORT ---")
# This will now include logs of the Python code the model ran to zoom in!
for part in response.candidates[0].content.parts:
    if part.thought:
        print(f"\n[INTERNAL THOUGHTS]:\n{part.text}")
    if part.executable_code:
        print(f"\n[AGENTIC ACTION - PYTHON]:\n{part.executable_code.code}")
    if part.code_execution_result:
        print(f"\n[TOOL OUTPUT]:\n{part.code_execution_result.output}")

print(f"\n[FINAL REPORT]:\n{response.text}")
