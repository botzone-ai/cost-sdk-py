# Run: COST_API_KEY=cost_sk_... GEMINI_API_KEY=AI... python examples/gemini_example.py
import os
import google.generativeai as genai
from botzone_cost import wrap

genai.configure(api_key=os.environ["GEMINI_API_KEY"])

model = wrap(
    genai.GenerativeModel("gemini-2.5-flash"),
    route="demo:translate",
    feature_tag="examples",
)

result = model.generate_content(
    "Translate to French, then back to English: 'The early bird catches the worm.'",
)

print(result.text)
