# Run: COST_API_KEY=cost_sk_... OPENAI_API_KEY=sk-... python examples/openai_example.py
from openai import OpenAI
from botzone_cost import wrap

openai = wrap(
    OpenAI(),
    route="demo:classify",
    feature_tag="examples",
)

reply = openai.chat.completions.create(
    model="gpt-4.1-mini",
    max_tokens=64,
    messages=[
        {"role": "system", "content": "Classify the user input as positive, negative, or neutral. Reply with one word."},
        {"role": "user", "content": "I love how fast this dashboard renders."},
    ],
)

print(reply.choices[0].message.content)
