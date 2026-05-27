# Run: COST_API_KEY=cost_sk_... ANTHROPIC_API_KEY=sk-ant-... python examples/anthropic_example.py
from anthropic import Anthropic
from botzone_cost import wrap

anthropic = wrap(
    Anthropic(),
    route="demo:summarise",
    feature_tag="examples",
)

reply = anthropic.messages.create(
    model="claude-haiku-4-5",
    max_tokens=256,
    messages=[
        {"role": "user", "content": "Summarise the plot of Hamlet in two sentences."},
    ],
)

print(reply.content[0].text)
