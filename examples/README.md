# Examples

Paste-and-run snippets for each provider. Each script:
- imports the real provider SDK
- wraps it with `botzone-cost`
- makes one LLM call
- prints the response

## Prerequisites

- A Cost API key from https://cost.botzone.ai (`cost_sk_...`)
- A provider API key (Anthropic / OpenAI / Gemini, depending on which script)
- `pip install botzone-cost anthropic openai google-generativeai`

## Run

```bash
COST_API_KEY=cost_sk_... ANTHROPIC_API_KEY=sk-ant-... python examples/anthropic_example.py
COST_API_KEY=cost_sk_... OPENAI_API_KEY=sk-...        python examples/openai_example.py
COST_API_KEY=cost_sk_... GEMINI_API_KEY=AI...         python examples/gemini_example.py
```

Within a few seconds the call should appear on your Cost dashboard under the `examples` feature tag.
