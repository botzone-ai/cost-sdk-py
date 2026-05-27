# botzone-cost

[![PyPI version](https://img.shields.io/pypi/v/botzone-cost.svg)](https://pypi.org/project/botzone-cost/)
[![Python versions](https://img.shields.io/pypi/pyversions/botzone-cost.svg)](https://pypi.org/project/botzone-cost/)
[![license](https://img.shields.io/pypi/l/botzone-cost.svg)](https://github.com/botzone-ai/cost-sdk-py/blob/main/LICENSE)

Cost-tracking SDK for Anthropic, OpenAI, and Gemini Python clients. Wrap your
existing client; per-call usage flows to your Cost dashboard. Adds zero
measurable latency to the host call.

## Install

```
pip install botzone-cost
```

## Usage

```python
from anthropic import Anthropic
from botzone_cost import wrap

client = wrap(Anthropic(), api_key="cost_sk_...", route="follow-up-draft")
```

Same surface for OpenAI and Gemini:

```python
from openai import OpenAI
import google.generativeai as genai
from botzone_cost import wrap

openai_client = wrap(OpenAI(), route="summariser")
gemini = wrap(genai.GenerativeModel("gemini-2.5-flash"), route="classifier")
```

## Options

| arg              | default                                          |
| ---------------- | ------------------------------------------------ |
| `api_key`        | env `COST_API_KEY`                               |
| `endpoint`       | env `COST_ENDPOINT` or `https://cost.botzone.ai` |
| `route`          | (none: strongly recommended)                     |
| `user_id`        | (sha256-hashed in the SDK before send)           |
| `feature_tag`    | (none)                                           |
| `enabled`        | `True`                                           |
| `capture_bodies` | `False` (reserved, no effect today, see below)   |

## What gets captured

Token counts (including Anthropic prompt-cache reads / writes and OpenAI cached
prompt tokens), latency, model, route, user id (hashed), feature tag. Computed
USD cost is added server-side from the live pricing table. The Python SDK is
**metadata-only today**: it does not send raw request or response bodies, and
the `capture_bodies` parameter is reserved for future parity with the
TypeScript SDK.

End-user identifiers passed via `user_id` are SHA-256 hashed in the SDK before
send; the plaintext never leaves your process.
