---
name: Feature request
about: Propose a change to the SDK
labels: enhancement
---

## Problem

<!-- What can't you do today? Why does it matter? -->

## Proposed change

<!-- What would the API look like? Code example preferred. -->

```python
# before
client = wrap(Anthropic(), api_key=key, route="x")

# after
client = wrap(Anthropic(), api_key=key, route="x", new_option=True)
```

## Alternatives considered

<!-- Other approaches you thought about and why this one is better. -->

## Out of scope

<!-- Things you do NOT want this proposal to cover. -->
