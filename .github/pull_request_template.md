## What this changes

<!-- One or two sentences. Reference the issue if there is one (e.g. "Closes #12"). -->

## Why

<!-- The motivation. What problem does this solve? -->

## How to test

<!-- Steps a reviewer can run locally. -->

```bash
pip install -e '.[test]'
pytest -v
```

## Checklist

- [ ] `pytest` passes locally
- [ ] CHANGELOG.md updated if user-facing
- [ ] README updated if API surface changed
- [ ] No new runtime dependencies (or justified in PR body)
- [ ] Python 3.10+ compatibility preserved
