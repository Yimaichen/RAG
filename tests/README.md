# Tests

This directory is for your custom test scripts. Test files are **gitignored** except for this README and the example below.

## Running Tests

```bash
# Run all tests
uv run pytest tests/

# Run a specific test
uv run pytest tests/test_import.py -v
```

## Writing Tests

Place your test files here following the `test_*.py` naming convention. Example:

```python
def test_my_feature():
    from arag import BaseAgent
    assert BaseAgent is not None
```
