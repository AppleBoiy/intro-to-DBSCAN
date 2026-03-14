# CI/CD Quick Reference

## Common Commands

### Run Tests Locally
```bash
# Fast tests (same as CI on push/PR)
pytest -v -m "not slow" --cov=src --cov-report=term-missing -n auto

# All tests (same as CI on schedule)
pytest -v --cov=src --cov-report=term-missing -n auto

# Check coverage threshold
coverage report --fail-under=90
```

### Format Code
```bash
black src tests
isort src tests
flake8 src tests
```

### Run Specific Tests
```bash
pytest tests/test_dbscan.py -v
pytest tests/test_dbscan.py::test_function_name -v
pytest -m "property" -v
pytest -m "slow" -v
```

## CI/CD Triggers

| Event | Tests Run | Duration |
|-------|-----------|----------|
| Push to main/develop | Fast (195 tests) | ~2-3 min |
| Pull request | Fast (195 tests) | ~2-3 min |
| Schedule (Sunday 00:00) | All (213 tests) | ~5-10 min |
| Manual trigger | Fast or All | Variable |

## Coverage Reports

- **Codecov**: codecov.io (if configured)
- **Artifact**: Actions → Run → Artifacts → coverage-report
- **Local**: `pytest --cov=src --cov-report=html && open htmlcov/index.html`

## Test Markers

- `@pytest.mark.slow` - Long-running tests
- `@pytest.mark.property` - Property-based tests
- `@pytest.mark.visualization` - Visualization tests
- `@pytest.mark.integration` - Integration tests

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Tests fail in CI but pass locally | Check Python version, dependencies |
| Coverage below 90% | Add tests, check `coverage report --show-missing` |
| Linting failures | Run `black src tests && isort src tests` |
| Slow builds | Clear cache in Actions → Caches |

## Files

- `.github/workflows/tests.yml` - Main workflow
- `.github/workflows/README.md` - Full documentation
- `.github/CI_SETUP.md` - Setup guide
- `.github/CONTRIBUTING.md` - Contribution guide
