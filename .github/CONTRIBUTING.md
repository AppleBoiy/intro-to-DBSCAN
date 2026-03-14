# Contributing to DBSCAN Learning Repository

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## Development Setup

### Prerequisites

- Python 3.8 or higher
- Git
- Virtual environment tool (venv, virtualenv, or conda)

### Initial Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/YOUR_USERNAME/intro-to-DBSCAN.git
   cd intro-to-DBSCAN
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   pip install pytest-cov pytest-xdist flake8 black isort
   ```

4. **Verify installation:**
   ```bash
   pytest -v -m "not slow"
   ```

## Development Workflow

### Before Making Changes

1. **Create a new branch:**
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/your-bug-fix
   ```

2. **Ensure tests pass:**
   ```bash
   pytest -m "not slow"
   ```

### Making Changes

1. **Write code following the style guide** (see below)

2. **Add tests for new functionality:**
   - Unit tests in appropriate `test_*.py` file
   - Property-based tests if applicable
   - Update fixtures in `tests/conftest.py` if needed

3. **Run tests frequently:**
   ```bash
   # Run specific test file
   pytest tests/test_your_file.py -v
   
   # Run specific test
   pytest tests/test_your_file.py::test_your_function -v
   
   # Run with coverage
   pytest --cov=src --cov-report=term-missing
   ```

4. **Update documentation:**
   - Add docstrings to new functions/classes
   - Update relevant markdown files in `docs/`
   - Add paper citations where applicable

### Before Committing

1. **Format your code:**
   ```bash
   black src tests
   isort src tests
   ```

2. **Check for linting issues:**
   ```bash
   flake8 src tests
   ```

3. **Run the full test suite:**
   ```bash
   pytest -m "not slow" --cov=src --cov-report=term-missing
   ```

4. **Verify coverage is above 90%:**
   ```bash
   coverage report --fail-under=90
   ```

5. **Commit your changes:**
   ```bash
   git add .
   git commit -m "Brief description of changes"
   ```

### Submitting a Pull Request

1. **Push your branch:**
   ```bash
   git push origin feature/your-feature-name
   ```

2. **Create a pull request on GitHub:**
   - Provide a clear title and description
   - Reference any related issues
   - Ensure all CI checks pass

3. **Respond to review feedback:**
   - Make requested changes
   - Push updates to the same branch
   - CI will automatically re-run

## Code Style Guide

### Python Code

We follow PEP 8 with some modifications:

- **Line length:** 100 characters (not 79)
- **Formatting:** Use `black` for automatic formatting
- **Import sorting:** Use `isort` for consistent import ordering
- **Type hints:** Required for all function signatures
- **Docstrings:** Required for all public functions and classes

### Docstring Format

Use NumPy-style docstrings:

```python
def example_function(param1: int, param2: str) -> bool:
    """
    Brief one-line description.
    
    Longer description if needed. Can span multiple lines.
    Include paper citations where applicable [Paper §X.Y].
    
    Parameters
    ----------
    param1 : int
        Description of param1
    param2 : str
        Description of param2
    
    Returns
    -------
    bool
        Description of return value
    
    Raises
    ------
    ValueError
        When param1 is negative
    
    Examples
    --------
    >>> example_function(5, "test")
    True
    
    Notes
    -----
    Additional notes about implementation or complexity.
    
    References
    ----------
    Ester, M., et al. (1996). DBSCAN. KDD-96.
    """
    pass
```

### Test Code

- **Test names:** Use descriptive names starting with `test_`
- **Markers:** Use appropriate pytest markers (`@pytest.mark.slow`, etc.)
- **Fixtures:** Use fixtures from `conftest.py` when possible
- **Assertions:** Use descriptive assertion messages

Example:
```python
@pytest.mark.property
def test_density_reachability_transitivity(sample_data):
    """
    Test that density-reachability is transitive [Paper Lemma 1].
    
    **Validates: Requirements 12.4**
    """
    dbscan = DBSCAN(eps=0.5, min_pts=5)
    labels = dbscan.fit_predict(sample_data)
    
    # Test logic here
    assert condition, "Descriptive message about what failed"
```

### Documentation

- **Markdown files:** Use proper heading hierarchy
- **Paper citations:** Use format `[Paper §X.Y, p.ZZZ]`
- **Code examples:** Use syntax highlighting
- **Cross-references:** Use relative links

## Testing Guidelines

### Test Categories

1. **Unit Tests:** Test individual functions/methods
2. **Property Tests:** Test mathematical properties with hypothesis
3. **Integration Tests:** Test component interactions
4. **Performance Tests:** Test scalability and complexity

### Writing Tests

1. **Use fixtures:** Leverage existing fixtures in `conftest.py`
2. **Test edge cases:** Empty inputs, single points, boundary values
3. **Test error handling:** Invalid inputs should raise appropriate errors
4. **Add markers:** Use `@pytest.mark.slow` for long-running tests

### Running Tests

```bash
# Fast tests only (default for CI)
pytest -m "not slow"

# All tests
pytest

# Specific category
pytest -m "property"
pytest -m "visualization"

# With coverage
pytest --cov=src --cov-report=html

# Parallel execution
pytest -n auto

# Verbose output
pytest -v

# Stop on first failure
pytest -x
```

## Continuous Integration

### CI Pipeline

The CI pipeline runs automatically on:
- Every push to `main` or `develop`
- Every pull request
- Weekly schedule (Sundays at 00:00 UTC)
- Manual trigger

### CI Jobs

1. **Test Job:** Runs tests on Python 3.8, 3.9, 3.10, 3.11
2. **Lint Job:** Checks code formatting and style
3. **Notebook Test Job:** Validates notebook execution

### Coverage Requirements

- Minimum coverage: **90%**
- Coverage is checked only on Python 3.11
- Coverage reports uploaded to Codecov

### Handling CI Failures

1. **Test failures:**
   - Run tests locally: `pytest -v`
   - Fix the failing tests
   - Ensure tests pass before pushing

2. **Coverage failures:**
   - Check coverage: `pytest --cov=src --cov-report=term-missing`
   - Add tests for uncovered code
   - Aim for >90% coverage

3. **Linting failures:**
   - Format code: `black src tests`
   - Sort imports: `isort src tests`
   - Fix flake8 issues: `flake8 src tests`

## Documentation Guidelines

### Adding Documentation

1. **Code documentation:**
   - Add docstrings to all public functions/classes
   - Include paper citations where applicable
   - Add complexity analysis comments

2. **Markdown documentation:**
   - Update relevant files in `docs/`
   - Follow existing structure and format
   - Add cross-references to related topics

3. **Notebook documentation:**
   - Add markdown cells explaining concepts
   - Include learning objectives and prerequisites
   - Add exercises with solutions

### Paper Citations

Always cite the original paper when discussing DBSCAN concepts:

```markdown
**Definition (ε-neighborhood)** [Paper §4.1, p.227]:
The ε-neighborhood of a point p is defined as...
```

## Commit Message Guidelines

Use clear, descriptive commit messages:

```
Short summary (50 chars or less)

More detailed explanation if needed. Wrap at 72 characters.
Explain what changed and why, not how.

- Bullet points are okay
- Use present tense: "Add feature" not "Added feature"
- Reference issues: "Fixes #123" or "Relates to #456"
```

### Commit Message Examples

Good:
```
Add k-distance graph visualization

Implement plot_k_distance_graph() method in DBSCANVisualizer
to help users select optimal epsilon parameter. Includes
automatic elbow detection.

Validates: Requirements 3.8, 4.4
```

Bad:
```
Fixed stuff
```

## Review Process

### For Contributors

1. **Self-review:** Review your own changes before submitting
2. **Tests:** Ensure all tests pass
3. **Documentation:** Update relevant documentation
4. **Respond promptly:** Address review feedback quickly

### For Reviewers

1. **Be constructive:** Provide helpful, actionable feedback
2. **Check tests:** Verify adequate test coverage
3. **Check documentation:** Ensure changes are documented
4. **Check style:** Verify code follows style guide

## Getting Help

- **Questions:** Open a GitHub issue with the "question" label
- **Bugs:** Open a GitHub issue with the "bug" label
- **Feature requests:** Open a GitHub issue with the "enhancement" label
- **Discussions:** Use GitHub Discussions for general topics

## License

By contributing, you agree that your contributions will be licensed under the same license as the project.

## Recognition

Contributors will be recognized in the project README and release notes.

Thank you for contributing to the DBSCAN Learning Repository!
