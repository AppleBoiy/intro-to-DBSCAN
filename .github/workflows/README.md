# CI/CD Pipeline Documentation

This directory contains GitHub Actions workflows for automated testing and continuous integration.

## Workflows

### `tests.yml` - Main Test Pipeline

The main test pipeline runs on every push and pull request to ensure code quality and correctness.

#### Triggers

1. **Push/Pull Request** (default): Runs fast tests only
   - Triggered on pushes to `main` and `develop` branches
   - Triggered on pull requests to `main` and `develop` branches
   - Excludes tests marked with `@pytest.mark.slow`

2. **Scheduled** (weekly): Runs all tests including slow ones
   - Runs every Sunday at 00:00 UTC
   - Includes performance tests and property-based tests with high iteration counts

3. **Manual Trigger**: Run on-demand with optional slow tests
   - Can be triggered manually from GitHub Actions UI
   - Option to include slow tests via checkbox

#### Test Matrix

Tests run on multiple Python versions to ensure compatibility:
- Python 3.8
- Python 3.9
- Python 3.10
- Python 3.11

#### Jobs

##### 1. Test Job

Runs the test suite with coverage reporting.

**Steps:**
1. Checkout code
2. Set up Python with pip caching
3. Cache pip packages for faster builds
4. Install dependencies from `requirements.txt`
5. Install test dependencies (`pytest-cov`, `pytest-xdist`)
6. Run tests with coverage (fast or all, depending on trigger)
7. Upload coverage to Codecov (Python 3.11 only)
8. Upload HTML coverage report as artifact
9. Check coverage threshold (90% minimum)

**Test Commands:**

Fast tests (default):
```bash
pytest -v -m "not slow" --cov=src --cov-report=xml --cov-report=term-missing --cov-report=html -n auto
```

All tests (scheduled/manual):
```bash
pytest -v --cov=src --cov-report=xml --cov-report=term-missing --cov-report=html -n auto
```

**Flags explained:**
- `-v`: Verbose output
- `-m "not slow"`: Exclude slow tests
- `--cov=src`: Measure coverage for src/ directory
- `--cov-report=xml`: Generate XML report for Codecov
- `--cov-report=term-missing`: Show missing lines in terminal
- `--cov-report=html`: Generate HTML report
- `-n auto`: Run tests in parallel using all available CPUs

##### 2. Lint Job

Checks code quality and formatting.

**Steps:**
1. Checkout code
2. Set up Python 3.11
3. Install linting tools (`flake8`, `black`, `isort`)
4. Run flake8 for syntax errors and code quality
5. Check code formatting with black
6. Check import sorting with isort

**Note:** Linting failures are non-blocking (continue-on-error: true) to allow tests to run even if formatting needs improvement.

##### 3. Notebook Test Job

Validates that Jupyter notebooks execute without errors.

**Steps:**
1. Checkout code
2. Set up Python 3.11
3. Install dependencies
4. Execute all notebooks in `notebooks/` directory

**Note:** Notebook failures are non-blocking to allow for notebooks that may require specific data or long execution times.

## Coverage Reporting

### Codecov Integration

Coverage reports are automatically uploaded to Codecov for Python 3.11 runs.

#### Setup Instructions

1. **Sign up for Codecov:**
   - Go to [codecov.io](https://codecov.io)
   - Sign in with your GitHub account
   - Add your repository

2. **Get your Codecov token:**
   - Navigate to your repository settings on Codecov
   - Copy the repository upload token

3. **Add token to GitHub Secrets:**
   - Go to your GitHub repository
   - Navigate to Settings → Secrets and variables → Actions
   - Click "New repository secret"
   - Name: `CODECOV_TOKEN`
   - Value: Paste your Codecov token
   - Click "Add secret"

4. **View coverage reports:**
   - Coverage reports will appear on Codecov after each push
   - Pull requests will show coverage changes
   - Badge can be added to README.md

#### Codecov Badge

Add this badge to your README.md:

```markdown
[![codecov](https://codecov.io/gh/YOUR_USERNAME/YOUR_REPO/branch/main/graph/badge.svg)](https://codecov.io/gh/YOUR_USERNAME/YOUR_REPO)
```

Replace `YOUR_USERNAME` and `YOUR_REPO` with your GitHub username and repository name.

### Local Coverage Reports

HTML coverage reports are uploaded as artifacts and can be downloaded from the GitHub Actions run page.

To generate coverage reports locally:

```bash
# Run tests with coverage
pytest --cov=src --cov-report=html

# Open the report
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
start htmlcov/index.html  # Windows
```

## Caching Strategy

The pipeline uses multiple caching strategies to speed up builds:

1. **Python setup cache**: Caches Python installation
   ```yaml
   uses: actions/setup-python@v4
   with:
     cache: 'pip'
   ```

2. **Pip package cache**: Caches downloaded packages
   ```yaml
   uses: actions/cache@v3
   with:
     path: ~/.cache/pip
     key: ${{ runner.os }}-pip-${{ hashFiles('requirements.txt') }}
   ```

This reduces build time from ~2-3 minutes to ~30-60 seconds for cached builds.

## Test Markers

Tests are organized using pytest markers:

- `@pytest.mark.slow`: Long-running tests (performance, extensive property tests)
- `@pytest.mark.property`: Property-based tests
- `@pytest.mark.visualization`: Visualization tests
- `@pytest.mark.integration`: Integration tests

### Running Specific Test Categories Locally

```bash
# Run only fast tests
pytest -m "not slow"

# Run only slow tests
pytest -m "slow"

# Run only property tests
pytest -m "property"

# Run only visualization tests
pytest -m "visualization"

# Run multiple categories
pytest -m "property or visualization"
```

## Coverage Threshold

The pipeline enforces a minimum coverage threshold of **90%**.

If coverage falls below 90%, the build will fail with:
```bash
coverage report --fail-under=90
```

To check coverage locally:
```bash
pytest --cov=src --cov-report=term-missing
coverage report --fail-under=90
```

## Parallel Test Execution

Tests run in parallel using `pytest-xdist` with the `-n auto` flag, which automatically detects the number of available CPUs.

To run tests in parallel locally:
```bash
# Auto-detect CPUs
pytest -n auto

# Specify number of workers
pytest -n 4
```

## Troubleshooting

### Build Failures

1. **Coverage below 90%:**
   - Add tests for uncovered code
   - Check coverage report: `coverage report --show-missing`

2. **Test failures:**
   - Run tests locally: `pytest -v`
   - Check specific test: `pytest tests/test_file.py::test_name -v`

3. **Linting failures:**
   - Format code: `black src tests`
   - Sort imports: `isort src tests`
   - Check issues: `flake8 src tests`

4. **Notebook execution failures:**
   - Test locally: `jupyter nbconvert --to notebook --execute notebook.ipynb`
   - Check for missing data files or dependencies

### Slow Builds

1. **Clear cache:**
   - Go to Actions → Caches
   - Delete old caches

2. **Check dependency changes:**
   - Large dependency updates may require full reinstall
   - Consider pinning versions in requirements.txt

## Manual Workflow Dispatch

To manually trigger the workflow with slow tests:

1. Go to Actions tab in GitHub
2. Select "Tests" workflow
3. Click "Run workflow"
4. Check "Run slow tests" if desired
5. Click "Run workflow"

## Best Practices

1. **Before pushing:**
   ```bash
   # Run fast tests
   pytest -m "not slow"
   
   # Check coverage
   pytest --cov=src --cov-report=term-missing
   
   # Format code
   black src tests
   isort src tests
   ```

2. **Before merging:**
   - Ensure all CI checks pass
   - Review coverage report
   - Check for new warnings

3. **Weekly:**
   - Review scheduled test results
   - Address any slow test failures
   - Update dependencies if needed

## Future Enhancements

Potential improvements to the CI/CD pipeline:

1. **Security scanning:**
   - Add `bandit` for security vulnerability scanning
   - Add `safety` for dependency vulnerability checking

2. **Documentation:**
   - Auto-generate API documentation with Sphinx
   - Deploy documentation to GitHub Pages

3. **Release automation:**
   - Automatic version bumping
   - PyPI package publishing
   - GitHub release creation

4. **Performance tracking:**
   - Track test execution time over commits
   - Alert on performance regressions

5. **Matrix expansion:**
   - Test on multiple operating systems (Windows, macOS)
   - Test with different dependency versions

## References

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [pytest Documentation](https://docs.pytest.org/)
- [pytest-cov Documentation](https://pytest-cov.readthedocs.io/)
- [Codecov Documentation](https://docs.codecov.com/)
- [pytest-xdist Documentation](https://pytest-xdist.readthedocs.io/)

**Validates: Requirements 12.1, 12.8**
