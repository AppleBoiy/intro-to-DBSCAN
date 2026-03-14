# CI/CD Setup Guide

This guide walks you through setting up the CI/CD pipeline for the DBSCAN Learning Repository.

## Quick Start

The CI/CD pipeline is already configured and will work automatically once you push to GitHub. However, to get full functionality (coverage reporting), you need to set up Codecov.

## Prerequisites

- GitHub repository
- GitHub account with admin access to the repository
- (Optional) Codecov account for coverage reporting

## Automatic Setup (No Configuration Needed)

The following features work automatically without any setup:

✅ **Test execution** on push and pull requests  
✅ **Multi-version testing** (Python 3.8, 3.9, 3.10, 3.11)  
✅ **Parallel test execution** with pytest-xdist  
✅ **Code linting** with flake8, black, isort  
✅ **Notebook validation**  
✅ **Dependency caching** for faster builds  
✅ **HTML coverage reports** as artifacts  

## Optional Setup: Codecov Integration

Codecov provides beautiful coverage reports and PR comments. Setup takes ~5 minutes.

### Step 1: Sign Up for Codecov

1. Go to [codecov.io](https://codecov.io)
2. Click "Sign up with GitHub"
3. Authorize Codecov to access your GitHub account

### Step 2: Add Your Repository

1. After signing in, you'll see your repositories
2. Find your repository in the list
3. Click "Setup repo" or toggle it to enable

### Step 3: Get Your Upload Token

1. Click on your repository in Codecov
2. Go to "Settings" → "General"
3. Copy the "Repository Upload Token"

### Step 4: Add Token to GitHub Secrets

1. Go to your GitHub repository
2. Click "Settings" (repository settings, not your account)
3. In the left sidebar, click "Secrets and variables" → "Actions"
4. Click "New repository secret"
5. Enter the following:
   - **Name:** `CODECOV_TOKEN`
   - **Secret:** Paste the token from Step 3
6. Click "Add secret"

### Step 5: Verify Setup

1. Push a commit to your repository
2. Go to the "Actions" tab
3. Wait for the workflow to complete
4. Check Codecov dashboard for coverage report

### Step 6: Add Coverage Badge (Optional)

Add a coverage badge to your README.md:

```markdown
[![codecov](https://codecov.io/gh/YOUR_USERNAME/YOUR_REPO/branch/main/graph/badge.svg)](https://codecov.io/gh/YOUR_USERNAME/YOUR_REPO)
```

Replace:
- `YOUR_USERNAME` with your GitHub username
- `YOUR_REPO` with your repository name

## Verifying the Setup

### 1. Check Workflow Files

Ensure the workflow file exists:
```bash
ls -la .github/workflows/tests.yml
```

### 2. Test Locally

Run the same commands that CI runs:

```bash
# Install test dependencies
pip install pytest-cov pytest-xdist

# Run fast tests (what CI runs on push/PR)
pytest -v -m "not slow" --cov=src --cov-report=xml --cov-report=term-missing -n auto

# Check coverage threshold
coverage report --fail-under=90
```

### 3. Push and Monitor

1. Make a small change (e.g., update README)
2. Commit and push:
   ```bash
   git add .
   git commit -m "Test CI pipeline"
   git push
   ```
3. Go to GitHub → Actions tab
4. Watch the workflow run
5. Verify all jobs pass (green checkmarks)

## Understanding the Pipeline

### Triggers

The pipeline runs on:

1. **Push to main/develop:**
   ```yaml
   on:
     push:
       branches: [ main, develop ]
   ```

2. **Pull requests to main/develop:**
   ```yaml
   on:
     pull_request:
       branches: [ main, develop ]
   ```

3. **Weekly schedule (Sundays at 00:00 UTC):**
   ```yaml
   on:
     schedule:
       - cron: '0 0 * * 0'
   ```

4. **Manual trigger:**
   - Go to Actions → Tests → Run workflow

### Jobs

#### 1. Test Job (Matrix Strategy)

Runs tests on 4 Python versions in parallel:

```
Python 3.8  ✓
Python 3.9  ✓
Python 3.10 ✓
Python 3.11 ✓ (+ coverage upload)
```

**Fast tests (default):**
- Excludes `@pytest.mark.slow` tests
- Runs in ~2-3 minutes

**All tests (scheduled/manual):**
- Includes all tests
- Runs in ~5-10 minutes

#### 2. Lint Job

Checks code quality:
- flake8: Syntax errors and code quality
- black: Code formatting
- isort: Import sorting

**Note:** Linting failures don't block the build (continue-on-error: true)

#### 3. Notebook Test Job

Validates notebooks:
- Executes all notebooks in `notebooks/`
- Ensures notebooks run without errors

**Note:** Notebook failures don't block the build

### Caching

The pipeline uses caching to speed up builds:

1. **Python setup cache:**
   - Caches Python installation
   - Saves ~30 seconds per run

2. **Pip package cache:**
   - Caches downloaded packages
   - Saves ~1-2 minutes per run

**Cache key:** Based on `requirements.txt` hash  
**Cache invalidation:** Automatic when requirements.txt changes

### Artifacts

The pipeline uploads artifacts:

1. **Coverage HTML report:**
   - Available for 30 days
   - Download from Actions run page
   - View detailed coverage in browser

## Customization

### Changing Python Versions

Edit `.github/workflows/tests.yml`:

```yaml
matrix:
  python-version: ['3.8', '3.9', '3.10', '3.11', '3.12']  # Add 3.12
```

### Changing Coverage Threshold

Edit `.github/workflows/tests.yml`:

```yaml
- name: Check coverage threshold
  run: |
    coverage report --fail-under=85  # Change from 90 to 85
```

### Adding More Branches

Edit `.github/workflows/tests.yml`:

```yaml
on:
  push:
    branches: [ main, develop, staging ]  # Add staging
  pull_request:
    branches: [ main, develop, staging ]
```

### Changing Schedule

Edit `.github/workflows/tests.yml`:

```yaml
schedule:
  - cron: '0 0 * * 1'  # Run on Mondays instead of Sundays
```

Cron syntax:
```
┌───────────── minute (0 - 59)
│ ┌───────────── hour (0 - 23)
│ │ ┌───────────── day of month (1 - 31)
│ │ │ ┌───────────── month (1 - 12)
│ │ │ │ ┌───────────── day of week (0 - 6) (Sunday to Saturday)
│ │ │ │ │
* * * * *
```

## Troubleshooting

### Issue: Workflow Not Running

**Symptoms:** No workflow runs appear in Actions tab

**Solutions:**
1. Check workflow file exists: `.github/workflows/tests.yml`
2. Check YAML syntax: Use a YAML validator
3. Check branch name matches trigger configuration
4. Ensure Actions are enabled: Settings → Actions → Allow all actions

### Issue: Coverage Upload Fails

**Symptoms:** "Codecov upload failed" error

**Solutions:**
1. Verify `CODECOV_TOKEN` secret is set correctly
2. Check token hasn't expired
3. Verify repository is enabled in Codecov
4. Check Codecov service status

**Note:** Coverage upload failure doesn't fail the build (`fail_ci_if_error: false`)

### Issue: Tests Pass Locally But Fail in CI

**Possible causes:**
1. **Missing dependencies:** Add to `requirements.txt`
2. **Environment differences:** Check Python version
3. **File paths:** Use relative paths, not absolute
4. **Random seeds:** Ensure tests use fixed random seeds
5. **Timing issues:** Avoid time-dependent tests

**Debug steps:**
```bash
# Run tests in same environment as CI
python -m venv ci_test_env
source ci_test_env/bin/activate
pip install -r requirements.txt
pip install pytest-cov pytest-xdist
pytest -v -m "not slow" -n auto
```

### Issue: Coverage Below 90%

**Solutions:**
1. Check coverage report:
   ```bash
   pytest --cov=src --cov-report=term-missing
   ```
2. Identify uncovered lines
3. Add tests for uncovered code
4. Or adjust threshold if 90% is too strict

### Issue: Slow Build Times

**Causes and solutions:**

1. **Cache not working:**
   - Clear cache: Actions → Caches → Delete
   - Check cache key in workflow file

2. **Too many tests:**
   - Mark slow tests: `@pytest.mark.slow`
   - Run slow tests only on schedule

3. **Large dependencies:**
   - Consider using lighter alternatives
   - Pin versions to avoid re-downloading

### Issue: Linting Failures

**Solutions:**
```bash
# Format code
black src tests

# Sort imports
isort src tests

# Check issues
flake8 src tests

# Fix automatically where possible
autopep8 --in-place --recursive src tests
```

## Best Practices

### 1. Keep Tests Fast

- Mark slow tests with `@pytest.mark.slow`
- Use small datasets in tests
- Mock external dependencies
- Run slow tests only on schedule

### 2. Maintain High Coverage

- Aim for >90% coverage
- Test edge cases
- Test error handling
- Use property-based testing

### 3. Keep Dependencies Updated

- Review Dependabot PRs
- Test with latest versions
- Pin versions for reproducibility

### 4. Monitor CI Health

- Check weekly scheduled runs
- Address flaky tests
- Keep build times under 5 minutes

### 5. Use Branches Effectively

- Create feature branches
- Keep PRs small and focused
- Ensure CI passes before merging
- Use draft PRs for work in progress

## Advanced Features

### Running Specific Tests in CI

You can trigger specific test categories manually:

1. Go to Actions → Tests → Run workflow
2. Check "Run slow tests" to include slow tests
3. Click "Run workflow"

### Viewing Coverage Reports

**Option 1: Codecov (recommended)**
- Go to codecov.io
- View interactive coverage report
- See coverage changes in PRs

**Option 2: Download Artifact**
- Go to Actions → Select run → Artifacts
- Download "coverage-report"
- Extract and open `htmlcov/index.html`

**Option 3: Local**
```bash
pytest --cov=src --cov-report=html
open htmlcov/index.html
```

### Matrix Testing Strategies

**Current strategy:** Test all versions on every push

**Alternative: Minimal + Full**
```yaml
strategy:
  matrix:
    python-version: ['3.11']  # Fast: test only latest
    include:
      - python-version: '3.8'  # Full: test on schedule
        if: github.event_name == 'schedule'
```

## Security Considerations

### Secrets Management

- Never commit secrets to repository
- Use GitHub Secrets for sensitive data
- Rotate tokens periodically
- Use minimal permissions

### Dependency Security

Consider adding security scanning:

```yaml
- name: Security scan
  run: |
    pip install safety bandit
    safety check
    bandit -r src/
```

## Getting Help

- **GitHub Actions docs:** https://docs.github.com/en/actions
- **pytest docs:** https://docs.pytest.org/
- **Codecov docs:** https://docs.codecov.com/
- **Issues:** Open a GitHub issue with "ci" label

## Summary

✅ **Automatic:** Tests run on every push/PR  
✅ **Multi-version:** Python 3.8, 3.9, 3.10, 3.11  
✅ **Fast:** Caching reduces build time to ~1-2 minutes  
✅ **Comprehensive:** Unit, property, integration, performance tests  
✅ **Coverage:** 90% minimum threshold enforced  
✅ **Quality:** Linting and formatting checks  
✅ **Flexible:** Fast tests by default, slow tests on schedule  

The CI/CD pipeline is production-ready and requires no additional configuration to work. Codecov integration is optional but recommended for better coverage visualization.

**Validates: Requirements 12.1, 12.8**
