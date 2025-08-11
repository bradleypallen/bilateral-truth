# PyPI Publishing Setup Guide

This guide explains how to set up automated PyPI publishing for the bilateral-truth package using GitHub Actions and PyPI's trusted publishing feature.

## Overview

We use GitHub Actions with PyPI's trusted publishing (OIDC) for secure, automatic package publishing without storing API tokens. The workflow:

1. **Test PyPI**: Publishes to Test PyPI on every push to `main` branch
2. **Production PyPI**: Publishes to PyPI only when a GitHub release is created

## Setup Steps

### 1. Configure PyPI Trusted Publishing

#### For Test PyPI:
1. Go to https://test.pypi.org/manage/account/
2. Navigate to "Publishing" → "Add a new pending publisher"
3. Fill in:
   - **PyPI project name**: `bilateral-truth`
   - **Owner**: `bradleypallen` (your GitHub username)
   - **Repository name**: `bilateral-truth`
   - **Workflow name**: `ci.yml`
   - **Environment name**: `test-pypi`

#### For Production PyPI:
1. Go to https://pypi.org/manage/account/
2. Navigate to "Publishing" → "Add a new pending publisher"  
3. Fill in:
   - **PyPI project name**: `bilateral-truth`
   - **Owner**: `bradleypallen` (your GitHub username)
   - **Repository name**: `bilateral-truth`
   - **Workflow name**: `ci.yml`
   - **Environment name**: `pypi`

### 2. Configure GitHub Repository Environments

1. Go to your GitHub repository settings
2. Navigate to "Environments"
3. Create two environments:

#### Test PyPI Environment:
- **Name**: `test-pypi`
- **Protection rules**: None needed (auto-publishes on main branch pushes)

#### Production PyPI Environment:
- **Name**: `pypi`
- **Protection rules**: 
  - ✅ Required reviewers (add yourself)
  - ✅ Restrict deployments to protected branches
  - Add `main` as protected branch

### 3. Workflow Overview

The workflows are already configured in `.github/workflows/`:

- **`ci.yml`**: Main CI/CD pipeline with testing and publishing
- **`release.yml`**: Manual release creation workflow
- **`pr-checks.yml`**: Quick checks for pull requests

## Publishing Process

### Development Publishing (Test PyPI)
1. Push changes to `main` branch
2. All tests pass automatically
3. Package is built and published to Test PyPI
4. Install with: `pip install -i https://test.pypi.org/simple/ bilateral-truth`

### Production Publishing (PyPI)
1. Use the manual release workflow:
   ```bash
   # Go to GitHub Actions → Release → Run workflow
   # Enter version (e.g., "1.0.0")
   ```
2. This creates a GitHub release
3. The release triggers the production PyPI publish
4. Install with: `pip install bilateral-truth`

## Version Management

Versions are managed in two places:
- `pyproject.toml`: `version = "0.1.0"`
- `bilateral_truth/__init__.py`: `__version__ = "0.1.0"`

The release workflow automatically updates both files.

## Package Structure

The package now has optional dependencies:

```bash
# Core package (minimal dependencies)
pip install bilateral-truth

# With OpenAI support
pip install bilateral-truth[openai]

# With Anthropic support  
pip install bilateral-truth[anthropic]

# With all LLM providers
pip install bilateral-truth[all]

# Development dependencies
pip install bilateral-truth[dev]
```

## Testing the Setup

1. **Test the build locally**:
   ```bash
   python -m build
   twine check dist/*
   ```

2. **Test installation from wheel**:
   ```bash
   pip install dist/*.whl
   bilateral-truth --help
   ```

3. **Test CI pipeline**:
   - Push a small change to `main`
   - Check GitHub Actions for successful run
   - Verify package appears on Test PyPI

## Security Features

- ✅ **No API tokens stored**: Uses OIDC trusted publishing
- ✅ **Environment protection**: Production requires manual approval
- ✅ **Signature verification**: PyPI verifies GitHub's signed tokens
- ✅ **Audit trail**: All publishes logged in GitHub Actions

## Troubleshooting

### Common Issues:

1. **Trusted publisher not found**:
   - Ensure PyPI trusted publisher is configured correctly
   - Double-check repository name, workflow name, and environment name

2. **Environment not found**:
   - Create the environment in GitHub repository settings
   - Ensure environment name matches workflow configuration

3. **Build fails**:
   - Check `pyproject.toml` syntax
   - Ensure all files are properly included
   - Test build locally first

4. **Version conflicts**:
   - PyPI doesn't allow re-uploading same version
   - Use `skip-existing: true` for Test PyPI
   - Increment version for new releases

## Monitoring

- **GitHub Actions**: Monitor workflow runs
- **PyPI Download Stats**: Check package adoption
- **Codecov**: Track test coverage
- **GitHub Releases**: Track version history

## Next Steps

1. Set up PyPI trusted publishing (follow steps above)
2. Create GitHub environments
3. Test with a development push to `main`
4. Create your first release using the Release workflow