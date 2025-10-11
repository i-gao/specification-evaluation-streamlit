# Docker Execution Options for APPS Dataset

This document describes the new Docker execution options available in the APPS dataset, allowing you to choose between secure Docker sandboxing and faster local execution.

## Overview

The APPS dataset now supports two execution modes:
1. **Docker Execution** (default): Secure, sandboxed code execution using Docker containers
2. **Local Execution**: Faster execution in the current environment without Docker

## Usage

### Basic Usage with Docker (Default)

```python
from data.apps.data import APPSDataset

# Load dataset with Docker execution (default behavior)
dataset = APPSDataset(
    dev=True,
    docker_image="apps",
    use_docker=True,  # This is the default
    indexes=[0, 1, 2]
)
```

### Usage without Docker (Local Execution)

```python
from data.apps.data import APPSDataset

# Load dataset with local execution (no Docker)
dataset = APPSDataset(
    dev=True,
    docker_image=None,  # Not needed for local execution
    use_docker=False,   # Use local execution
    indexes=[0, 1, 2]
)
```

### Mixed Usage (Choose per Instance)

```python
from data.apps.data import APPSDataset

# You can create multiple instances with different execution modes
dataset_docker = APPSDataset(dev=True, use_docker=True)
dataset_local = APPSDataset(dev=True, use_docker=False)
```

## Security Considerations

### Docker Execution (use_docker=True)
- ✅ **Secure**: Code runs in isolated containers
- ✅ **Safe**: No access to host filesystem or network
- ✅ **Controlled**: Limited resource access and execution time
- ✅ **Reproducible**: Consistent execution environment

### Local Execution (use_docker=False)
- ⚠️ **Less Secure**: Code runs in your current environment
- ⚠️ **Risky**: Potential access to host files, network, and system resources
- ⚠️ **Uncontrolled**: Code can potentially modify your system
- ⚠️ **Environment Dependent**: Results may vary based on local setup

## Performance Comparison

| Aspect | Docker Execution | Local Execution |
|--------|------------------|-----------------|
| **Startup Time** | Slower (container startup) | Faster (immediate) |
| **Resource Usage** | Higher (container overhead) | Lower (direct execution) |
| **Security** | High (isolated) | Low (shared environment) |
| **Reliability** | High (consistent) | Variable (environment dependent) |

## When to Use Each Mode

### Use Docker Execution (use_docker=True) When:
- Running in production environments
- Executing untrusted or unknown code
- Need consistent execution environment
- Security is a priority
- Running on shared systems

### Use Local Execution (use_docker=False) When:
- Running in controlled, trusted environments
- Need faster execution for development/testing
- Docker is not available or desired
- Executing only trusted, known code
- Running on isolated development machines

## Configuration Options

### APPSDataset Constructor Parameters

- `dev` (bool): Whether to use dev split (True) or test split (False)
- `docker_image` (str, optional): Docker image name (required if use_docker=True)
- `use_docker` (bool): Whether to use Docker execution (default: True)
- `indexes` (List[int], optional): Specific problem indexes to load

### Environment Variables

You can also control Docker usage through environment variables:

```bash
# Force local execution
export USE_DOCKER=false

# Force Docker execution
export USE_DOCKER=true
```

## Example Scripts

See `example_usage.py` for complete working examples of both execution modes.

## Migration Guide

### From Previous Versions

If you're upgrading from a previous version that only supported Docker:

```python
# Old code (Docker only)
dataset = APPSDataset(dev=True, docker_image="apps")

# New code (same behavior, explicit)
dataset = APPSDataset(dev=True, docker_image="apps", use_docker=True)

# New code (local execution)
dataset = APPSDataset(dev=True, use_docker=False)
```

### Backward Compatibility

The new `use_docker` parameter defaults to `True`, so existing code will continue to work without modification.

## Troubleshooting

### Docker Issues
- Ensure Docker daemon is running
- Check Docker image exists: `docker images | grep apps`
- Verify Docker permissions for your user

### Local Execution Issues
- Ensure Python environment has required dependencies
- Check file permissions in current directory
- Verify no conflicting environment variables

### Common Errors

```
ValueError: docker_image must be provided when use_docker=True
```
**Solution**: Either provide a docker_image or set use_docker=False

```
Warning: Running code without Docker sandboxing!
```
**Solution**: This is expected when use_docker=False. Suppress with warnings.filterwarnings() if needed.

## Contributing

When adding new features that involve code execution:
1. Always support both Docker and local execution modes
2. Use the `run_python_script_with_json_input_auto` function from `utils.code_sandbox`
3. Pass through the `use_docker` parameter to maintain consistency
4. Add appropriate security warnings for local execution

## Related Files

- `data/apps/data.py`: Main dataset implementation
- `utils/code_sandbox.py`: Execution engine with Docker/local support
- `data/apps/example_usage.py`: Usage examples
- `data/apps/README_DOCKER_OPTIONS.md`: This documentation



