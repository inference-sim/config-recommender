# Copilot Instructions for config-explorer

## Repository Overview
This repository is `config-explorer`, a Python-based tool for configuration exploration for inference workloads.

## Tech Stack
- **Language**: Python
- **Primary Purpose**: Configuration exploration for inference systems

## Coding Standards

### Python Style
- Follow PEP 8 style guidelines
- Use meaningful variable and function names
- Include docstrings for all public functions, classes, and modules
- Prefer type hints for function parameters and return values

### Code Organization
- Keep functions focused and single-purpose
- Use appropriate error handling and logging
- Write clean, readable code with minimal comments (let code be self-documenting)
- Add comments only when explaining complex logic or non-obvious decisions

## Development Practices

### Before Making Changes
1. Understand the existing code structure
2. Check for existing tests related to your changes
3. Follow the established patterns in the codebase

### Testing
- Write tests for new functionality
- Ensure existing tests pass before submitting changes
- Use pytest as the testing framework (if not already configured)
- Keep test files in a `tests/` directory

### Version Control
- Write clear, descriptive commit messages
- Keep commits focused and atomic
- Reference issue numbers in commit messages when applicable

## File Structure
- Source code should be organized in appropriate modules
- Tests should be in a `tests/` directory
- Documentation should be in markdown format
- Configuration files should be in the root or a `config/` directory

## Dependencies
- Use virtual environments for development
- Keep dependencies minimal and well-documented
- Document any new dependencies and their purpose
- Use `requirements.txt` or `pyproject.toml` for dependency management

## Best Practices
- Prioritize code clarity over cleverness
- Optimize for maintainability
- Consider edge cases and error conditions
- Use appropriate data structures for the problem
- Avoid premature optimization
- Write secure code (validate inputs, handle sensitive data properly)
