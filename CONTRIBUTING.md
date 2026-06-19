# Contributing to PolyMind

Thank you for your interest in contributing! This document provides guidelines for contributing to PolyMind.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/your-username/polymind.git`
3. Create a feature branch: `git checkout -b feat/amazing-feature`
4. Install dependencies: `poetry install`
5. Make your changes
6. Run tests: `make test`
7. Run linting: `make lint`
8. Commit your changes
9. Push to your fork and open a Pull Request

## Branch Strategy

We follow Git Flow with Conventional Commits:

| Branch | Purpose |
|--------|---------|
| `main` | Production-only, protected |
| `develop` | Integration branch |
| `phase/X-name` | Implementation phases |
| `feat/XXX` | New features |
| `fix/XXX` | Bug fixes |
| `docs/XXX` | Documentation |
| `refactor/XXX` | Code refactoring |
| `chore/XXX` | CI, deps, tooling |

## Commit Messages

We use [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <short description>

[optional body — WHY this change was made]

[optional footer — Breaking changes, closes #issue]
```

**Types:** `feat` | `fix` | `docs` | `refactor` | `test` | `chore` | `perf` | `ci`

**Examples:**
```
feat(graph): add HippoRAG multi-hop retrieval
fix(critic): prevent infinite retry loop when max_retries exceeded
docs(learning): add LangGraph deep dive explanation
test(asr): add Arabic language detection test fixture
```

## Code Style

- **Formatter:** Ruff (Black-compatible)
- **Linter:** Ruff
- **Type checker:** mypy (strict mode)
- **Line length:** 88 characters

## Testing Requirements

- All new code must have unit tests
- Coverage target: 80% minimum on `src/polymind/`
- Test naming: `test_<module>_<scenario>_<expected>`
- Use `pytest` fixtures in `conftest.py` (never inline)
- Parametrize over edge cases

## Architecture Rules

- Domain layer has ZERO external dependencies
- All infrastructure access via ABC interfaces
- No business logic in API routes
- No direct DB calls in agents — always via Repository pattern
- Use Dependency Injection everywhere

## Pull Request Process

1. Update documentation if needed
2. Add tests for new functionality
3. Ensure all CI checks pass
4. Request review from maintainers
5. Address review feedback
6. Squash and merge

## Questions?

Open an issue or start a discussion on GitHub.
