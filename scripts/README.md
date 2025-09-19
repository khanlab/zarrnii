# ZarrNii Quality Check Scripts

This directory contains scripts to ensure consistent code quality across local development and CI environments.

## Scripts

### `quality-check.sh`
**Purpose**: Development-friendly quality checks that match CI behavior exactly.

**Usage**:
```bash
./scripts/quality-check.sh
```

**What it does**:
- ✅ Enforces Black formatting (line-length: 88)
- ✅ Enforces isort import sorting (profile: black)
- ⚠️ Reports flake8 linting issues (allows pre-existing issues)
- ✅ Tests basic import functionality
- ✅ Builds documentation
- ✅ Validates package build

**Exit behavior**: Fails only on formatting/import sorting issues or critical failures. Pre-existing linting issues are reported but don't cause failure.

### `quality-check-strict.sh`
**Purpose**: Strict quality checks for new code that fails on any linting issues.

**Usage**:
```bash
./scripts/quality-check-strict.sh
```

**What it does**:
- ✅ Same as `quality-check.sh` but fails on ANY linting issues
- Intended for use when all linting issues are resolved

## Configuration Harmony

All scripts use consistent tool configurations:

| Tool | Configuration | Value |
|------|---------------|-------|
| Black | line-length | 88 |
| isort | profile | black |
| isort | line_length | 88 |
| flake8 | max-line-length | 88 |
| flake8 | extend-ignore | E203, W503 |

These settings are synchronized across:
- `pyproject.toml`
- `setup.cfg` 
- `.pre-commit-config.yaml`
- GitHub Actions CI

## Integration

### Local Development
```bash
# Run quality checks (recommended)
./scripts/quality-check.sh

# Or use justfile
just quality

# Or individual tools
uv run black .
uv run isort .
uv run flake8 .
```

### CI/CD
The CI workflow uses `quality-check.sh` to ensure identical behavior between local development and continuous integration.

### Pre-commit Hooks
Pre-commit hooks are configured with the same tool versions and settings to catch issues before commit.

## Troubleshooting

**Import sorting issues**: Run `uv run isort .` to fix automatically.

**Formatting issues**: Run `uv run black .` to fix automatically.

**Linting issues**: These require manual fixes. The quality check script reports the count but doesn't fail on pre-existing issues.

**Pre-commit failures**: Use `git commit --no-verify` for documentation-only changes if pre-commit has network issues.