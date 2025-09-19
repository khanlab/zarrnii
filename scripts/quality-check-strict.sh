#!/bin/bash
# Strict quality check script for CI - fails on any linting issues
# This ensures new code meets quality standards

set -e

echo "🔍 Running strict quality checks for CI..."
echo ""

echo "📝 Checking code formatting with black..."
if ! uv run black --check --diff .; then
    echo "❌ Code formatting issues found"
    exit 1
fi
echo "✅ Code formatting is correct"
echo ""

echo "📝 Checking import sorting with isort..."
if ! uv run isort --check --diff .; then
    echo "❌ Import sorting issues found"
    exit 1
fi
echo "✅ Import sorting is correct"
echo ""

echo "📝 Running linting with flake8..."
if ! uv run flake8 .; then
    echo "❌ Linting issues found"
    exit 1
fi
echo "✅ Linting passed"
echo ""

echo "🧪 Testing basic import..."
uv run python -c "from zarrnii import ZarrNii; print('✅ Import successful')"
echo ""

echo "📦 Checking package build..."
if ! uv build --quiet; then
    echo "❌ Package build failed"
    exit 1
fi
echo "✅ Package builds successfully"
echo ""

echo "🎉 All strict quality checks passed!"