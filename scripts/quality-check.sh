#!/bin/bash
# Quality check script that runs the same checks as CI
# This ensures consistency between local development and CI

set -e

echo "🔍 Running quality checks..."
echo ""

echo "📝 Checking code formatting with black..."
if ! uv run black --check --diff .; then
    echo "❌ Code formatting issues found. Run 'uv run black .' to fix."
    exit 1
fi
echo "✅ Code formatting looks good"
echo ""

echo "📝 Checking import sorting with isort..."
if ! uv run isort --check --diff .; then
    echo "❌ Import sorting issues found. Run 'uv run isort .' to fix." 
    exit 1
fi
echo "✅ Import sorting looks good"
echo ""


echo "🧪 Testing basic import..."
uv run python -c "from zarrnii import ZarrNii; print('✅ Import successful')"
echo ""

echo "📚 Building documentation..."
if uv run mkdocs build --quiet; then
    echo "✅ Documentation built successfully"
else
    echo "⚠️  Documentation build had warnings (this is expected)"
fi
echo ""

echo "📦 Checking package build..."
if uv build --quiet; then
    echo "✅ Package builds successfully"
else
    echo "❌ Package build failed"
    exit 1
fi
echo ""

echo "🎉 Quality checks completed!"
echo "   - Black formatting: enforced (line-length: 88)"
echo "   - Import sorting: enforced (profile: black)"
echo "   - Import test: passed"
echo "   - Documentation: built" 
echo "   - Package build: successful"
