#!/bin/bash
# Test script to verify path filtering logic locally
# Usage: ./.github/workflows/test-path-filter.sh

set -e

echo "================================================"
echo "Testing GitHub Actions Path Filter Logic"
echo "================================================"

# Function to test file change detection
test_file() {
    local file=$1
    local expected_docs_only=$2
    local expected_api=$3
    local expected_celery=$4
    local expected_docker=$5

    echo ""
    echo "Testing: $file"
    echo "  Expected docs-only: $expected_docs_only"
    echo "  Expected api: $expected_api"
    echo "  Expected celery: $expected_celery"
    echo "  Expected docker: $expected_docker"

    # Simulate change detection logic
    DOCS_ONLY=true
    API_CHANGED=false
    CELERY_CHANGED=false
    DOCKER_CHANGED=false

    case "$file" in
        *.md|LICENSE|.gitignore)
            echo "  → Documentation file"
            ;;
        .github/workflows/*)
            echo "  → Workflow file (infrastructure - rebuild all)"
            DOCKER_CHANGED=true
            API_CHANGED=true
            CELERY_CHANGED=true
            DOCS_ONLY=false
            ;;
        api/*|scrapers/*|config/*)
            echo "  → API/scraper file"
            API_CHANGED=true
            CELERY_CHANGED=true
            DOCS_ONLY=false
            ;;
        tasks/*|training/*)
            echo "  → Celery task file"
            CELERY_CHANGED=true
            DOCS_ONLY=false
            ;;
        database/*)
            echo "  → Database file"
            API_CHANGED=true
            CELERY_CHANGED=true
            DOCS_ONLY=false
            ;;
        Dockerfile|docker-compose.yml|requirements.txt)
            echo "  → Docker/dependency file"
            DOCKER_CHANGED=true
            API_CHANGED=true
            CELERY_CHANGED=true
            DOCS_ONLY=false
            ;;
        utils/*)
            echo "  → Utility file"
            API_CHANGED=true
            CELERY_CHANGED=true
            DOCS_ONLY=false
            ;;
        *)
            echo "  → Other file"
            DOCS_ONLY=false
            ;;
    esac

    # Verify expectations
    local failed=false

    if [ "$DOCS_ONLY" != "$expected_docs_only" ]; then
        echo "  ❌ FAIL: docs-only is $DOCS_ONLY, expected $expected_docs_only"
        failed=true
    fi

    if [ "$API_CHANGED" != "$expected_api" ]; then
        echo "  ❌ FAIL: api is $API_CHANGED, expected $expected_api"
        failed=true
    fi

    if [ "$CELERY_CHANGED" != "$expected_celery" ]; then
        echo "  ❌ FAIL: celery is $CELERY_CHANGED, expected $expected_celery"
        failed=true
    fi

    if [ "$DOCKER_CHANGED" != "$expected_docker" ]; then
        echo "  ❌ FAIL: docker is $DOCKER_CHANGED, expected $expected_docker"
        failed=true
    fi

    if [ "$failed" = false ]; then
        echo "  ✓ PASS"
    fi
}

# Test cases
# Format: test_file <file> <docs_only> <api> <celery> <docker>

echo ""
echo "=== Documentation Files (Should Skip Deployment) ==="
test_file "CLAUDE.md" "true" "false" "false" "false"
test_file "README.md" "true" "false" "false" "false"
test_file "docs/guide.md" "true" "false" "false" "false"
test_file "LICENSE" "true" "false" "false" "false"
test_file ".gitignore" "true" "false" "false" "false"

echo ""
echo "=== API Files (Rebuild API + Celery) ==="
test_file "api/main.py" "false" "true" "true" "false"
test_file "scrapers/reddit_scraper.py" "false" "true" "true" "false"
test_file "config/settings.py" "false" "true" "true" "false"

echo ""
echo "=== Celery Files (Rebuild Celery Only) ==="
test_file "tasks/scraping_tasks.py" "false" "false" "true" "false"
test_file "training/train_lora.py" "false" "false" "true" "false"

echo ""
echo "=== Database Files (Rebuild API + Celery) ==="
test_file "database/models.py" "false" "true" "true" "false"
test_file "database/migrations/add_table.py" "false" "true" "true" "false"

echo ""
echo "=== Docker/Infrastructure Files (Rebuild All) ==="
test_file ".github/workflows/deploy.yml" "false" "true" "true" "true"
test_file "Dockerfile" "false" "true" "true" "true"
test_file "docker-compose.yml" "false" "true" "true" "true"
test_file "requirements.txt" "false" "true" "true" "true"

echo ""
echo "=== Utility Files (Rebuild API + Celery) ==="
test_file "utils/metrics_logger.py" "false" "true" "true" "false"

echo ""
echo "================================================"
echo "Test Complete"
echo "================================================"
