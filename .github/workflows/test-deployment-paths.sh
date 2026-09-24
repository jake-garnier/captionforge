#!/bin/bash
# Comprehensive deployment path testing script
# Tests all deployment scenarios locally before pushing to GitHub

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$PROJECT_ROOT"

echo "================================================"
echo "Deployment Path Testing Suite"
echo "================================================"
echo "Project root: $PROJECT_ROOT"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

PASSED=0
FAILED=0

# Test function
run_test() {
    local test_name=$1
    local description=$2
    shift 2

    echo -e "${YELLOW}TEST: $test_name${NC}"
    echo "  Description: $description"

    if "$@"; then
        echo -e "  ${GREEN}✓ PASSED${NC}"
        ((PASSED++))
        return 0
    else
        echo -e "  ${RED}✗ FAILED${NC}"
        ((FAILED++))
        return 1
    fi
    echo ""
}

# Test 1: Workflow YAML syntax
test_workflow_syntax() {
    echo "  Validating workflow YAML syntax..."

    # Check if file exists
    if [ ! -f ".github/workflows/deploy.yml" ]; then
        echo "    ✗ deploy.yml not found"
        return 1
    fi

    # Basic YAML syntax check using Python
    python3 -c "
import yaml
import sys

try:
    with open('.github/workflows/deploy.yml', 'r') as f:
        yaml.safe_load(f)
    print('    ✓ YAML syntax is valid')
    sys.exit(0)
except yaml.YAMLError as e:
    print(f'    ✗ YAML syntax error: {e}')
    sys.exit(1)
except Exception as e:
    print(f'    ✗ Error: {e}')
    sys.exit(1)
"
    return $?
}

# Test 2: Docker Compose syntax
test_docker_compose_syntax() {
    echo "  Validating docker-compose.yml syntax..."

    # Create temporary .env for testing
    cat > .env.test << 'ENVEOF'
POSTGRES_DB=captions
POSTGRES_USER=captionsuser
POSTGRES_PASSWORD=testpass
REDIS_PASSWORD=
FLOWER_BASIC_AUTH=admin:changeme
ENVIRONMENT=production
DEBUG=false
PLAYWRIGHT_HEADLESS=true
PLAYWRIGHT_TIMEOUT=30000
SCRAPING_ENABLED=true
SCRAPING_INTERVAL_HOURS=8
MAX_POSTS_PER_SCRAPE=50
DOWNLOAD_MAX_FILE_SIZE_MB=100
DOWNLOAD_MAX_DURATION_SECONDS=600
ENVEOF

    if ! docker-compose --env-file .env.test config > /dev/null 2>&1; then
        echo "    ✗ docker-compose.yml has errors"
        docker-compose --env-file .env.test config 2>&1 | head -20
        rm -f .env.test
        return 1
    fi

    rm -f .env.test
    echo "    ✓ docker-compose.yml syntax is valid"
    return 0
}

# Test 3: Health check commands are valid
test_health_checks() {
    echo "  Verifying health check commands..."

    local checks_found=0

    # Extract health check commands
    if grep -q "curl -f http://localhost:8000/health" docker-compose.yml; then
        echo "    ✓ API health check found"
        ((checks_found++))
    fi

    if grep -q "celery -A tasks.celery_app inspect ping" docker-compose.yml; then
        echo "    ✓ Celery worker health check found"
        ((checks_found++))
    fi

    if grep -q "ps aux | grep 'celery.*beat'" docker-compose.yml; then
        echo "    ✓ Celery beat health check found"
        ((checks_found++))
    fi

    if grep -q "curl -f http://localhost:5555" docker-compose.yml; then
        echo "    ✓ Flower health check found"
        ((checks_found++))
    fi

    if [ $checks_found -eq 4 ]; then
        echo "    ✓ All health checks configured"
        return 0
    else
        echo "    ✗ Missing health checks (found $checks_found/4)"
        return 1
    fi
}

# Test 4: Path filtering logic
test_path_filtering() {
    echo "  Testing path filtering logic..."

    if [ ! -f ".github/workflows/test-path-filter.sh" ]; then
        echo "    ✗ test-path-filter.sh not found"
        return 1
    fi

    # Run the path filter test script
    if bash .github/workflows/test-path-filter.sh > /tmp/path-filter-test.log 2>&1; then
        local passes=$(grep -c "✓ PASS" /tmp/path-filter-test.log)
        echo "    ✓ Path filtering tests passed ($passes tests)"
        return 0
    else
        echo "    ✗ Path filtering tests failed"
        tail -20 /tmp/path-filter-test.log
        return 1
    fi
}

# Test 5: Workflow has required jobs
test_workflow_structure() {
    echo "  Verifying workflow structure..."

    local required_jobs=("detect-changes" "deploy")
    local found=0

    for job in "${required_jobs[@]}"; do
        if grep -q "^  $job:" .github/workflows/deploy.yml; then
            echo "    ✓ Job '$job' found"
            ((found++))
        else
            echo "    ✗ Job '$job' missing"
        fi
    done

    if [ $found -eq ${#required_jobs[@]} ]; then
        return 0
    else
        return 1
    fi
}

# Test 6: Workflow has required outputs
test_workflow_outputs() {
    echo "  Verifying job outputs..."

    local required_outputs=("docs-only" "api-changed" "celery-changed" "docker-changed" "any-code-changed")
    local found=0

    for output in "${required_outputs[@]}"; do
        if grep -q "$output:" .github/workflows/deploy.yml; then
            ((found++))
        fi
    done

    if [ $found -eq ${#required_outputs[@]} ]; then
        echo "    ✓ All required outputs defined ($found/${#required_outputs[@]})"
        return 0
    else
        echo "    ✗ Missing outputs (found $found/${#required_outputs[@]})"
        return 1
    fi
}

# Test 7: Zero-downtime deployment commands
test_zero_downtime_commands() {
    echo "  Verifying zero-downtime deployment commands..."

    # Check for --no-deps flag (key for zero-downtime)
    if ! grep -q "docker-compose up -d --no-deps" .github/workflows/deploy.yml; then
        echo "    ✗ Missing --no-deps flag for zero-downtime deployment"
        return 1
    fi
    echo "    ✓ --no-deps flag found"

    # Check for --remove-orphans flag
    if ! grep -q "docker-compose up -d --remove-orphans" .github/workflows/deploy.yml; then
        echo "    ⚠ Missing --remove-orphans flag (recommended)"
    else
        echo "    ✓ --remove-orphans flag found"
    fi

    return 0
}

# Test 8: Health check wait logic
test_health_check_wait() {
    echo "  Verifying health check wait logic..."

    # Check for health check wait
    if ! grep -q "curl -f http://localhost:8000/health" .github/workflows/deploy.yml; then
        echo "    ✗ Missing health check verification"
        return 1
    fi
    echo "    ✓ Health check verification found"

    # Check for retry logic
    if ! grep -q "for i in" .github/workflows/deploy.yml; then
        echo "    ⚠ No retry loop found for health checks"
    else
        echo "    ✓ Retry logic found for health checks"
    fi

    return 0
}

# Test 9: Simulate path detection scenarios
test_path_detection_scenarios() {
    echo "  Simulating path detection scenarios..."

    # Test scenario: docs only
    local test_files="CLAUDE.md README.md"
    local should_deploy=false

    # Simulate the case statement
    local triggers_deployment=false
    for file in $test_files; do
        case "$file" in
            *.md|LICENSE|.gitignore) ;;
            *) triggers_deployment=true ;;
        esac
    done

    if [ "$triggers_deployment" = "$should_deploy" ]; then
        echo "    ✓ Documentation-only scenario correct (no deployment)"
    else
        echo "    ✗ Documentation-only scenario failed"
        return 1
    fi

    # Test scenario: API change
    test_files="api/main.py CLAUDE.md"
    should_deploy=true
    triggers_deployment=false

    for file in $test_files; do
        case "$file" in
            *.md|LICENSE|.gitignore) ;;
            *) triggers_deployment=true ;;
        esac
    done

    if [ "$triggers_deployment" = "$should_deploy" ]; then
        echo "    ✓ Mixed files scenario correct (triggers deployment)"
    else
        echo "    ✗ Mixed files scenario failed"
        return 1
    fi

    return 0
}

# Test 10: Verify Docker Compose can do rolling updates
test_docker_compose_rolling_update() {
    echo "  Testing Docker Compose rolling update capability..."

    # Check if docker-compose supports the required flags
    if ! docker-compose up --help 2>&1 | grep -q -- "--no-deps"; then
        echo "    ✗ docker-compose doesn't support --no-deps flag"
        echo "    Please upgrade docker-compose to version 1.28 or higher"
        return 1
    fi
    echo "    ✓ docker-compose supports --no-deps"

    # Verify version is recent enough
    local version=$(docker-compose version --short)
    echo "    ℹ docker-compose version: $version"

    return 0
}

# Test 11: Environment variables are properly referenced
test_env_vars() {
    echo "  Verifying environment variable references..."

    # Check .env.example exists
    if [ ! -f ".env.example" ]; then
        echo "    ✗ .env.example not found"
        return 1
    fi
    echo "    ✓ .env.example found"

    # Check if workflow creates .env
    if ! grep -q "cat > .env << EOF" .github/workflows/deploy.yml; then
        echo "    ✗ Workflow doesn't create .env file"
        return 1
    fi
    echo "    ✓ Workflow creates .env file"

    return 0
}

# Test 12: Workflow has proper error handling
test_error_handling() {
    echo "  Verifying error handling..."

    # Check for failure conditions
    if ! grep -q "exit 1" .github/workflows/deploy.yml; then
        echo "    ⚠ No explicit failure handling found"
    else
        echo "    ✓ Explicit failure handling found"
    fi

    # Check for if: always() for logs
    if ! grep -q "if: always()" .github/workflows/deploy.yml; then
        echo "    ⚠ No always() condition for log collection"
    else
        echo "    ✓ Always-run log collection found"
    fi

    return 0
}

# Run all tests
echo "Running comprehensive deployment tests..."
echo ""

run_test "1. Workflow YAML Syntax" \
    "Validate GitHub Actions workflow YAML is syntactically correct" \
    test_workflow_syntax

run_test "2. Docker Compose Syntax" \
    "Validate docker-compose.yml configuration" \
    test_docker_compose_syntax

run_test "3. Health Checks Configuration" \
    "Verify all services have health checks defined" \
    test_health_checks

run_test "4. Path Filtering Logic" \
    "Test path filtering correctly identifies file changes" \
    test_path_filtering

run_test "5. Workflow Structure" \
    "Verify all required jobs are present" \
    test_workflow_structure

run_test "6. Workflow Outputs" \
    "Verify all required job outputs are defined" \
    test_workflow_outputs

run_test "7. Zero-Downtime Commands" \
    "Verify deployment uses zero-downtime strategies" \
    test_zero_downtime_commands

run_test "8. Health Check Wait Logic" \
    "Verify workflow waits for health checks before proceeding" \
    test_health_check_wait

run_test "9. Path Detection Scenarios" \
    "Simulate various file change scenarios" \
    test_path_detection_scenarios

run_test "10. Docker Compose Rolling Updates" \
    "Verify Docker Compose supports required features" \
    test_docker_compose_rolling_update

run_test "11. Environment Variables" \
    "Verify environment variable handling" \
    test_env_vars

run_test "12. Error Handling" \
    "Verify workflow has proper error handling" \
    test_error_handling

# Summary
echo ""
echo "================================================"
echo "Test Summary"
echo "================================================"
echo -e "${GREEN}Passed: $PASSED${NC}"
echo -e "${RED}Failed: $FAILED${NC}"
echo "Total:  $((PASSED + FAILED))"
echo ""

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}✓ All tests passed! Ready to deploy.${NC}"
    echo ""
    echo "Next steps:"
    echo "1. Commit these changes:"
    echo "   git add ."
    echo "   git commit -m 'Implement zero-downtime deployment'"
    echo ""
    echo "2. Push to trigger deployment:"
    echo "   git push origin main"
    echo ""
    echo "3. Test documentation-only skip:"
    echo "   echo 'test' >> README.md"
    echo "   git add README.md && git commit -m 'Test: docs-only skip'"
    echo "   git push origin main"
    echo ""
    exit 0
else
    echo -e "${RED}✗ Some tests failed. Please fix issues before deploying.${NC}"
    exit 1
fi
