#!/bin/bash
# Monitor caption extraction progress and notify when complete
# Usage: ./monitor_extraction.sh [check_interval_seconds]

# Point at your API host; defaults to a local instance.
API_URL="${API_URL:-http://localhost:8000}"
CHECK_INTERVAL="${1:-300}"  # Default: check every 5 minutes

echo "=== Caption Extraction Monitor ==="
echo "API: $API_URL"
echo "Check interval: ${CHECK_INTERVAL}s"
echo "Started: $(date)"
echo ""

while true; do
    # Get progress
    RESPONSE=$(curl -s "${API_URL}/dashboard/extraction/progress")

    if [ $? -ne 0 ]; then
        echo "[$(date +%H:%M:%S)] Failed to fetch progress"
        sleep $CHECK_INTERVAL
        continue
    fi

    # Parse JSON response
    TOTAL=$(echo "$RESPONSE" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('total_videos', 0))")
    COMPLETED=$(echo "$RESPONSE" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('completed', 0))")
    LLM_REFINED=$(echo "$RESPONSE" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('with_llm_refined', 0))")
    PENDING=$(echo "$RESPONSE" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('pending', 0))")
    PERCENT=$(echo "$RESPONSE" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('completion_percent', 0))")
    LLM_PERCENT=$(echo "$RESPONSE" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('llm_completion_percent', 0))")
    RATE=$(echo "$RESPONSE" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('rate_per_hour', 0))")
    ETA=$(echo "$RESPONSE" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('eta_display', 'unknown'))")
    IS_COMPLETE=$(echo "$RESPONSE" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('is_complete', False))")

    # Display progress
    echo "[$(date +%H:%M:%S)] Progress: ${COMPLETED}/${TOTAL} (${PERCENT}%) | LLM: ${LLM_REFINED} (${LLM_PERCENT}%) | Rate: ${RATE}/hr | ETA: ${ETA}"

    # Check if complete
    if [ "$IS_COMPLETE" = "True" ]; then
        echo ""
        echo "=============================================="
        echo "  EXTRACTION COMPLETE!"
        echo "  Total videos processed: $COMPLETED"
        echo "  Videos with LLM refinement: $LLM_REFINED"
        echo "  Completed at: $(date)"
        echo "=============================================="

        # macOS notification
        if command -v osascript &> /dev/null; then
            osascript -e 'display notification "Caption extraction complete! '${COMPLETED}' videos processed." with title "Captions Service"'
        fi

        # Sound alert (macOS)
        if command -v afplay &> /dev/null; then
            afplay /System/Library/Sounds/Glass.aiff 2>/dev/null
        fi

        exit 0
    fi

    sleep $CHECK_INTERVAL
done
