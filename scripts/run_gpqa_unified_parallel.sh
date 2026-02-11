#!/bin/bash
#
# GPQA Unified Parallel Evaluation
# Runs GA-5gen with checkpoints at generations 0, 3, and 5
# This captures baseline, ga3, and ga5 results in a single run
#
# Usage:
#   ./run_gpqa_unified_parallel.sh
#   ./run_gpqa_unified_parallel.sh --workers 4
#   ./run_gpqa_unified_parallel.sh --workers 8 --questions 50

# Don't use set -e - we want to continue monitoring even if grep/printf fail

# Default settings
WORKERS=8
MODEL="gpt-5-mini"
RUN_NAME="gpqa_unified_new"
TOTAL_QUESTIONS=198

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --workers)
            WORKERS="$2"
            shift 2
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        --name)
            RUN_NAME="$2"
            shift 2
            ;;
        --questions)
            TOTAL_QUESTIONS="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
PYTHON_SCRIPT="$SCRIPT_DIR/run_gpqa_unified.py"
OUTPUT_DIR="$PROJECT_ROOT/output/$RUN_NAME"
LOG_DIR="$OUTPUT_DIR/logs"
RESULTS_DIR="$OUTPUT_DIR/results"

# Create output directories
mkdir -p "$LOG_DIR"
mkdir -p "$RESULTS_DIR"

echo "============================================================"
echo "GPQA Unified Parallel Evaluation"
echo "============================================================"
echo "Workers: $WORKERS"
echo "Model: $MODEL"
echo "Total Questions: $TOTAL_QUESTIONS"
echo "Output Directory: $OUTPUT_DIR"
echo "============================================================"
echo ""

# Calculate batch sizes
BATCH_SIZE=$(( (TOTAL_QUESTIONS + WORKERS - 1) / WORKERS ))

# Store worker PIDs
declare -a WORKER_PIDS

# Launch workers
echo "Launching $WORKERS workers..."
for ((i=0; i<WORKERS; i++)); do
    START_IDX=$((i * BATCH_SIZE))
    END_IDX=$((START_IDX + BATCH_SIZE))
    
    # Clamp end index
    if [ $END_IDX -gt $TOTAL_QUESTIONS ]; then
        END_IDX=$TOTAL_QUESTIONS
    fi
    
    # Skip if no questions for this worker
    if [ $START_IDX -ge $TOTAL_QUESTIONS ]; then
        continue
    fi
    
    WORKER_LOG="$LOG_DIR/worker_$i.log"
    WORKER_RESULT="$RESULTS_DIR/batch_$i.json"
    
    echo "  Worker $i: questions $START_IDX-$((END_IDX-1)) -> $WORKER_RESULT"
    
    python "$PYTHON_SCRIPT" \
        --start $START_IDX \
        --end $END_IDX \
        --model "$MODEL" \
        --output-file "$WORKER_RESULT" \
        > "$WORKER_LOG" 2>&1 &
    
    WORKER_PIDS+=($!)
done

echo ""
echo "All workers launched. Monitoring progress..."
echo ""

# Monitor progress
while true; do
    RUNNING=0
    for pid in "${WORKER_PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            ((RUNNING++))
        fi
    done
    
    # Count completed results (handle empty case)
    RESULTS_COMPLETED=$(find "$RESULTS_DIR" -name "*.json" 2>/dev/null | xargs cat 2>/dev/null | grep -c '"question_index"' 2>/dev/null || echo 0)
    
    printf "\rWorkers running: %d / %d | Questions completed: %d    " "$RUNNING" "$WORKERS" "$RESULTS_COMPLETED"
    
    if [ $RUNNING -eq 0 ]; then
        break
    fi
    
    sleep 10
done

echo ""
echo ""
echo "All workers completed. Merging results..."

# Merge results
MERGED_FILE="$OUTPUT_DIR/results.json"

python3 << EOF
import json
import os
from pathlib import Path

results_dir = Path("$RESULTS_DIR")
merged = {
    "metadata": {
        "model": "$MODEL",
        "total_questions": $TOTAL_QUESTIONS,
        "evaluation_type": "unified_ga5gen_parallel"
    },
    "questions": [],
    "summary": {
        "baseline": {"correct": 0, "total": 0, "accuracy": 0, "total_time": 0, "total_cost": 0},
        "ga3": {"correct": 0, "total": 0, "accuracy": 0, "total_time": 0, "total_cost": 0},
        "ga5": {"correct": 0, "total": 0, "accuracy": 0, "total_time": 0, "total_cost": 0}
    }
}

# Load all batch files
for batch_file in sorted(results_dir.glob("batch_*.json")):
    try:
        with open(batch_file, 'r') as f:
            batch = json.load(f)
        merged["questions"].extend(batch.get("questions", []))
    except Exception as e:
        print(f"Error loading {batch_file}: {e}")

# Sort by question index
merged["questions"].sort(key=lambda q: q.get("question_index", 0))

# Calculate summary
for q in merged["questions"]:
    for checkpoint in ["baseline", "ga3", "ga5"]:
        result = q.get(checkpoint, {})
        merged["summary"][checkpoint]["total"] += 1
        if result.get("correct"):
            merged["summary"][checkpoint]["correct"] += 1
        merged["summary"][checkpoint]["total_time"] += result.get("time_seconds", 0)
        merged["summary"][checkpoint]["total_cost"] += result.get("cost_usd", 0)

# Calculate accuracies
for checkpoint in ["baseline", "ga3", "ga5"]:
    total = merged["summary"][checkpoint]["total"]
    correct = merged["summary"][checkpoint]["correct"]
    merged["summary"][checkpoint]["accuracy"] = round(correct / total * 100, 2) if total > 0 else 0
    merged["summary"][checkpoint]["avg_time_per_q"] = round(
        merged["summary"][checkpoint]["total_time"] / total, 2
    ) if total > 0 else 0
    merged["summary"][checkpoint]["total_time"] = round(merged["summary"][checkpoint]["total_time"], 2)
    merged["summary"][checkpoint]["total_cost"] = round(merged["summary"][checkpoint]["total_cost"], 4)

# Save merged results
with open("$MERGED_FILE", 'w') as f:
    json.dump(merged, f, indent=2, default=str)

print(f"Merged {len(merged['questions'])} questions into $MERGED_FILE")
EOF

echo ""
echo "============================================================"
echo "GPQA UNIFIED EVALUATION COMPLETE"
echo "============================================================"
echo ""

# Print summary
python3 << EOF
import json
with open("$MERGED_FILE", 'r') as f:
    data = json.load(f)

summary = data.get("summary", {})
print("Results Summary:")
print("-" * 60)
for checkpoint in ["baseline", "ga3", "ga5"]:
    s = summary.get(checkpoint, {})
    print(f"{checkpoint.upper():10s}: {s.get('accuracy', 0):6.2f}% accuracy ({s.get('correct', 0)}/{s.get('total', 0)})")
    print(f"           Time: {s.get('total_time', 0):8.1f}s | Cost: \${s.get('total_cost', 0):.4f}")
    print()
EOF

echo "Results saved to: $MERGED_FILE"
echo "============================================================"
