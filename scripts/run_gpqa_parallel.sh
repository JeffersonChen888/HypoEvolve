#!/bin/bash
#
# GPQA Parallel Evaluation Script
# Runs GPQA evaluation in parallel batches for faster execution
#
# Usage:
#   ./scripts/run_gpqa_parallel.sh                    # Run baseline only (default)
#   ./scripts/run_gpqa_parallel.sh --all-methods      # Run all methods
#   ./scripts/run_gpqa_parallel.sh --workers 8        # Use 8 parallel workers
#

set -e

# Configuration
SCRIPT_DIR="/Users/jeffersonchen/programming/MixLab/DeepScientists/AI_Coscientist_Rep/scripts"
PROJECT_DIR="/Users/jeffersonchen/programming/MixLab/DeepScientists/AI_Coscientist_Rep"
PYTHON_SCRIPT="$SCRIPT_DIR/run_gpqa_ga_evaluation.py"

# Default settings
TOTAL_QUESTIONS=198
NUM_WORKERS=8
MODEL="gpt-5-mini"
RUN_BASELINE=true
RUN_GA3=false
RUN_GA5=true

# Output folder name (no timestamp)
RUN_NAME="gpqa_eval"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --workers)
            NUM_WORKERS="$2"
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
        --baseline-only)
            RUN_BASELINE=true
            RUN_GA3=false
            RUN_GA5=false
            shift
            ;;
        --baseline)
            RUN_BASELINE=true
            shift
            ;;
        --ga3)
            RUN_GA3=true
            shift
            ;;
        --ga5)
            RUN_GA5=true
            shift
            ;;
        --all-methods)
            RUN_BASELINE=true
            RUN_GA3=true
            RUN_GA5=true
            shift
            ;;
        --questions)
            TOTAL_QUESTIONS="$2"
            shift 2
            ;;
        --help|-h)
            echo "GPQA Parallel Evaluation Script"
            echo ""
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --workers N        Number of parallel workers (default: 4)"
            echo "  --model NAME       Model to use (default: gpt-5-mini)"
            echo "  --name NAME        Run name for output folder (default: gpqa_eval)"
            echo "  --baseline-only    Only run baseline evaluation"
            echo "  --baseline         Run baseline evaluation"
            echo "  --ga3              Run GA with 3 generations"
            echo "  --ga5              Run GA with 5 generations"
            echo "  --all-methods      Run all methods (baseline + GA3 + GA5)"
            echo "  --questions N      Total questions to evaluate (default: 198)"
            echo "  -h, --help         Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Build method flags
METHOD_FLAGS=""
if [ "$RUN_BASELINE" = true ]; then
    METHOD_FLAGS="$METHOD_FLAGS --baseline"
fi
if [ "$RUN_GA3" = true ]; then
    METHOD_FLAGS="$METHOD_FLAGS --ga3"
fi
if [ "$RUN_GA5" = true ]; then
    METHOD_FLAGS="$METHOD_FLAGS --ga5"
fi

# If no method specified, default to baseline only (fastest)
if [ -z "$METHOD_FLAGS" ]; then
    METHOD_FLAGS="--baseline"
    RUN_BASELINE=true
fi

# Calculate batch size
BATCH_SIZE=$((TOTAL_QUESTIONS / NUM_WORKERS))
if [ $((TOTAL_QUESTIONS % NUM_WORKERS)) -ne 0 ]; then
    BATCH_SIZE=$((BATCH_SIZE + 1))
fi

# Create output directory
OUTPUT_DIR="$PROJECT_DIR/output/$RUN_NAME"
mkdir -p "$OUTPUT_DIR/logs"
mkdir -p "$OUTPUT_DIR/results"

echo "============================================================"
echo "GPQA PARALLEL EVALUATION"
echo "============================================================"
echo "Model: $MODEL"
echo "Total questions: $TOTAL_QUESTIONS"
echo "Parallel workers: $NUM_WORKERS"
echo "Batch size: $BATCH_SIZE"
echo "Methods: $METHOD_FLAGS"
echo "Output directory: $OUTPUT_DIR"
echo "============================================================"
echo ""

cd "$PROJECT_DIR"

# Array to store PIDs
declare -a PIDS

# Start time
START_TIME=$(date +%s)

# Launch parallel workers
for ((i=0; i<NUM_WORKERS; i++)); do
    START_IDX=$((i * BATCH_SIZE))
    END_IDX=$(((i + 1) * BATCH_SIZE))
    
    # Clamp end index to total questions
    if [ $END_IDX -gt $TOTAL_QUESTIONS ]; then
        END_IDX=$TOTAL_QUESTIONS
    fi
    
    # Skip if start >= end
    if [ $START_IDX -ge $END_IDX ]; then
        continue
    fi
    
    WORKER_LOG="$OUTPUT_DIR/logs/worker_${i}.log"
    WORKER_RESULT="$OUTPUT_DIR/results/batch_${i}.json"
    
    echo "Starting worker $i: questions $START_IDX to $((END_IDX - 1))"
    
    # Launch worker in background
    python "$PYTHON_SCRIPT" \
        --start $START_IDX \
        --end $END_IDX \
        --model "$MODEL" \
        --output-file "$WORKER_RESULT" \
        $METHOD_FLAGS \
        > "$WORKER_LOG" 2>&1 &
    
    PIDS+=($!)
done

echo ""
echo "Launched ${#PIDS[@]} workers. Waiting for completion..."
echo ""

# Monitor progress
while true; do
    RUNNING=0
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            RUNNING=$((RUNNING + 1))
        fi
    done
    
    if [ $RUNNING -eq 0 ]; then
        break
    fi
    
    # Count completed results
    COMPLETED=$(ls -1 "$OUTPUT_DIR/results/"*.json 2>/dev/null | wc -l || echo 0)
    echo -ne "\rWorkers running: $RUNNING / ${#PIDS[@]} | Results completed: $COMPLETED    "
    sleep 5
done

echo ""
echo ""

# Wait for all workers to finish
declare -a STATUSES
for pid in "${PIDS[@]}"; do
    wait "$pid" 2>/dev/null || true
    STATUSES+=($?)
done

# End time
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo "============================================================"
echo "ALL WORKERS COMPLETED"
echo "============================================================"
echo "Total time: ${ELAPSED}s ($(echo "scale=2; $ELAPSED/60" | bc)m)"
echo ""

# Check worker statuses
FAILED=0
for i in "${!STATUSES[@]}"; do
    if [ "${STATUSES[$i]}" -ne 0 ]; then
        echo "Worker $i: FAILED (exit code ${STATUSES[$i]})"
        FAILED=$((FAILED + 1))
    else
        echo "Worker $i: SUCCESS"
    fi
done

echo ""

# Merge results
echo "============================================================"
echo "MERGING RESULTS"
echo "============================================================"

python3 << EOF
import json
from pathlib import Path

output_dir = Path("$OUTPUT_DIR")
results_dir = output_dir / "results"

# Find all batch result files
batch_files = sorted(results_dir.glob("batch_*.json"))
print(f"Found {len(batch_files)} batch files to merge")

if not batch_files:
    print("No result files found!")
    exit(1)

# Merge results
merged = {
    'metadata': {
        'model': '$MODEL',
        'total_questions': $TOTAL_QUESTIONS,
        'workers': $NUM_WORKERS,
        'elapsed_seconds': $ELAPSED
    },
    'questions': [],
    'summary': {
        'baseline': {'correct': 0, 'total': 0, 'accuracy': 0},
        'ga_3gen': {'correct': 0, 'total': 0, 'accuracy': 0},
        'ga_5gen': {'correct': 0, 'total': 0, 'accuracy': 0}
    }
}

for f in batch_files:
    try:
        with open(f, 'r') as fp:
            data = json.load(fp)
        
        if 'questions' in data:
            merged['questions'].extend(data['questions'])
            print(f"  {f.name}: {len(data['questions'])} questions")
    except Exception as e:
        print(f"  Error reading {f.name}: {e}")

# Sort by question index
merged['questions'].sort(key=lambda x: x.get('question_index', 0))

# Calculate summary
for approach in ['baseline', 'ga_3gen', 'ga_5gen']:
    correct = sum(1 for q in merged['questions'] if q.get(approach, {}).get('correct', False))
    total = sum(1 for q in merged['questions'] if q.get(approach, {}))
    
    if total > 0:
        merged['summary'][approach] = {
            'correct': correct,
            'total': total,
            'accuracy': round(correct / total * 100, 2),
            'total_time': sum(q.get(approach, {}).get('time_seconds', 0) for q in merged['questions']),
            'total_cost': sum(q.get(approach, {}).get('cost_usd', 0) for q in merged['questions'])
        }

# Save merged result
merged_file = output_dir / "results.json"
with open(merged_file, 'w') as f:
    json.dump(merged, f, indent=2)

print()
print(f"Merged results saved to: {merged_file}")
print(f"Total questions: {len(merged['questions'])}")
print()
print("=" * 60)
print("FINAL RESULTS")
print("=" * 60)
for approach, stats in merged['summary'].items():
    if stats.get('total', 0) > 0:
        print(f"{approach.upper():12} | Accuracy: {stats['accuracy']:5.1f}% ({stats['correct']}/{stats['total']})")
print("=" * 60)
EOF

echo ""
echo "============================================================"
echo "DONE!"
echo "============================================================"
echo "Results: $OUTPUT_DIR/results.json"
echo "Logs: $OUTPUT_DIR/logs/"
echo "Total time: ${ELAPSED}s"
