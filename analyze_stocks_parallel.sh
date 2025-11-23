#!/usr/bin/env zsh

# Parallel Stock Analysis Pipeline
# Usage: ./analyze_stocks_parallel.sh [--jobs N] SYMBOL1 SYMBOL2 SYMBOL3 ...
# Example: ./analyze_stocks_parallel.sh --jobs 3 IGI SPC NABIL HBL ADBL

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m' # No Color

# Default number of parallel jobs
MAX_JOBS=2

# Function to format time
format_time() {
    local seconds=$1
    if [ $seconds -lt 60 ]; then
        echo "${seconds}s"
    elif [ $seconds -lt 3600 ]; then
        echo "$((seconds / 60))m $((seconds % 60))s"
    else
        echo "$((seconds / 3600))h $(((seconds % 3600) / 60))m"
    fi
}

# Parse arguments
SYMBOLS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --jobs|-j)
            MAX_JOBS="$2"
            shift 2
            ;;
        *)
            SYMBOLS+=("$1")
            shift
            ;;
    esac
done

# Validate inputs
if [ ${#SYMBOLS[@]} -eq 0 ]; then
    echo -e "${RED}Error: No stock symbols provided${NC}"
    echo "Usage: ./analyze_stocks_parallel.sh [--jobs N] SYMBOL1 SYMBOL2 SYMBOL3 ..."
    echo "Example: ./analyze_stocks_parallel.sh --jobs 3 IGI SPC NABIL HBL ADBL"
    echo ""
    echo "Options:"
    echo "  --jobs N, -j N    Number of parallel jobs (default: 2, recommended: 2-3)"
    exit 1
fi

if [ $MAX_JOBS -lt 1 ]; then
    echo -e "${RED}Error: --jobs must be at least 1${NC}"
    exit 1
fi

if [ $MAX_JOBS -gt 4 ]; then
    echo -e "${YELLOW}Warning: Running more than 4 parallel jobs may cause memory issues${NC}"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

TOTAL=${#SYMBOLS[@]}
OVERALL_START=$(date +%s)

# Create logs directory
mkdir -p logs

echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║     Parallel Stock Analysis - ${TOTAL} Stock(s) - ${MAX_JOBS} Jobs     ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${CYAN}💡 Tip: Monitor system resources with 'Stats' app or 'htop'${NC}"
echo ""

# Arrays to track status
typeset -A GRID_SEARCH_STATUS
typeset -A GRID_SEARCH_PID
typeset -A GRID_SEARCH_START
typeset -A PREDICT_STATUS
typeset -A PREDICT_PID
typeset -A PREDICT_START

# Initialize all statuses
for SYMBOL in "${SYMBOLS[@]}"; do
    GRID_SEARCH_STATUS[$SYMBOL]="pending"
    PREDICT_STATUS[$SYMBOL]="pending"
done

# Function to run grid search for a symbol
run_grid_search() {
    local SYMBOL=$1
    local LOG_FILE="logs/${SYMBOL}_grid_search.log"
    
    GRID_SEARCH_START[$SYMBOL]=$(date +%s)
    
    if python3 stock_predictor.py "$SYMBOL" > "$LOG_FILE" 2>&1; then
        GRID_SEARCH_STATUS[$SYMBOL]="success"
        local END_TIME=$(date +%s)
        local DURATION=$((END_TIME - GRID_SEARCH_START[$SYMBOL]))
        echo -e "${GREEN}✓ Grid search completed for ${SYMBOL} in $(format_time $DURATION)${NC}" | tee -a "$LOG_FILE"
    else
        GRID_SEARCH_STATUS[$SYMBOL]="failed"
        echo -e "${RED}✗ Grid search failed for ${SYMBOL}${NC}" | tee -a "$LOG_FILE"
    fi
}

# Function to run prediction for a symbol
run_prediction() {
    local SYMBOL=$1
    local LOG_FILE="logs/${SYMBOL}_predict.log"
    
    PREDICT_START[$SYMBOL]=$(date +%s)
    
    if python3 predict_single_config.py "$SYMBOL" > "$LOG_FILE" 2>&1; then
        PREDICT_STATUS[$SYMBOL]="success"
        local END_TIME=$(date +%s)
        local DURATION=$((END_TIME - PREDICT_START[$SYMBOL]))
        echo -e "${GREEN}✓ Prediction completed for ${SYMBOL} in $(format_time $DURATION)${NC}" | tee -a "$LOG_FILE"
    else
        PREDICT_STATUS[$SYMBOL]="failed"
        echo -e "${RED}✗ Prediction failed for ${SYMBOL}${NC}" | tee -a "$LOG_FILE"
    fi
}

# Function to check running jobs
count_running_jobs() {
    local count=0
    for pid in ${(k)GRID_SEARCH_PID}; do
        if kill -0 ${GRID_SEARCH_PID[$pid]} 2>/dev/null; then
            count=$((count + 1))
        fi
    done
    for pid in ${(k)PREDICT_PID}; do
        if kill -0 ${PREDICT_PID[$pid]} 2>/dev/null; then
            count=$((count + 1))
        fi
    done
    echo $count
}

# Function to show status
show_status() {
    local completed=0
    local running=0
    local failed=0
    local pending=0
    
    echo -e "\n${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${CYAN}Current Status:${NC}"
    for SYMBOL in "${SYMBOLS[@]}"; do
        local gs_status="${GRID_SEARCH_STATUS[$SYMBOL]}"
        local pred_status="${PREDICT_STATUS[$SYMBOL]}"
        
        if [ "$gs_status" = "success" ] && [ "$pred_status" = "success" ]; then
            echo -e "  ${GREEN}✓${NC} ${SYMBOL}: Grid Search ✓ | Prediction ✓"
            completed=$((completed + 1))
        elif [ "$gs_status" = "failed" ] || [ "$pred_status" = "failed" ]; then
            echo -e "  ${RED}✗${NC} ${SYMBOL}: Grid Search ${gs_status} | Prediction ${pred_status}"
            failed=$((failed + 1))
        elif [ "$gs_status" = "running" ] || [ "$pred_status" = "running" ]; then
            echo -e "  ${YELLOW}⏳${NC} ${SYMBOL}: Grid Search ${gs_status} | Prediction ${pred_status}"
            running=$((running + 1))
        else
            echo -e "  ${BLUE}○${NC} ${SYMBOL}: Grid Search ${gs_status} | Prediction ${pred_status}"
            pending=$((pending + 1))
        fi
    done
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${CYAN}Summary: ${GREEN}${completed} completed${NC} | ${YELLOW}${running} running${NC} | ${BLUE}${pending} pending${NC} | ${RED}${failed} failed${NC}"
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n"
}

# STEP 1: Grid Search with parallel execution
echo -e "${YELLOW}[STEP 1/3] Running Grid Search (Parallel: ${MAX_JOBS} jobs)${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

STEP1_START=$(date +%s)

for SYMBOL in "${SYMBOLS[@]}"; do
    # Wait if max jobs reached
    while [[ $(count_running_jobs) -ge $MAX_JOBS ]]; do
        sleep 2
    done
    
    echo -e "${MAGENTA}🚀 Starting grid search: ${SYMBOL}${NC}"
    echo -e "${CYAN}   Log: logs/${SYMBOL}_grid_search.log${NC}"
    
    GRID_SEARCH_STATUS[$SYMBOL]="running"
    run_grid_search "$SYMBOL" &
    GRID_SEARCH_PID[$SYMBOL]=$!
    
    sleep 1  # Stagger starts slightly
done

# Wait for all grid searches to complete
echo -e "\n${CYAN}Waiting for all grid searches to complete...${NC}"
for SYMBOL in "${SYMBOLS[@]}"; do
    if [ -n "${GRID_SEARCH_PID[$SYMBOL]}" ]; then
        wait ${GRID_SEARCH_PID[$SYMBOL]} 2>/dev/null || true
    fi
done

STEP1_END=$(date +%s)
STEP1_TOTAL=$((STEP1_END - STEP1_START))

show_status

echo -e "${CYAN}⏱️  Step 1 completed in $(format_time $STEP1_TOTAL)${NC}"
echo ""

# STEP 2: Predictions with parallel execution
echo -e "${YELLOW}[STEP 2/3] Training & Predicting (Parallel: ${MAX_JOBS} jobs)${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

STEP2_START=$(date +%s)

for SYMBOL in "${SYMBOLS[@]}"; do
    # Skip if grid search failed
    if [ "${GRID_SEARCH_STATUS[$SYMBOL]}" != "success" ]; then
        echo -e "${YELLOW}⊘ Skipping ${SYMBOL} (grid search ${GRID_SEARCH_STATUS[$SYMBOL]})${NC}"
        PREDICT_STATUS[$SYMBOL]="skipped"
        continue
    fi
    
    # Wait if max jobs reached
    while [[ $(count_running_jobs) -ge $MAX_JOBS ]]; do
        sleep 2
    done
    
    echo -e "${MAGENTA}🚀 Starting prediction: ${SYMBOL}${NC}"
    echo -e "${CYAN}   Log: logs/${SYMBOL}_predict.log${NC}"
    
    PREDICT_STATUS[$SYMBOL]="running"
    run_prediction "$SYMBOL" &
    PREDICT_PID[$SYMBOL]=$!
    
    sleep 1  # Stagger starts slightly
done

# Wait for all predictions to complete
echo -e "\n${CYAN}Waiting for all predictions to complete...${NC}"
for SYMBOL in "${SYMBOLS[@]}"; do
    if [ -n "${PREDICT_PID[$SYMBOL]}" ]; then
        wait ${PREDICT_PID[$SYMBOL]} 2>/dev/null || true
    fi
done

STEP2_END=$(date +%s)
STEP2_TOTAL=$((STEP2_END - STEP2_START))

show_status

echo -e "${CYAN}⏱️  Step 2 completed in $(format_time $STEP2_TOTAL)${NC}"
echo ""

# STEP 3: Generate Website
echo -e "${YELLOW}[STEP 3/3] Generating Website${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

STEP3_START=$(date +%s)
echo -e "${CYAN}⏳ Generating website...${NC}"

if python3 generate_website.py; then
    STEP3_END=$(date +%s)
    STEP3_TOTAL=$((STEP3_END - STEP3_START))
    echo -e "${GREEN}✓ Website generated in $(format_time $STEP3_TOTAL)${NC}"
else
    echo -e "${RED}✗ Website generation failed${NC}"
    exit 1
fi

OVERALL_END=$(date +%s)
OVERALL_TOTAL=$((OVERALL_END - OVERALL_START))

# Final summary
SUCCESS_COUNT=0
FAILED_SYMBOLS=()
for SYMBOL in "${SYMBOLS[@]}"; do
    if [ "${PREDICT_STATUS[$SYMBOL]}" = "success" ]; then
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    else
        FAILED_SYMBOLS+=("$SYMBOL")
    fi
done

echo ""
echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║                   ANALYSIS COMPLETE                        ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${GREEN}Successfully analyzed: ${SUCCESS_COUNT}/${TOTAL} stocks${NC}"
echo -e "${CYAN}⏱️  Total time: $(format_time $OVERALL_TOTAL)${NC}"
echo ""
echo -e "${CYAN}Time breakdown:${NC}"
echo -e "  • Grid Search (parallel): $(format_time $STEP1_TOTAL)"
echo -e "  • Training & Prediction (parallel): $(format_time $STEP2_TOTAL)"
echo -e "  • Website Generation: $(format_time $STEP3_TOTAL)"
echo ""
echo -e "${MAGENTA}⚡ Speedup vs sequential: ~${TOTAL}x faster with ${MAX_JOBS} parallel jobs${NC}"
echo ""

if [ ${#FAILED_SYMBOLS[@]} -gt 0 ]; then
    echo -e "${RED}Failed stocks: ${FAILED_SYMBOLS[*]}${NC}"
    echo -e "${YELLOW}Check logs in logs/ directory for details${NC}"
    echo ""
fi

echo -e "${YELLOW}→ Open stock_website/index.html to view results${NC}"
echo -e "${CYAN}→ View logs in logs/ directory${NC}"
echo ""
