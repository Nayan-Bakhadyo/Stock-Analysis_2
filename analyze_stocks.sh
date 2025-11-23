#!/bin/bash

# Stock Analysis Pipeline
# Usage: ./analyze_stocks.sh SYMBOL1 SYMBOL2 SYMBOL3 ...
# Example: ./analyze_stocks.sh IGI SPC NABIL

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

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

# Check if symbols provided
if [ $# -eq 0 ]; then
    echo -e "${RED}Error: No stock symbols provided${NC}"
    echo "Usage: ./analyze_stocks.sh SYMBOL1 SYMBOL2 SYMBOL3 ..."
    echo "Example: ./analyze_stocks.sh IGI SPC NABIL"
    exit 1
fi

SYMBOLS=("$@")
TOTAL=${#SYMBOLS[@]}
SUCCESS_COUNT=0
FAILED_SYMBOLS=()
OVERALL_START=$(date +%s)

echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║        Stock Analysis Pipeline - ${TOTAL} Stock(s)        ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Step 1: Grid Search for all symbols
echo -e "${YELLOW}[STEP 1/3] Running Grid Search to Find Best Configurations${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

STEP1_START=$(date +%s)
COMPLETED_GRIDSEARCH=0

for i in "${!SYMBOLS[@]}"; do
    SYMBOL="${SYMBOLS[$i]}"
    NUM=$((i + 1))
    
    TASK_START=$(date +%s)
    echo -e "${BLUE}[$NUM/$TOTAL] Grid Search: ${SYMBOL}${NC}"
    echo -e "${CYAN}⏳ Starting grid search (this may take 2-4 hours)...${NC}"
    
    if python3 stock_predictor.py "$SYMBOL"; then
        TASK_END=$(date +%s)
        TASK_DURATION=$((TASK_END - TASK_START))
        COMPLETED_GRIDSEARCH=$((COMPLETED_GRIDSEARCH + 1))
        
        # Calculate ETA for remaining grid searches
        if [ $COMPLETED_GRIDSEARCH -gt 0 ] && [ $NUM -lt $TOTAL ]; then
            STEP1_ELAPSED=$((TASK_END - STEP1_START))
            AVG_TIME=$((STEP1_ELAPSED / COMPLETED_GRIDSEARCH))
            REMAINING_GRIDSEARCH=$((TOTAL - COMPLETED_GRIDSEARCH))
            ETA=$((AVG_TIME * REMAINING_GRIDSEARCH))
            echo -e "${GREEN}✓ Grid search completed in $(format_time $TASK_DURATION)${NC}"
            echo -e "${CYAN}📊 Progress: $COMPLETED_GRIDSEARCH/$TOTAL | ETA for remaining: $(format_time $ETA)${NC}"
        else
            echo -e "${GREEN}✓ Grid search completed in $(format_time $TASK_DURATION)${NC}"
        fi
        echo ""
    else
        echo -e "${RED}✗ Grid search failed for ${SYMBOL}${NC}"
        FAILED_SYMBOLS+=("$SYMBOL")
        echo ""
        continue
    fi
done

STEP1_END=$(date +%s)
STEP1_TOTAL=$((STEP1_END - STEP1_START))
echo -e "${CYAN}⏱️  Step 1 completed in $(format_time $STEP1_TOTAL)${NC}"
echo ""

# Step 2: Train and Predict for all symbols
echo -e "${YELLOW}[STEP 2/3] Training Final Models and Generating Predictions${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

STEP2_START=$(date +%s)
COMPLETED_TRAINING=0

for i in "${!SYMBOLS[@]}"; do
    SYMBOL="${SYMBOLS[$i]}"
    NUM=$((i + 1))
    
    # Skip if grid search failed
    if [[ " ${FAILED_SYMBOLS[@]} " =~ " ${SYMBOL} " ]]; then
        echo -e "${YELLOW}[$NUM/$TOTAL] Skipping ${SYMBOL} (grid search failed)${NC}"
        continue
    fi
    
    TASK_START=$(date +%s)
    echo -e "${BLUE}[$NUM/$TOTAL] Training & Predicting: ${SYMBOL}${NC}"
    echo -e "${CYAN}⏳ Training final model (typically 2-5 minutes)...${NC}"
    
    if python3 predict_single_config.py "$SYMBOL"; then
        TASK_END=$(date +%s)
        TASK_DURATION=$((TASK_END - TASK_START))
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
        COMPLETED_TRAINING=$((COMPLETED_TRAINING + 1))
        
        # Calculate ETA for remaining training
        VALID_REMAINING=$((TOTAL - NUM - ${#FAILED_SYMBOLS[@]}))
        if [ $COMPLETED_TRAINING -gt 0 ] && [ $VALID_REMAINING -gt 0 ]; then
            STEP2_ELAPSED=$((TASK_END - STEP2_START))
            AVG_TIME=$((STEP2_ELAPSED / COMPLETED_TRAINING))
            ETA=$((AVG_TIME * VALID_REMAINING))
            echo -e "${GREEN}✓ Predictions generated in $(format_time $TASK_DURATION)${NC}"
            echo -e "${CYAN}📊 Progress: $COMPLETED_TRAINING/$((TOTAL - ${#FAILED_SYMBOLS[@]})) | ETA for remaining: $(format_time $ETA)${NC}"
        else
            echo -e "${GREEN}✓ Predictions generated in $(format_time $TASK_DURATION)${NC}"
        fi
        echo ""
    else
        echo -e "${RED}✗ Prediction failed for ${SYMBOL}${NC}"
        FAILED_SYMBOLS+=("$SYMBOL")
        echo ""
    fi
done

STEP2_END=$(date +%s)
STEP2_TOTAL=$((STEP2_END - STEP2_START))
echo -e "${CYAN}⏱️  Step 2 completed in $(format_time $STEP2_TOTAL)${NC}"
echo ""

# Step 3: Generate Website
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

echo ""
echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║                   ANALYSIS COMPLETE                        ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${GREEN}Successfully analyzed: ${SUCCESS_COUNT}/${TOTAL} stocks${NC}"
echo -e "${CYAN}⏱️  Total time: $(format_time $OVERALL_TOTAL)${NC}"
echo ""
echo -e "${CYAN}Time breakdown:${NC}"
echo -e "  • Grid Search: $(format_time $STEP1_TOTAL)"
echo -e "  • Training & Prediction: $(format_time $STEP2_TOTAL)"
echo -e "  • Website Generation: $(format_time $STEP3_TOTAL)"
echo ""

if [ ${#FAILED_SYMBOLS[@]} -gt 0 ]; then
    echo -e "${RED}Failed stocks: ${FAILED_SYMBOLS[*]}${NC}"
    echo ""
fi

echo -e "${YELLOW}→ Open stock_website/index.html to view results${NC}"
echo ""
