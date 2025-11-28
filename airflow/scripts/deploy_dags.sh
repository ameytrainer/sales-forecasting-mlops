#!/bin/bash
# DAG Deployment Script
# Deploy DAGs to Airflow with validation

set -e

echo "📦 Deploying Airflow DAGs..."

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Configuration
AIRFLOW_HOME="${AIRFLOW_HOME:-$HOME/airflow}"
PROJECT_ROOT="/home/ubuntu/sales-forecasting-mlops"
DAG_SOURCE="$PROJECT_ROOT/airflow/dags"
DAG_DEST="$AIRFLOW_HOME/dags"

# Step 1: Validate DAG source directory
echo -e "\n${YELLOW}Step 1: Validating source directory...${NC}"
if [ ! -d "$DAG_SOURCE" ]; then
    echo -e "${RED}❌ DAG source directory not found: $DAG_SOURCE${NC}"
    exit 1
fi

DAG_COUNT=$(ls -1 "$DAG_SOURCE"/*.py 2>/dev/null | wc -l)
if [ "$DAG_COUNT" -eq 0 ]; then
    echo -e "${RED}❌ No DAG files found in $DAG_SOURCE${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Found $DAG_COUNT DAG files${NC}"

# Step 2: Create backup of existing DAGs
echo -e "\n${YELLOW}Step 2: Creating backup of existing DAGs...${NC}"
if [ -d "$DAG_DEST" ] && [ "$(ls -A $DAG_DEST)" ]; then
    BACKUP_DIR="$AIRFLOW_HOME/dags_backup_$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$BACKUP_DIR"
    cp -r "$DAG_DEST"/* "$BACKUP_DIR/"
    echo -e "${GREEN}✅ Backup created: $BACKUP_DIR${NC}"
else
    echo "ℹ️  No existing DAGs to backup"
fi

# Step 3: Validate DAG syntax
echo -e "\n${YELLOW}Step 3: Validating DAG syntax...${NC}"
VALIDATION_FAILED=0

for dag_file in "$DAG_SOURCE"/*.py; do
    dag_name=$(basename "$dag_file")
    echo "  Validating: $dag_name"
    
    if python3 -m py_compile "$dag_file"; then
        echo "    ✅ Syntax OK"
    else
        echo -e "    ${RED}❌ Syntax Error${NC}"
        VALIDATION_FAILED=1
    fi
done

if [ $VALIDATION_FAILED -eq 1 ]; then
    echo -e "${RED}❌ Validation failed. Deployment aborted.${NC}"
    exit 1
fi

echo -e "${GREEN}✅ All DAGs passed syntax validation${NC}"

# Step 4: Test DAG imports
echo -e "\n${YELLOW}Step 4: Testing DAG imports...${NC}"
export AIRFLOW_HOME="$AIRFLOW_HOME"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

IMPORT_FAILED=0
for dag_file in "$DAG_SOURCE"/*.py; do
    dag_name=$(basename "$dag_file" .py)
    echo "  Importing: $dag_name"
    
    if python3 -c "import sys; sys.path.insert(0, '$DAG_SOURCE'); import $dag_name" 2>/dev/null; then
        echo "    ✅ Import OK"
    else
        echo -e "    ${RED}❌ Import Failed${NC}"
        python3 -c "import sys; sys.path.insert(0, '$DAG_SOURCE'); import $dag_name" 2>&1 || true
        IMPORT_FAILED=1
    fi
done

if [ $IMPORT_FAILED -eq 1 ]; then
    echo -e "${RED}❌ Import testing failed. Deployment aborted.${NC}"
    exit 1
fi

echo -e "${GREEN}✅ All DAGs imported successfully${NC}"

# Step 5: Copy DAGs to Airflow
echo -e "\n${YELLOW}Step 5: Copying DAGs to Airflow...${NC}"
mkdir -p "$DAG_DEST"
cp -v "$DAG_SOURCE"/*.py "$DAG_DEST/"
echo -e "${GREEN}✅ DAGs copied to $DAG_DEST${NC}"

# Step 6: Copy plugins if they exist
echo -e "\n${YELLOW}Step 6: Checking for plugins...${NC}"
if [ -d "$PROJECT_ROOT/airflow/plugins" ]; then
    mkdir -p "$AIRFLOW_HOME/plugins"
    cp -rv "$PROJECT_ROOT/airflow/plugins/"* "$AIRFLOW_HOME/plugins/" 2>/dev/null || true
    echo -e "${GREEN}✅ Plugins copied${NC}"
else
    echo "ℹ️  No plugins to copy"
fi

# Step 7: Refresh Airflow DAGs
echo -e "\n${YELLOW}Step 7: Refreshing Airflow DAGs...${NC}"
if pgrep -f "airflow scheduler" > /dev/null; then
    echo "Scheduler is running, DAGs will be picked up automatically"
    # Optionally pause and unpause DAGs to force refresh
    # airflow dags pause ml_training_pipeline
    # airflow dags unpause ml_training_pipeline
else
    echo "⚠️  Scheduler not running. Start it with: airflow scheduler"
fi

# Step 8: List deployed DAGs
echo -e "\n${YELLOW}Step 8: Verifying deployed DAGs...${NC}"
sleep 3  # Wait for Airflow to scan
airflow dags list 2>/dev/null || echo "Run 'airflow dags list' to see deployed DAGs"

# Step 9: Check for errors
echo -e "\n${YELLOW}Step 9: Checking for DAG errors...${NC}"
if command -v airflow &> /dev/null; then
    airflow dags list-import-errors 2>/dev/null || true
fi

# Summary
echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}✅ DAG Deployment Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Deployed DAGs:"
ls -1 "$DAG_DEST"/*.py 2>/dev/null || echo "  (none)"
echo ""
echo "Next steps:"
echo "1. Check Airflow UI: http://localhost:8080"
echo "2. Verify DAGs appear in the list"
echo "3. Unpause DAGs to enable them"
echo "4. Monitor the first run in Airflow logs"
echo ""
echo "Useful commands:"
echo "  airflow dags list                    # List all DAGs"
echo "  airflow dags list-runs -d <dag_id>  # Show DAG runs"
echo "  airflow dags trigger <dag_id>       # Manual trigger"
echo "  airflow dags pause <dag_id>         # Pause DAG"
echo "  airflow dags unpause <dag_id>       # Unpause DAG"
echo ""

# Optional: Trigger a test run
read -p "Do you want to trigger ml_training_pipeline now? (y/N) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    airflow dags trigger ml_training_pipeline
    echo -e "${GREEN}✅ Pipeline triggered. Check Airflow UI for progress.${NC}"
fi
