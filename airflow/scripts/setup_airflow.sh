#!/bin/bash
# Airflow Setup Script
# Sets up Airflow 3.1.3 for production use

set -e

echo "🚀 Setting up Apache Airflow 3.1.3..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
AIRFLOW_HOME="${AIRFLOW_HOME:-$HOME/airflow}"
PROJECT_ROOT="/home/ubuntu/sales-forecasting-mlops"

echo -e "${GREEN}AIRFLOW_HOME: $AIRFLOW_HOME${NC}"

# Step 1: Create Airflow directories
echo -e "\n${YELLOW}Step 1: Creating Airflow directories...${NC}"
mkdir -p "$AIRFLOW_HOME"/{dags,logs,plugins,config}
echo "✅ Directories created"

# Step 2: Install Airflow (if not already installed)
echo -e "\n${YELLOW}Step 2: Checking Airflow installation...${NC}"
if ! command -v airflow &> /dev/null; then
    echo "Installing Airflow 3.1.3..."
    pip install apache-airflow==3.1.3 --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-3.1.3/constraints-3.12.txt"
else
    AIRFLOW_VERSION=$(airflow version)
    echo "✅ Airflow already installed: $AIRFLOW_VERSION"
fi

# Step 3: Set environment variables
echo -e "\n${YELLOW}Step 3: Setting environment variables...${NC}"
export AIRFLOW_HOME="$AIRFLOW_HOME"
export AIRFLOW__CORE__DAGS_FOLDER="$AIRFLOW_HOME/dags"
export AIRFLOW__CORE__LOAD_EXAMPLES="False"

echo "✅ Environment variables set"

# Step 4: Copy configuration
echo -e "\n${YELLOW}Step 4: Configuring Airflow...${NC}"
if [ -f "$PROJECT_ROOT/airflow/config/airflow.cfg.template" ]; then
    cp "$PROJECT_ROOT/airflow/config/airflow.cfg.template" "$AIRFLOW_HOME/airflow.cfg"
    echo "✅ Configuration file copied"
else
    echo "⚠️  No configuration template found, using defaults"
fi

# Step 5: Initialize database
echo -e "\n${YELLOW}Step 5: Initializing Airflow database...${NC}"
airflow db migrate
echo "✅ Database initialized"

# Step 6: Create admin user
echo -e "\n${YELLOW}Step 6: Creating admin user...${NC}"
if ! airflow users list | grep -q "admin"; then
    airflow users create \
        --username admin \
        --password admin123 \
        --firstname Admin \
        --lastname User \
        --role Admin \
        --email ameytalkatkar169@gmail.com
    echo "✅ Admin user created (username: admin, password: admin123)"
else
    echo "ℹ️  Admin user already exists"
fi

# Step 7: Copy DAGs
echo -e "\n${YELLOW}Step 7: Copying DAGs to Airflow...${NC}"
if [ -d "$PROJECT_ROOT/airflow/dags" ]; then
    cp -r "$PROJECT_ROOT/airflow/dags/"* "$AIRFLOW_HOME/dags/"
    echo "✅ DAGs copied: $(ls -1 $AIRFLOW_HOME/dags/*.py | wc -l) files"
else
    echo "⚠️  No DAGs found in project"
fi

# Step 8: Copy plugins
echo -e "\n${YELLOW}Step 8: Copying plugins...${NC}"
if [ -d "$PROJECT_ROOT/airflow/plugins" ]; then
    cp -r "$PROJECT_ROOT/airflow/plugins/"* "$AIRFLOW_HOME/plugins/"
    echo "✅ Plugins copied"
else
    echo "ℹ️  No plugins found"
fi

# Step 9: Import connections
echo -e "\n${YELLOW}Step 9: Importing connections...${NC}"
if [ -f "$PROJECT_ROOT/airflow/config/connections.yaml" ]; then
    echo "⚠️  Please manually configure connections in connections.yaml, then run:"
    echo "    airflow connections import $PROJECT_ROOT/airflow/config/connections.yaml"
else
    echo "ℹ️  No connections file found"
fi

# Step 10: Import variables
echo -e "\n${YELLOW}Step 10: Importing variables...${NC}"
if [ -f "$PROJECT_ROOT/airflow/config/variables.json" ]; then
    airflow variables import "$PROJECT_ROOT/airflow/config/variables.json"
    echo "✅ Variables imported"
else
    echo "ℹ️  No variables file found"
fi

# Step 11: Test DAG imports
echo -e "\n${YELLOW}Step 11: Testing DAG imports...${NC}"
python3 << EOF
import sys
sys.path.insert(0, '$PROJECT_ROOT')
sys.path.insert(0, '$AIRFLOW_HOME/dags')

try:
    import ml_training_pipeline
    import data_ingestion_pipeline
    import batch_prediction_pipeline
    import model_monitoring_pipeline
    import retraining_trigger_pipeline
    print("✅ All DAGs imported successfully")
except Exception as e:
    print(f"❌ DAG import failed: {e}")
    sys.exit(1)
EOF

# Step 12: Create systemd services (optional)
echo -e "\n${YELLOW}Step 12: Creating systemd service files...${NC}"
cat > /tmp/airflow-webserver.service << 'SERVICEFILE'
[Unit]
Description=Airflow Webserver
After=network.target postgresql.service

[Service]
Type=simple
User=ubuntu
Group=ubuntu
Environment="PATH=/home/ubuntu/sales-forecasting-mlops/venv/bin:/usr/local/bin"
Environment="AIRFLOW_HOME=/home/ubuntu/airflow"
ExecStart=/home/ubuntu/sales-forecasting-mlops/venv/bin/airflow webserver --port 8080
Restart=on-failure
RestartSec=10s

[Install]
WantedBy=multi-user.target
SERVICEFILE

cat > /tmp/airflow-scheduler.service << 'SERVICEFILE'
[Unit]
Description=Airflow Scheduler
After=network.target postgresql.service airflow-webserver.service

[Service]
Type=simple
User=ubuntu
Group=ubuntu
Environment="PATH=/home/ubuntu/sales-forecasting-mlops/venv/bin:/usr/local/bin"
Environment="AIRFLOW_HOME=/home/ubuntu/airflow"
ExecStart=/home/ubuntu/sales-forecasting-mlops/venv/bin/airflow scheduler
Restart=on-failure
RestartSec=10s

[Install]
WantedBy=multi-user.target
SERVICEFILE

cat > /tmp/airflow-dag-processor.service << 'SERVICEFILE'
[Unit]
Description=Airflow DAG Processor
After=network.target postgresql.service

[Service]
Type=simple
User=ubuntu
Group=ubuntu
Environment="PATH=/home/ubuntu/sales-forecasting-mlops/venv/bin:/usr/local/bin"
Environment="AIRFLOW_HOME=/home/ubuntu/airflow"
ExecStart=/home/ubuntu/sales-forecasting-mlops/venv/bin/airflow dag-processor
Restart=on-failure
RestartSec=10s

[Install]
WantedBy=multi-user.target
SERVICEFILE

echo "✅ Service files created in /tmp/"
echo "   To install, run:"
echo "   sudo cp /tmp/airflow-*.service /etc/systemd/system/"
echo "   sudo systemctl daemon-reload"
echo "   sudo systemctl enable airflow-webserver airflow-scheduler airflow-dag-processor"
echo "   sudo systemctl start airflow-webserver airflow-scheduler airflow-dag-processor"

# Final summary
echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}✅ Airflow Setup Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Next steps:"
echo "1. Configure connections in: $PROJECT_ROOT/airflow/config/connections.yaml"
echo "2. Update variables if needed: $PROJECT_ROOT/airflow/config/variables.json"
echo "3. Start Airflow:"
echo "   airflow standalone"
echo "   OR use systemd services (see instructions above)"
echo ""
echo "4. Access Airflow UI:"
echo "   http://localhost:8080"
echo "   Username: admin"
echo "   Password: admin123"
echo ""
echo "5. Verify DAGs are loaded:"
echo "   airflow dags list"
echo ""
echo -e "${YELLOW}⚠️  Remember to:${NC}"
echo "   - Update passwords for production"
echo "   - Configure email/Slack notifications"
echo "   - Set up SSL for web UI"
echo "   - Configure proper firewall rules"
echo ""
