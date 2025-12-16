#!/usr/bin/env python3
"""
Helper script untuk upload model ke QuantConnect.
Generate commands dan instructions untuk upload.
"""

import os
import json
from datetime import datetime
from pathlib import Path

def generate_upload_instructions():
    """Generate upload instructions for QuantConnect."""

    # Check staging directory
    staging_dir = Path('./qc_staging')
    if not staging_dir.exists():
        print("❌ Staging directory not found. Run integration_bridge.py --sync first")
        return

    # Check required files
    required_files = ['latest_model.joblib', 'model_metadata.json', 'dataset_summary.txt']
    missing_files = []

    for file in required_files:
        if not (staging_dir / file).exists():
            missing_files.append(file)

    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return

    print("🚀 **QUANTCONNECT UPLOAD INSTRUCTIONS**")
    print("=" * 50)

    # Get model info
    metadata_file = staging_dir / 'model_metadata.json'
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)

    print(f"📊 Model Info:")
    print(f"   Version: {metadata.get('created_at', 'Unknown')}")
    print(f"   Features: {metadata.get('n_features', 'Unknown')}")
    print(f"   AUC: {metadata.get('performance', {}).get('latest_auc', 'Unknown')}")
    print()

    print("🌐 **Method 1: Web Upload**")
    print("1. Login to https://quantconnect.com")
    print("2. Open your project")
    print("3. Click 'Files' tab")
    print("4. Click 'Create New File'")
    print("5. Upload these 3 files:")
    for file in required_files:
        file_path = staging_dir / file
        size_mb = file_path.stat().st_size / (1024 * 1024)
        print(f"   - {file} ({size_mb:.1f}MB)")
    print()

    print("💻 **Method 2: CLI Upload**")
    print("1. Install QC CLI: pip install quantconnect")
    print("2. Set your project ID:")

    # Get project ID from config
    config_file = Path('qc_integration_config.json')
    project_id = "YOUR_PROJECT_ID"
    if config_file.exists():
        with open(config_file, 'r') as f:
            config = json.load(f)
        project_id = config.get('project_id', 'YOUR_PROJECT_ID')

    print(f"   export QC_PROJECT_ID={project_id}")
    print("3. Run these commands:")
    for file in required_files:
        print(f"   qc upload --project {project_id} ./qc_staging/{file}")
    print()

    print("✅ **After Upload**")
    print("1. In QuantConnect, open your algorithm file")
    print("2. Update to use XGBoostTradingAlgorithm_RealtimeSync.py")
    print("3. Run backtest to verify")
    print("4. Deploy to live trading")
    print()

    print("📁 **Staging Location:**")
    print(f"   {staging_dir.absolute()}")

    # Generate shell script for easy upload
    script_content = f"""#!/bin/bash
# QuantConnect Upload Script
# Generated: {datetime.now()}

PROJECT_ID="{project_id}"
STAGING_DIR="./qc_staging"

echo "🚀 Uploading to QuantConnect Project: $PROJECT_ID"

# Check if QC CLI is installed
if ! command -v qc &> /dev/null; then
    echo "❌ QuantConnect CLI not found. Install with: pip install quantconnect"
    exit 1
fi

# Upload files
for file in latest_model.joblib model_metadata.json dataset_summary.txt; do
    if [ -f "$STAGING_DIR/$file" ]; then
        echo "📤 Uploading $file..."
        qc upload --project "$PROJECT_ID" "$STAGING_DIR/$file"
    else
        echo "❌ File not found: $file"
    fi
done

echo "✅ Upload complete!"
echo "📋 Next steps:"
echo "   1. Open QuantConnect project"
echo "   2. Update algorithm to XGBoostTradingAlgorithm_RealtimeSync.py"
echo "   3. Run backtest"
echo "   4. Deploy to live"
"""

    script_path = Path('./upload_to_qc.sh')
    with open(script_path, 'w') as f:
        f.write(script_content)

    os.chmod(script_path, 0o755)
    print(f"\n🔧 **Generated upload script:** {script_path.absolute()}")
    print("   Run: ./upload_to_qc.sh")

if __name__ == "__main__":
    generate_upload_instructions()