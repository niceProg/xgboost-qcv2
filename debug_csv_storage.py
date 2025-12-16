#!/usr/bin/env python3
"""
Debug CSV storage issues
"""

import os
import pandas as pd
from pathlib import Path

output_dir = Path('./output_train')

# Check each CSV file
print("=== Checking CSV Files ===")

csv_files = [
    ('feature_importance_20251216_184037.csv', 'Feature Importance'),
    ('trades.csv', 'Trades'),
    ('rekening_koran.csv', 'Account Statement - Equity'),
    ('rekening_koran_cash.csv', 'Account Statement - Cash')
]

for csv_file, description in csv_files:
    file_path = output_dir / csv_file
    print(f"\n{description}: {csv_file}")

    if file_path.exists():
        df = pd.read_csv(file_path)
        print(f"  Rows: {len(df)}")
        print(f"  Columns: {list(df.columns)}")
        print(f"  Sample data:")
        print(df.head(2))
    else:
        print("  File not found!")