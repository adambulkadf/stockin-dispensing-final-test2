#!/bin/bash
# Script to check if archiving happened

echo "=========================================="
echo "ARCHIVING STATUS CHECK"
echo "=========================================="
echo ""

# Check scheduler status
echo "1. Scheduler Status:"
curl -s http://localhost:8000/scheduler/status | python3 -m json.tool
echo ""

# Check active log size
echo "2. Active Log Size:"
ls -lh dispense_logs.json
echo ""

# Count daily records in active log
echo "3. Daily Records in Active Log:"
python3 -c "
import json
with open('dispense_logs.json', 'r') as f:
    logs = json.load(f)
total = sum(len(p.get('daily_dispenses', {})) for p in logs['products'].values())
print(f'  Total daily records: {total}')
for pid, pdata in sorted(logs['products'].items(), key=lambda x: int(x[0]) if x[0].isdigit() else 999):
    print(f'  Product {pid}: {len(pdata.get(\"daily_dispenses\", {}))} records')
"
echo ""

# Check history folder
echo "4. History Folder Contents:"
if [ -d "history" ]; then
    find history -name "*.json" -exec sh -c 'echo "  {}" && wc -l < "{}"' \;
else
    echo "  No history folder yet"
fi
echo ""

# Check oldest date in active log
echo "5. Oldest Record in Active Log:"
python3 -c "
import json
from datetime import datetime
with open('dispense_logs.json', 'r') as f:
    logs = json.load(f)
all_dates = []
for p in logs['products'].values():
    all_dates.extend(p.get('daily_dispenses', {}).keys())
if all_dates:
    oldest = min(all_dates)
    age = (datetime.now() - datetime.fromisoformat(oldest)).days
    print(f'  Oldest: {oldest} ({age} days old)')
else:
    print('  No records')
"
echo ""

echo "=========================================="
echo "If archiving happened, you should see:"
echo "  - History folder with monthly files"
echo "  - Active log oldest record < 30 days"
echo "  - Scheduler last run time updated"
echo "=========================================="
