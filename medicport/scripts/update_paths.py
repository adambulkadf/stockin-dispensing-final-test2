"""
Update file paths in code after reorganization
"""
import numpy 
import re

# Files to update
files_to_update = [
    "main.py",
    "dispensing.py",
    "stockin_logging.py",
    "scheduler.py"
]

# Path mappings (old -> new)
path_replacements = {
    '"ml_robot_updated.json"': '"data/ml_robot_updated.json"',
    "'ml_robot_updated.json'": "'data/ml_robot_updated.json'",
    '"ml_robot.json"': '"data/ml_robot.json"',
    "'ml_robot.json'": "'data/ml_robot.json'",
    '"robot_post.json"': '"data/robot_post.json"',
    "'robot_post.json'": "'data/robot_post.json'",
    '"warehouse_layout.json"': '"data/warehouse_layout.json"',
    "'warehouse_layout.json'": "'data/warehouse_layout.json'",
    '"weights.json"': '"data/weights.json"',
    "'weights.json'": "'data/weights.json'",
    '"dispense_logs.json"': '"data/dispense_logs.json"',
    "'dispense_logs.json'": "'data/dispense_logs.json'",
    '"stockin_logs.json"': '"data/stockin_logs.json"',
    "'stockin_logs.json'": "'data/stockin_logs.json'",
    '"incoming.json"': '"data/incoming.json"',
    "'incoming.json'": "'data/incoming.json'",
    'HISTORY_FOLDER = "history"': 'HISTORY_FOLDER = "data/history"',
    "HISTORY_FOLDER = 'history'": "HISTORY_FOLDER = 'data/history'",
}

def update_file(filename):
    """Update paths in a file"""
    try:
        with open(filename, 'r') as f:
            content = f.read()

        original_content = content
        changes = []

        # Apply replacements
        for old_path, new_path in path_replacements.items():
            if old_path in content:
                count = content.count(old_path)
                content = content.replace(old_path, new_path)
                changes.append(f"  {old_path} -> {new_path} ({count}x)")

        # Only write if changes were made
        if content != original_content:
            with open(filename, 'w') as f:
                f.write(content)
            print(f"Updated {filename}:")
            for change in changes:
                print(change)
            return True
        else:
            print(f"  {filename}: No changes needed")
            return False

    except Exception as e:
        print(f"Error updating {filename}: {e}")
        return False

def main():
    print("="*60)
    print("UPDATING FILE PATHS AFTER REORGANIZATION")
    print("="*60)
    print()

    total_updated = 0
    for filename in files_to_update:
        if update_file(filename):
            total_updated += 1
        print()

    print("="*60)
    print(f"SUMMARY: Updated {total_updated}/{len(files_to_update)} files")
    print("="*60)

if __name__ == "__main__":
    main()
