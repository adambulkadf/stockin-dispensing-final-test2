"""
Product Archive Module - Archives dispensed items for historical tracking
"""

import json
import os
from datetime import datetime
from typing import Optional, Any

ARCHIVE_FILE = "data/dispensed_archive.json"


def load_archive() -> dict:
    """Load dispensed items archive"""
    try:
        if os.path.exists(ARCHIVE_FILE):
            with open(ARCHIVE_FILE, 'r') as f:
                return json.load(f)
    except Exception as e:
        print(f"Error loading archive: {e}")

    return {
        "dispensed_items": [],
        "summary": {
            "total_archived": 0,
            "last_updated": None
        }
    }


def save_archive(archive: dict):
    """Save dispensed items archive"""
    try:
        os.makedirs(os.path.dirname(ARCHIVE_FILE), exist_ok=True)
        with open(ARCHIVE_FILE, 'w') as f:
            json.dump(archive, f, indent=2, default=str)
    except Exception as e:
        print(f"Error saving archive: {e}")


def archive_dispensed_item(
    item_id: int,
    item: Any,
    vsu: Any,
    shelf_id: Optional[int],
    shelf_name: str,
    task_id: str
):
    """
    Archive a dispensed item for historical tracking

    Args:
        item_id: The item ID
        item: The Item object
        vsu: The VSU object where item was stored
        shelf_id: The shelf ID
        shelf_name: The shelf name
        task_id: The dispense task ID
    """
    archive = load_archive()

    archived_item = {
        "item_id": item_id,
        "product_id": item.metadata.product_id,
        "barcode": item.metadata.barcode,
        "batch": item.metadata.batch,
        "expiration": item.metadata.expiration.isoformat() if hasattr(item.metadata.expiration, 'isoformat') else str(item.metadata.expiration),
        "vsu_code": vsu.code if vsu else None,
        "shelf_id": shelf_id,
        "shelf_name": shelf_name,
        "stock_index": item.stock_index,
        "dispensed_at": datetime.now().isoformat(),
        "task_id": task_id
    }

    archive["dispensed_items"].append(archived_item)
    archive["summary"]["total_archived"] = len(archive["dispensed_items"])
    archive["summary"]["last_updated"] = datetime.now().isoformat()

    save_archive(archive)
    print(f"  Archived dispensed item {item_id} (barcode: {item.metadata.barcode})")
