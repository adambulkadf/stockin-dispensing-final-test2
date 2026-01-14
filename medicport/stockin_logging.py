"""
STOCKIN LOGGING MODULE - MedicPort Sequential Stock System

Tracks all stock-in operations with:
- Product-level tracking (by product_id and barcode)
- Daily/weekly/monthly aggregations
- Complete history with timestamps
- Barcode and batch tracking
"""

from datetime import datetime
from typing import Dict
import json
import os

STOCKIN_LOG_FILE = "data/stockin_logs.json"


def get_week_number(date: datetime) -> str:
    """Get ISO week number in format YYYY-Www"""
    return f"{date.year}-W{date.isocalendar()[1]:02d}"


def get_month_string(date: datetime) -> str:
    """Get month string in format YYYY-MM"""
    return f"{date.year}-{date.month:02d}"


def load_stockin_logs() -> dict:
    """
    Load stock-in logs

    Structure:
    {
        "products": {
            "1": {
                "barcode": "XXX",
                "total_stocked": 15,
                "last_stocked": "2025-11-14T10:30:00",
                "daily_stockins": {
                    "2025-11-14": 5
                },
                "weekly_stockins": {
                    "2025-W46": 15
                },
                "monthly_stockins": {
                    "2025-11": 15
                },
                "history": [
                    {
                        "timestamp": "2025-11-14T10:30:00",
                        "barcode": "XXX",
                        "batch": "BATCH123",
                        "quantity": 1,
                        "vsu_code": "vu1",
                        "shelf": "rack1_1",
                        "coordinates": {"x": 100, "y": 200, "z": 50}
                    }
                ]
            }
        },
        "summary": {
            "total_stockins": 150,
            "last_updated": "2025-11-14T10:30:00",
            "current_week": "2025-W46",
            "current_month": "2025-11"
        }
    }
    """
    try:
        if os.path.exists(STOCKIN_LOG_FILE):
            with open(STOCKIN_LOG_FILE, 'r') as f:
                return json.load(f)
        else:
            return {
                "products": {},
                "summary": {
                    "total_stockins": 0,
                    "last_updated": datetime.now().isoformat(),
                    "current_week": get_week_number(datetime.now()),
                    "current_month": get_month_string(datetime.now())
                }
            }
    except Exception as e:
        print(f"Error loading stock-in logs: {e}")
        return {
            "products": {},
            "summary": {
                "total_stockins": 0,
                "last_updated": datetime.now().isoformat(),
                "current_week": get_week_number(datetime.now()),
                "current_month": get_month_string(datetime.now())
            }
        }


def save_stockin_logs(logs: dict):
    """Save stock-in logs to file"""
    try:
        with open(STOCKIN_LOG_FILE, 'w') as f:
            json.dump(logs, f, indent=2)
        print(f"Stock-in logs saved to {STOCKIN_LOG_FILE}")
    except Exception as e:
        print(f"Error saving stock-in logs: {e}")
        raise


def log_stockin(
    product_id: int,
    barcode: str,
    batch: str,
    vsu_code: str = None,
    shelf_name: str = None,
    coordinates: Dict = None
):
    """
    Log a stock-in operation

    Args:
        product_id: Product ID
        barcode: Product barcode
        batch: Batch number
        vsu_code: VSU code where item was placed
        shelf_name: Shelf name
        coordinates: {"x": 100, "y": 200, "z": 50}
    """
    logs = load_stockin_logs()

    product_key = str(product_id)
    current_date = datetime.now()
    date_str = current_date.date().isoformat()
    week_str = get_week_number(current_date)
    month_str = get_month_string(current_date)

    # Initialize product entry if doesn't exist
    if product_key not in logs["products"]:
        logs["products"][product_key] = {
            "barcode": barcode,
            "total_stocked": 0,
            "last_stocked": None,
            "daily_stockins": {},
            "weekly_stockins": {},
            "monthly_stockins": {},
            "history": []
        }

    product_log = logs["products"][product_key]

    # Update totals
    product_log["total_stocked"] += 1
    product_log["last_stocked"] = current_date.isoformat()

    # Update daily stockins
    if date_str not in product_log["daily_stockins"]:
        product_log["daily_stockins"][date_str] = 0
    product_log["daily_stockins"][date_str] += 1

    # Update weekly stockins
    if week_str not in product_log["weekly_stockins"]:
        product_log["weekly_stockins"][week_str] = 0
    product_log["weekly_stockins"][week_str] += 1

    # Update monthly stockins
    if month_str not in product_log["monthly_stockins"]:
        product_log["monthly_stockins"][month_str] = 0
    product_log["monthly_stockins"][month_str] += 1

    # Add to history
    history_entry = {
        "timestamp": current_date.isoformat(),
        "barcode": barcode,
        "batch": batch,
        "quantity": 1,
        "vsu_code": vsu_code,
        "shelf": shelf_name,
        "coordinates": coordinates or {}
    }
    product_log["history"].append(history_entry)

    # UNLIMITED HISTORY - No cap on history entries

    # Update summary
    logs["summary"]["total_stockins"] += 1
    logs["summary"]["last_updated"] = current_date.isoformat()
    logs["summary"]["current_week"] = week_str
    logs["summary"]["current_month"] = month_str

    # Save updated logs
    save_stockin_logs(logs)

    print(f"Stock-in logged: Product {product_id} (+1)")
    print(f"  Daily total: {product_log['daily_stockins'][date_str]}")
    print(f"  Weekly total ({week_str}): {product_log['weekly_stockins'][week_str]}")
    print(f"  Monthly total ({month_str}): {product_log['monthly_stockins'][month_str]}")
    print(f"  All-time total: {product_log['total_stocked']}")


def get_stockin_logs():
    """Get all stock-in logs"""
    logs = load_stockin_logs()
    return {
        "status": "success",
        "summary": {
            "total_products_tracked": len(logs["products"]),
            "total_stockins_all_time": logs["summary"]["total_stockins"],
            "last_updated": logs["summary"]["last_updated"]
        },
        "products": [
            {
                "product_id": int(pid),
                "barcode": data["barcode"],
                "total_stocked": data["total_stocked"],
                "last_stocked": data["last_stocked"]
            }
            for pid, data in sorted(
                logs["products"].items(),
                key=lambda x: x[1]["total_stocked"],
                reverse=True
            )
        ]
    }


def get_product_stockin_history(product_id: int):
    """Get stock-in history for a specific product"""
    logs = load_stockin_logs()
    product_key = str(product_id)

    if product_key not in logs["products"]:
        return {
            "status": "error",
            "message": f"No stock-in history for product {product_id}"
        }

    product_data = logs["products"][product_key]

    # Sort daily, weekly, monthly data
    daily_sorted = sorted(
        product_data["daily_stockins"].items(),
        key=lambda x: x[0],
        reverse=True
    )[:30]  # Last 30 days

    weekly_sorted = sorted(
        product_data["weekly_stockins"].items(),
        key=lambda x: x[0],
        reverse=True
    )[:12]  # Last 12 weeks

    monthly_sorted = sorted(
        product_data.get("monthly_stockins", {}).items(),
        key=lambda x: x[0],
        reverse=True
    )[:12]  # Last 12 months

    return {
        "status": "success",
        "product_id": product_id,
        "barcode": product_data["barcode"],
        "total_stocked": product_data["total_stocked"],
        "last_stocked": product_data["last_stocked"],
        "daily_stockins": [
            {"date": date, "quantity": qty}
            for date, qty in daily_sorted
        ],
        "weekly_stockins": [
            {"week": week, "quantity": qty}
            for week, qty in weekly_sorted
        ],
        "monthly_stockins": [
            {"month": month, "quantity": qty}
            for month, qty in monthly_sorted
        ],
        "recent_history": product_data["history"][-20:]  # Last 20 entries
    }


def reset_stockin_logs():
    """
    Reset stock-in logs by archiving current file and starting fresh

    Archives current stockin_logs.json to stockin_logs_archive_{timestamp}.json
    Creates new empty stockin_logs.json
    """
    try:
        # Check if current log file exists
        if os.path.exists(STOCKIN_LOG_FILE):
            # Load current logs
            current_logs = load_stockin_logs()

            # Create archive filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            archive_filename = f"stockin_logs_archive_{timestamp}.json"

            # Save to archive
            with open(archive_filename, 'w') as f:
                json.dump(current_logs, f, indent=2)

            print(f"Archived current logs to {archive_filename}")

        # Create fresh logs
        fresh_logs = {
            "products": {},
            "summary": {
                "total_stockins": 0,
                "last_updated": datetime.now().isoformat(),
                "current_week": get_week_number(datetime.now()),
                "current_month": get_month_string(datetime.now())
            }
        }

        # Save fresh logs
        save_stockin_logs(fresh_logs)

        return {
            "status": "success",
            "message": "Stock-in logs reset successfully",
            "archive_file": archive_filename if os.path.exists(STOCKIN_LOG_FILE) else None,
            "new_log_file": STOCKIN_LOG_FILE
        }

    except Exception as e:
        print(f"Error resetting stock-in logs: {e}")
        return {
            "status": "error",
            "message": str(e)
        }
