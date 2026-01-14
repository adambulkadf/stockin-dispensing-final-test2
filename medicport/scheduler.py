"""
Daily archiving scheduler for dispense logs
Automatically archives records older than 30 days to monthly files
"""

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from datetime import datetime
import logging
import json
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SCHEDULER_HISTORY_FILE = "data/scheduler_runs.json"


def save_scheduler_run(result: dict):
    try:
        if os.path.exists(SCHEDULER_HISTORY_FILE):
            with open(SCHEDULER_HISTORY_FILE, 'r') as f:
                history = json.load(f)
        else:
            history = {"runs": []}

        history["runs"].insert(0, result)
        history["runs"] = history["runs"][:30]
        history["last_updated"] = datetime.now().isoformat()

        os.makedirs(os.path.dirname(SCHEDULER_HISTORY_FILE), exist_ok=True)
        with open(SCHEDULER_HISTORY_FILE, 'w') as f:
            json.dump(history, f, indent=2)

    except Exception as e:
        logger.error(f"Error saving scheduler run history: {e}")


def run_daily_archiving():
    start_time = datetime.now()

    try:
        from dispensing import (
            load_dispense_logs,
            save_dispense_logs,
            archive_old_daily_dispenses
        )

        logger.info("="*70)
        logger.info(f"DAILY ARCHIVING TASK STARTED - {start_time}")
        logger.info("="*70)

        logs = load_dispense_logs()

        total_records_before = sum(
            len(p.get('daily_dispenses', {}))
            for p in logs['products'].values()
        )
        products_before = len(logs['products'])

        logger.info(f"Total daily records before archiving: {total_records_before}")
        logger.info(f"Products tracked: {products_before}")

        archived_count = archive_old_daily_dispenses(logs, days_threshold=30)

        total_records_after = sum(
            len(p.get('daily_dispenses', {}))
            for p in logs['products'].values()
        )
        products_after = len(logs['products'])

        save_dispense_logs(logs)

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        logger.info("="*70)
        logger.info(f"DAILY ARCHIVING TASK COMPLETED")
        logger.info(f"  Archived: {archived_count} records")
        logger.info(f"  Records before: {total_records_before}")
        logger.info(f"  Records after: {total_records_after}")
        logger.info(f"  Removed: {total_records_before - total_records_after}")
        logger.info(f"  Duration: {duration:.2f}s")
        logger.info("="*70)

        result = {
            "success": True,
            "timestamp": start_time.isoformat(),
            "completed_at": end_time.isoformat(),
            "duration_seconds": round(duration, 2),
            "records_archived": archived_count,
            "records_before": total_records_before,
            "records_after": total_records_after,
            "records_removed": total_records_before - total_records_after,
            "products_before": products_before,
            "products_after": products_after,
            "trigger": "scheduled"
        }

        save_scheduler_run(result)
        return result

    except Exception as e:
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        logger.error(f"ERROR in daily archiving task: {e}")
        import traceback
        traceback.print_exc()

        result = {
            "success": False,
            "timestamp": start_time.isoformat(),
            "completed_at": end_time.isoformat(),
            "duration_seconds": round(duration, 2),
            "error": str(e),
            "trigger": "scheduled"
        }

        save_scheduler_run(result)
        return result


scheduler = BackgroundScheduler()


def start_scheduler():
    scheduler.add_job(
        run_daily_archiving,
        trigger=CronTrigger(hour=11, minute=0),
        id='daily_archiving',
        name='Daily Dispense Log Archiving',
        replace_existing=True,
        misfire_grace_time=3600
    )

    scheduler.start()
    logger.info("Daily archiving scheduler started")
    logger.info("  Schedule: Every day at 11:00 AM")
    logger.info(f"  Next run: {scheduler.get_job('daily_archiving').next_run_time}")


def stop_scheduler():
    if scheduler.running:
        scheduler.shutdown()
        logger.info("Scheduler stopped")


def get_scheduler_status():
    if not scheduler.running:
        return {
            "running": False,
            "message": "Scheduler not running"
        }

    jobs = scheduler.get_jobs()
    job_info = []

    for job in jobs:
        job_info.append({
            "id": job.id,
            "name": job.name,
            "next_run_time": job.next_run_time.isoformat() if job.next_run_time else None,
            "trigger": str(job.trigger)
        })

    return {
        "running": True,
        "jobs": job_info,
        "current_time": datetime.now().isoformat()
    }


def trigger_manual_archiving():
    logger.info("Manual archiving triggered")
    result = run_daily_archiving()
    result["trigger"] = "manual"
    save_scheduler_run(result)
    return result


def get_scheduler_runs(limit: int = 10):
    try:
        if not os.path.exists(SCHEDULER_HISTORY_FILE):
            return {
                "runs": [],
                "message": "No scheduler runs recorded yet",
                "total": 0
            }

        with open(SCHEDULER_HISTORY_FILE, 'r') as f:
            history = json.load(f)

        runs = history.get("runs", [])[:limit]

        return {
            "runs": runs,
            "total": len(history.get("runs", [])),
            "last_updated": history.get("last_updated"),
            "showing": len(runs)
        }

    except Exception as e:
        logger.error(f"Error loading scheduler run history: {e}")
        return {
            "runs": [],
            "error": str(e),
            "total": 0
        }


if __name__ == "__main__":
    print("Testing Daily Archiving Scheduler\n")
    print("Running archiving task once...")
    result = run_daily_archiving()
    print(f"\nResult: {result}")
