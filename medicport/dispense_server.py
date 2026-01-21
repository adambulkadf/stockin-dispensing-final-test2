"""
DISPENSE SERVER - Port 8001
Handles all dispensing operations on a separate port.
Shares memory with main server (stock-in on port 8000).
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Import shared data and functions from main
from main import (
    items,
    robots,
    virtual_units,
    shelves,
    racks,
    OUTPUT_POSITIONS,
    save_robots_to_file,
    save_warehouse_state,
    relocate_tasks_store,
)

# Import dispense request/response models and endpoint functions
from dispensing import (
    DispenseRequest,
    CompleteDispenseRequest,
    FailDispenseRequest,
    create_dispense_task_endpoint,
    complete_dispense_endpoint,
    fail_dispense_endpoint,
    get_dispense_logs_endpoint,
    get_product_dispense_history_endpoint,
    get_task_status_endpoint,
)

# Create dispense app
dispense_app = FastAPI(
    title="MedicPort Dispense Server",
    description="Dispensing operations on port 8001",
    version="1.0.0"
)

dispense_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Task counter for dispense (shared via import)
task_counter = 0


@dispense_app.post("/dispense/create", tags=["Dispensing Operations"])
async def create_dispense_task(request: DispenseRequest):
    """
    Create a new dispensing task with multi-pick batching
    """
    global task_counter
    from main import task_counter as main_task_counter, relocate_tasks_store as main_relocate_store

    response, updated_task_counter, updated_relocate_tasks = create_dispense_task_endpoint(
        request,
        items,
        robots,
        virtual_units,
        shelves,
        racks,
        OUTPUT_POSITIONS,
        main_task_counter
    )

    # Update main's task_counter via import
    import main
    main.task_counter = updated_task_counter
    main.relocate_tasks_store.update(updated_relocate_tasks)

    return response


@dispense_app.post("/dispense/complete", tags=["Dispensing Operations"])
async def complete_dispense(request: CompleteDispenseRequest):
    """
    Mark a dispense task as complete - removes items from inventory
    """
    from main import relocate_tasks_store
    return complete_dispense_endpoint(
        request,
        items,
        robots,
        virtual_units,
        shelves,
        save_robots_to_file,
        save_warehouse_state,
        relocate_tasks_store
    )


@dispense_app.post("/dispense/fail", tags=["Dispensing Operations"])
async def fail_dispense(request: FailDispenseRequest):
    """
    Mark a dispense task as failed with optional partial success.

    If successful_trips is provided, those items will be removed from inventory.
    Items from trips NOT in successful_trips remain in inventory.

    Example:
        {
            "task_id": "DISP_001",
            "successful_trips": [1, 2, 3]  // Trips 1-3 dispensed successfully
        }
    """
    return fail_dispense_endpoint(
        request,
        items,
        robots,
        virtual_units,
        shelves,
        save_robots_to_file,
        save_warehouse_state
    )


@dispense_app.get("/dispense/logs", tags=["Dispensing Logs & History"])
async def get_dispense_logs():
    """
    Get comprehensive dispense statistics
    """
    return get_dispense_logs_endpoint()


@dispense_app.get("/dispense/product/{product_id}", tags=["Dispensing Logs & History"])
async def get_product_dispense_history(product_id: int):
    """
    Get detailed dispense history for a specific product
    """
    return get_product_dispense_history_endpoint(product_id)


@dispense_app.get("/dispense/task/{task_id}", tags=["Dispensing Logs & History"])
async def get_task_status(task_id: str):
    """
    Get status of a dispense task
    """
    return get_task_status_endpoint(task_id)


@dispense_app.get("/health", tags=["System"])
async def health_check():
    """Health check for dispense server"""
    return {
        "status": "healthy",
        "server": "dispense",
        "port": 8001,
        "items_in_inventory": len(items),
        "robots_available": len([r for r in robots.values() if r.status == "IDLE"])
    }
