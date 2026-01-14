"""
RELOCATION MODULE - MedicPort Warehouse System

LOGIC (same as stock-in):
- When item needs relocation (obstruction) -> Find empty VSU or CREATE new VSU
- Place item at BACK of VSU (highest stock_index) - just like stock-in
- Update VSU in ml_robot_updated.json
- Keep relocation_history.json for audit/reference only
- Single source of truth: ml_robot_updated.json

Features:
- Find empty VSU or create new VSU on same/nearby shelf (like stock-in)
- VSU creation uses same logic: 5mm gap, 3mm top clearance
- Item placed at back (max stock_index)
- Update item's VSU directly in main inventory
- Track relocation history for auditing
"""

from fastapi import HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, List, Callable, Any
from datetime import datetime
from pathlib import Path
import json

# Constants (same as main.py)
VSU_HORIZONTAL_GAP = 5  # 5mm gap between VSUs
VSU_TOP_CLEARANCE = 3   # 3mm clearance from top
WIDTH_FIT_TOLERANCE = 1  # 1mm tolerance for fitting

# File paths
RELOCATION_HISTORY_FILE = Path("data/relocation_history.json")


class RelocationRecord(BaseModel):
    """Record of a single relocation event (for audit only)"""
    item_id: int
    product_id: int
    barcode: str
    original_vsu_id: int
    original_vsu_code: str
    new_vsu_id: int
    new_vsu_code: str
    original_coordinates: Dict[str, float]
    new_coordinates: Dict[str, float]
    relocated_at: datetime
    reason: str = "obstruction_removal"

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class RelocationHistory(BaseModel):
    """Relocation history state (reference only)"""
    relocations: Dict[int, RelocationRecord] = {}  # relocation_id -> RelocationRecord
    metadata: Dict = {
        "total_relocations": 0,
        "last_updated": None,
        "version": "1.0"
    }

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


# Global state
relocation_history_state: Optional[RelocationHistory] = None
next_relocation_id = 1


def load_relocation_history() -> RelocationHistory:
    """Load relocation history from JSON file"""
    global relocation_history_state, next_relocation_id

    if not RELOCATION_HISTORY_FILE.exists():
        print(f"[RELOCATION] Creating new history file at {RELOCATION_HISTORY_FILE}")
        relocation_history_state = RelocationHistory()
        save_relocation_history(relocation_history_state)
        return relocation_history_state

    try:
        with open(RELOCATION_HISTORY_FILE, 'r') as f:
            data = json.load(f)

        # Convert relocations dict
        relocations_dict = {}
        for reloc_id_str, reloc_data in data.get("relocations", {}).items():
            reloc_id = int(reloc_id_str)
            reloc_data["relocated_at"] = datetime.fromisoformat(reloc_data["relocated_at"])
            relocations_dict[reloc_id] = RelocationRecord(**reloc_data)

        # Update next ID
        if relocations_dict:
            next_relocation_id = max(relocations_dict.keys()) + 1

        relocation_history_state = RelocationHistory(
            relocations=relocations_dict,
            metadata=data.get("metadata", {
                "total_relocations": len(relocations_dict),
                "last_updated": None,
                "version": "1.0"
            })
        )

        print(f"[RELOCATION] Loaded {len(relocations_dict)} relocation records")
        return relocation_history_state

    except Exception as e:
        print(f"[RELOCATION] Error loading history: {e}")
        relocation_history_state = RelocationHistory()
        save_relocation_history(relocation_history_state)
        return relocation_history_state


def save_relocation_history(history: RelocationHistory):
    """Save relocation history to JSON file"""
    try:
        # Update metadata
        history.metadata["total_relocations"] = len(history.relocations)
        history.metadata["last_updated"] = datetime.now().isoformat()

        # Convert to dict for JSON serialization
        data = {
            "relocations": {
                str(reloc_id): reloc.dict()
                for reloc_id, reloc in history.relocations.items()
            },
            "metadata": history.metadata
        }

        # Ensure directory exists
        RELOCATION_HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)

        # Write to file
        with open(RELOCATION_HISTORY_FILE, 'w') as f:
            json.dump(data, f, indent=2, default=str)

        print(f"[RELOCATION] Saved {len(history.relocations)} records to {RELOCATION_HISTORY_FILE}")

    except Exception as e:
        print(f"[RELOCATION] Error saving history: {e}")
        raise


def _item_fits_vsu(item, vsu) -> bool:
    """Check if item dimensions fit within VSU dimensions"""
    item_width = item.metadata.dimensions.width
    item_height = item.metadata.dimensions.height
    item_depth = item.metadata.dimensions.depth

    vsu_width = vsu.dimensions.width
    vsu_height = vsu.dimensions.height
    vsu_depth = vsu.dimensions.depth

    # Item must fit in all dimensions
    fits = (item_width <= vsu_width and
            item_height <= vsu_height and
            item_depth <= vsu_depth)

    return fits


def _calculate_vsu_fit_score(item, vsu, is_same_shelf: bool, is_same_rack: bool) -> float:
    """
    Calculate how well an item fits in a VSU (lower = better fit)

    Scoring priorities:
    1. Same shelf preferred (bonus -1000)
    2. Same rack preferred (bonus -500)
    3. Minimal wasted space (volume difference)
    """
    item_volume = (item.metadata.dimensions.width *
                   item.metadata.dimensions.height *
                   item.metadata.dimensions.depth)
    vsu_volume = vsu.dimensions.volume if hasattr(vsu.dimensions, 'volume') else (
        vsu.dimensions.width * vsu.dimensions.height * vsu.dimensions.depth
    )

    # Wasted space (lower is better)
    wasted_space = vsu_volume - item_volume

    # Location bonuses (negative = preferred)
    location_bonus = 0
    if is_same_shelf:
        location_bonus = -1000
    elif is_same_rack:
        location_bonus = -500

    return wasted_space + location_bonus


def _calculate_next_vsu_position(shelf, item_width: float, virtual_units: Dict) -> Optional[float]:
    """
    Calculate X position for new VSU on shelf (same logic as stock-in)
    Returns None if no space available
    """
    shelf_start = shelf.position.x
    shelf_end = shelf_start + shelf.dimensions.width

    # Get VSUs on this shelf
    shelf_vsus = [virtual_units[vid] for vid in shelf.virtual_units if vid in virtual_units]

    if not shelf_vsus:
        # First VSU on shelf
        if shelf_start + item_width <= shelf_end + WIDTH_FIT_TOLERANCE:
            return shelf_start
        else:
            return None

    # Find rightmost VSU
    rightmost = max(shelf_vsus, key=lambda v: v.position.x + v.dimensions.width)
    next_x = rightmost.position.x + rightmost.dimensions.width + VSU_HORIZONTAL_GAP

    # Check if new VSU fits
    if next_x + item_width <= shelf_end + WIDTH_FIT_TOLERANCE:
        return next_x
    else:
        return None


def _create_new_vsu_for_relocation(
    item,
    shelf,
    virtual_units: Dict,
    shelves: Dict,
    vsu_counter: int,
    VirtualStorageUnit,
    Dimensions,
    Position
) -> Optional[tuple]:
    """
    Create new VSU for item on shelf (same logic as stock-in)

    Returns: (new_vsu, updated_vsu_counter) or (None, vsu_counter)
    """
    item_dims = item.metadata.dimensions

    # Validate dimensions
    if item_dims.width > shelf.dimensions.width:
        print(f"      Item width ({item_dims.width}mm) > shelf width ({shelf.dimensions.width}mm)")
        return None, vsu_counter
    if item_dims.height > shelf.dimensions.height - VSU_TOP_CLEARANCE:
        print(f"      Item height ({item_dims.height}mm) > shelf available height ({shelf.dimensions.height - VSU_TOP_CLEARANCE}mm)")
        return None, vsu_counter
    if item_dims.depth > shelf.dimensions.depth:
        print(f"      Item depth ({item_dims.depth}mm) > shelf depth ({shelf.dimensions.depth}mm)")
        return None, vsu_counter

    # Calculate position
    x_pos = _calculate_next_vsu_position(shelf, item_dims.width, virtual_units)
    if x_pos is None:
        print(f"      No horizontal space on shelf {shelf.name}")
        return None, vsu_counter

    # VSU dimensions (same as stock-in)
    vsu_width = item_dims.width
    vsu_height = shelf.dimensions.height - VSU_TOP_CLEARANCE
    vsu_depth = shelf.dimensions.depth

    vsu_counter += 1

    new_vsu = VirtualStorageUnit(
        id=vsu_counter,
        code=f"vu{vsu_counter}",
        dimensions=Dimensions(width=vsu_width, height=vsu_height, depth=vsu_depth),
        position=Position(x=x_pos, y=shelf.position.y, z=shelf.position.z),
        shelf_id=shelf.id,
        rack_id=shelf.rack_id,
        items=[],
        occupied=False
    )

    # Add to virtual_units dict
    virtual_units[new_vsu.id] = new_vsu

    # Add to shelf's virtual_units list
    if new_vsu.id not in shelf.virtual_units:
        shelf.virtual_units.append(new_vsu.id)

    # Also update global shelves dict
    if shelf.id in shelves:
        if new_vsu.id not in shelves[shelf.id].virtual_units:
            shelves[shelf.id].virtual_units.append(new_vsu.id)

    print(f"[RELOCATE-VSU] Created new VSU {new_vsu.code} (ID: {new_vsu.id}) at ({x_pos}, {shelf.position.y}, {shelf.position.z})")
    print(f"[RELOCATE-VSU] VSU dims: {vsu_width}W x {vsu_height}H x {vsu_depth}D on shelf {shelf.name}")

    return new_vsu, vsu_counter


def _find_or_create_vsu_for_relocation(
    item,
    original_shelf_id: int,
    items: Dict,
    virtual_units: Dict,
    shelves: Dict,
    vsu_counter: int,
    VirtualStorageUnit,
    Dimensions,
    Position
) -> tuple:
    """
    Find empty VSU or create new one for relocation (like stock-in)

    Priority:
    1. Empty VSU on same shelf
    2. Empty VSU on same rack
    3. Empty VSU on other shelves
    4. Create new VSU on same shelf
    5. Create new VSU on other shelves (smallest suitable shelf first)

    Returns: (vsu_id, vsu_code, vsu, is_new_vsu, updated_vsu_counter)
    """
    original_shelf = shelves.get(original_shelf_id)
    original_rack_id = original_shelf.rack_id if original_shelf else None

    # STEP 1: Try to find empty VSU
    print(f"[RELOCATE] Step 1: Looking for empty VSUs...")
    result = find_empty_vsu_for_relocate(
        item_to_relocate=item,
        shelf_id=original_shelf_id,
        items=items,
        virtual_units=virtual_units,
        shelves=shelves
    )

    if result is not None:
        vsu_id, vsu_code, vsu = result
        print(f"[RELOCATE] Found empty VSU {vsu_code}")
        return vsu_id, vsu_code, vsu, False, vsu_counter

    # STEP 2: No empty VSU found - create new one
    print(f"[RELOCATE] Step 2: No empty VSU found, creating new VSU...")

    item_dims = item.metadata.dimensions

    # Collect suitable shelves
    suitable_shelves = []
    for shelf in shelves.values():
        # Check height fits
        if item_dims.height > shelf.dimensions.height - VSU_TOP_CLEARANCE:
            continue

        # Check horizontal space available
        next_x = _calculate_next_vsu_position(shelf, item_dims.width, virtual_units)
        if next_x is not None:
            is_same_shelf = (shelf.id == original_shelf_id)
            is_same_rack = (shelf.rack_id == original_rack_id)
            suitable_shelves.append((shelf, is_same_shelf, is_same_rack))

    if not suitable_shelves:
        print(f"[RELOCATE] No suitable shelf found for new VSU")
        return None, None, None, False, vsu_counter

    # Sort: same shelf first, then same rack, then by height (smallest first)
    def shelf_priority(item):
        shelf, is_same, is_same_rack = item
        priority = 0
        if is_same:
            priority = 0
        elif is_same_rack:
            priority = 1
        else:
            priority = 2
        return (priority, shelf.dimensions.height)

    suitable_shelves.sort(key=shelf_priority)

    # Try to create VSU on best shelf
    for shelf, is_same_shelf, is_same_rack in suitable_shelves:
        loc_desc = "same shelf" if is_same_shelf else ("same rack" if is_same_rack else f"shelf {shelf.name}")
        print(f"[RELOCATE] Trying to create VSU on {loc_desc} (height={shelf.dimensions.height}mm)...")

        new_vsu, vsu_counter = _create_new_vsu_for_relocation(
            item=item,
            shelf=shelf,
            virtual_units=virtual_units,
            shelves=shelves,
            vsu_counter=vsu_counter,
            VirtualStorageUnit=VirtualStorageUnit,
            Dimensions=Dimensions,
            Position=Position
        )

        if new_vsu is not None:
            return new_vsu.id, new_vsu.code, new_vsu, True, vsu_counter

    print(f"[RELOCATE] Failed to create VSU on any shelf")
    return None, None, None, False, vsu_counter


def find_empty_vsu_for_relocate(
    item_to_relocate,
    shelf_id: int,
    items: Dict,
    virtual_units: Dict,
    shelves: Dict
) -> Optional[tuple]:
    """
    Find OPTIMAL empty VSU for relocated item (NO new VSU creation)

    Logic:
    1. Collect all empty VSUs that fit the item dimensions
    2. Score each by: same shelf > same rack > other racks, then minimal wasted space
    3. Return the best fit

    Returns: (vsu_id, vsu_code, vsu_object) or None
    """
    item_dims = item_to_relocate.metadata.dimensions
    print(f"[RELOCATE] Finding optimal empty VSU for item {item_to_relocate.id}")
    print(f"  Item dimensions: {item_dims.width}W x {item_dims.height}H x {item_dims.depth}D")

    current_shelf = shelves.get(shelf_id)
    current_rack_id = current_shelf.rack_id if current_shelf else None

    # Collect all candidate VSUs with their scores
    candidates = []  # List of (score, vsu_id, vsu_code, vsu, location_desc)

    # Check all shelves
    for sid, shelf in shelves.items():
        is_same_shelf = (sid == shelf_id)
        is_same_rack = (shelf.rack_id == current_rack_id)

        for vsu_id in shelf.virtual_units:
            vsu = virtual_units.get(vsu_id)
            if not vsu:
                continue

            # Check if VSU is empty
            if vsu.items and len(vsu.items) > 0:
                continue

            # Check if item fits
            if not _item_fits_vsu(item_to_relocate, vsu):
                continue

            # Calculate fit score
            score = _calculate_vsu_fit_score(item_to_relocate, vsu, is_same_shelf, is_same_rack)

            # Build location description
            if is_same_shelf:
                loc_desc = "same shelf"
            elif is_same_rack:
                loc_desc = f"shelf {shelf.name} (same rack)"
            else:
                loc_desc = f"shelf {shelf.name} (rack {shelf.rack_id})"

            candidates.append((score, vsu_id, vsu.code, vsu, loc_desc))

    if not candidates:
        print(f"  No suitable empty VSUs found (none fit item dimensions)")
        return None

    # Sort by score (lower = better)
    candidates.sort(key=lambda x: x[0])

    # Return the best candidate
    best_score, best_vsu_id, best_vsu_code, best_vsu, best_loc = candidates[0]

    print(f"  Found {len(candidates)} suitable empty VSUs")
    print(f"  Best fit: VSU {best_vsu_code} on {best_loc}")
    print(f"    VSU dims: {best_vsu.dimensions.width}W x {best_vsu.dimensions.height}H x {best_vsu.dimensions.depth}D")

    return (best_vsu_id, best_vsu_code, best_vsu)


def relocate_item(
    item_id: int,
    items: Dict,
    virtual_units: Dict,
    shelves: Dict,
    reason: str = "obstruction_removal",
    vsu_counter: int = None,
    VirtualStorageUnit = None,
    Dimensions = None,
    Position = None
) -> Dict:
    """
    Relocate item to new VSU (with VSU creation like stock-in)

    LOGIC (same as stock-in):
    - Find empty VSU or CREATE new VSU if needed
    - Place item at BACK of VSU (highest stock_index) - just like stock-in
    - Update item's VSU in main inventory (items dict)
    - Add relocation record to history file (for audit only)
    - Return new location details with updated vsu_counter

    Args:
        item_id: Item to relocate
        items: Main inventory items dict
        virtual_units: VSU dict
        shelves: Shelf dict
        reason: Reason for relocation
        vsu_counter: Current VSU counter for creating new VSUs
        VirtualStorageUnit: VSU class (passed from main.py)
        Dimensions: Dimensions class (passed from main.py)
        Position: Position class (passed from main.py)

    Returns:
        Dict with new location details including updated vsu_counter
    """
    global relocation_history_state, next_relocation_id

    if relocation_history_state is None:
        relocation_history_state = load_relocation_history()

    # Get item details
    if item_id not in items:
        raise HTTPException(status_code=404, detail=f"Item {item_id} not found")

    item = items[item_id]
    original_vsu_id = item.vsu_id

    if original_vsu_id is None:
        raise HTTPException(status_code=400, detail=f"Item {item_id} not in any VSU")

    original_vsu = virtual_units[original_vsu_id]
    shelf_id = original_vsu.shelf_id

    print(f"\n[RELOCATE] Relocating item {item_id} (product {item.metadata.product_id})")
    print(f"  From: VSU {original_vsu.code} (shelf {shelf_id})")

    # Check if we can create new VSUs
    can_create_vsu = (vsu_counter is not None and
                      VirtualStorageUnit is not None and
                      Dimensions is not None and
                      Position is not None)

    is_new_vsu = False
    updated_vsu_counter = vsu_counter if vsu_counter else 0

    if can_create_vsu:
        # Use new logic: find empty VSU or CREATE new one
        new_vsu_id, new_vsu_code, new_vsu, is_new_vsu, updated_vsu_counter = _find_or_create_vsu_for_relocation(
            item=item,
            original_shelf_id=shelf_id,
            items=items,
            virtual_units=virtual_units,
            shelves=shelves,
            vsu_counter=vsu_counter,
            VirtualStorageUnit=VirtualStorageUnit,
            Dimensions=Dimensions,
            Position=Position
        )
    else:
        # Fallback: only find empty VSU (old behavior)
        result = find_empty_vsu_for_relocate(
            item_to_relocate=item,
            shelf_id=shelf_id,
            items=items,
            virtual_units=virtual_units,
            shelves=shelves
        )
        if result is not None:
            new_vsu_id, new_vsu_code, new_vsu = result
        else:
            new_vsu_id, new_vsu_code, new_vsu = None, None, None

    if new_vsu is None:
        raise HTTPException(
            status_code=503,
            detail=f"No VSU available for relocation (no empty VSU and cannot create new one)"
        )

    # Remove item from original VSU
    if item_id in original_vsu.items:
        original_vsu.items.remove(item_id)
    if not original_vsu.items:
        original_vsu.occupied = False

    # Add item to new VSU
    if not new_vsu.items:
        new_vsu.items = []
    new_vsu.items.append(item_id)
    new_vsu.occupied = True

    # Calculate stock_index - place at BACK (max index) like stock-in
    # For new VSU or empty VSU: this is the first item, so index = max depth slots - 1
    # For simplicity: use the current count as the index (placed at back)
    if is_new_vsu:
        # New VSU - calculate max slots based on depth
        item_depth = item.metadata.dimensions.depth
        depth_slot_size = item_depth + 3  # 3mm gap between items
        max_slots = int(new_vsu.dimensions.depth // depth_slot_size)
        stock_index = max(0, max_slots - 1)  # Place at back (highest index)
    else:
        # Existing VSU - place at back (after existing items)
        # Get current max stock_index in this VSU
        existing_indices = []
        for iid, itm in items.items():
            if itm.vsu_id == new_vsu_id and iid != item_id:
                existing_indices.append(itm.stock_index if itm.stock_index is not None else 0)

        if existing_indices:
            stock_index = max(existing_indices) + 1
        else:
            # Empty VSU - calculate max slots and place at back
            item_depth = item.metadata.dimensions.depth
            depth_slot_size = item_depth + 3
            max_slots = int(new_vsu.dimensions.depth // depth_slot_size)
            stock_index = max(0, max_slots - 1)

    # Update item's VSU reference
    item.vsu_id = new_vsu_id
    item.stock_index = stock_index

    vsu_type = "NEW" if is_new_vsu else "existing"
    print(f"  To: {vsu_type} VSU {new_vsu_code} (stock_index {stock_index} = BACK)")

    # Add to relocation history (for audit only)
    relocation_record = RelocationRecord(
        item_id=item_id,
        product_id=item.metadata.product_id,
        barcode=item.metadata.barcode,
        original_vsu_id=original_vsu_id,
        original_vsu_code=original_vsu.code,
        new_vsu_id=new_vsu_id,
        new_vsu_code=new_vsu_code,
        original_coordinates={
            "x": original_vsu.position.x,
            "y": original_vsu.position.y,
            "z": original_vsu.position.z
        },
        new_coordinates={
            "x": new_vsu.position.x,
            "y": new_vsu.position.y,
            "z": new_vsu.position.z
        },
        relocated_at=datetime.now(),
        reason=reason
    )

    relocation_history_state.relocations[next_relocation_id] = relocation_record
    next_relocation_id += 1

    # Save history
    save_relocation_history(relocation_history_state)

    print(f"  Relocation complete - item updated in main inventory")
    if is_new_vsu:
        print(f"  NEW VSU created: {new_vsu_code} (ID: {new_vsu_id})")

    return {
        "item_id": item_id,
        "original_vsu_id": original_vsu_id,
        "original_vsu_code": original_vsu.code,
        "new_vsu_id": new_vsu_id,
        "new_vsu_code": new_vsu_code,
        "new_coordinates": {
            "x": new_vsu.position.x,
            "y": new_vsu.position.y,
            "z": new_vsu.position.z
        },
        "new_vsu_dimensions": {
            "width": new_vsu.dimensions.width,
            "height": new_vsu.dimensions.height,
            "depth": new_vsu.dimensions.depth
        },
        "stock_index": stock_index,
        "is_new_vsu": is_new_vsu,
        "vsu_counter": updated_vsu_counter,
        "relocated_at": relocation_record.relocated_at.isoformat(),
        "reason": reason
    }


def get_relocation_history() -> List[Dict]:
    """Get all relocation records"""
    global relocation_history_state

    if relocation_history_state is None:
        relocation_history_state = load_relocation_history()

    records = []
    for reloc_id, reloc in relocation_history_state.relocations.items():
        records.append({
            "relocation_id": reloc_id,
            "item_id": reloc.item_id,
            "product_id": reloc.product_id,
            "barcode": reloc.barcode,
            "original_vsu_id": reloc.original_vsu_id,
            "original_vsu_code": reloc.original_vsu_code,
            "new_vsu_id": reloc.new_vsu_id,
            "new_vsu_code": reloc.new_vsu_code,
            "original_coordinates": reloc.original_coordinates,
            "new_coordinates": reloc.new_coordinates,
            "relocated_at": reloc.relocated_at.isoformat(),
            "reason": reloc.reason
        })

    return sorted(records, key=lambda x: x["relocated_at"], reverse=True)


def get_relocation_stats() -> Dict:
    """Get relocation statistics"""
    global relocation_history_state

    if relocation_history_state is None:
        relocation_history_state = load_relocation_history()

    return {
        "total_relocations": len(relocation_history_state.relocations),
        "last_updated": relocation_history_state.metadata.get("last_updated")
    }
