#!/usr/bin/env python3
"""
smart_agro_db.py

Standalone SQLite3 database backbone for Smart Agro project.

- Creates smart_agro.db
- Auto-populates crops & diseases from dataset labels (including healthy entries)
- Provides functions to add/delete/list/search crops & diseases
- Does NOT include interpret_prediction (left for later / separate file)

Usage:
    python smart_agro_db.py   # runs CLI to test functions and will init+populate DB
"""

import sqlite3
from typing import List, Optional, Dict, Any

DB_NAME = "smart_agro.db"

# -------------------------
# Helper / Connection
# -------------------------
def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    return conn

# -------------------------
# Initialization
# -------------------------
def init_db():
    """Create tables if they don't exist."""
    conn = get_connection()
    c = conn.cursor()

    # Crops table
    c.execute("""
        CREATE TABLE IF NOT EXISTS crops (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            crop_name TEXT NOT NULL UNIQUE,
            info TEXT
        )
    """)

    # Diseases table; ensure a crop_id + disease_name pair is unique
    c.execute("""
        CREATE TABLE IF NOT EXISTS diseases (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            crop_id INTEGER NOT NULL,
            disease_name TEXT NOT NULL,
            description TEXT,
            cure TEXT,
            FOREIGN KEY(crop_id) REFERENCES crops(id),
            UNIQUE(crop_id, disease_name)
        )
    """)

    conn.commit()
    conn.close()

# -------------------------
# Auto-populate dataset
# -------------------------
_LABELS = [
"Apple___Apple_scab",
"Apple___Black_rot",
"Apple___Cedar_apple_rust",
"Apple___healthy",
"Blueberry___healthy",
"Cherry_(including_sour)_healthy",
"Cherry_(including_sour)_Powdery_mildew",
"Corn_(maize)_Cercospora_leaf_spot_Gray_leaf_spot",
"Corn_(maize)Common_rust",
"Corn_(maize)_healthy",
"Corn_(maize)_Northern_Leaf_Blight",
"Grape___Black_rot",
"Grape__Esca(Black_Measles)",
"Grape___healthy",
"Grape__Leaf_blight(Isariopsis_Leaf_Spot)",
"Orange__Haunglongbing(Citrus_greening)",
"Peach___Bacterial_spot",
"Peach___healthy",
"Pepper,bell__Bacterial_spot",
"Pepper,bell__healthy",
"Potato___Early_blight",
"Potato___healthy",
"Potato___Late_blight",
"Raspberry___healthy",
"Soybean___healthy",
"Squash___Powdery_mildew",
"Strawberry___healthy",
"Strawberry___Leaf_scorch",
"Tomato___Bacterial_spot",
"Tomato___Early_blight",
"Tomato___healthy",
"Tomato___Late_blight",
"Tomato___Leaf_Mold",
"Tomato___Septoria_leaf_spot",
"Tomato___Spider_mites_Two-spotted_spider_mite",
"Tomato___Target_Spot",
"Tomato___Tomato_mosaic_virus",
"Tomato___Tomato_Yellow_Leaf_Curl_Virus"
]

def _normalize_label(label: str) -> (str, str):
    """
    Split label into (crop_name, disease_name).
    If splitting rules are ambiguous, we keep the original label parts.
    This function assumes the dataset mostly uses '___' to separate crop and disease,
    but some labels in the provided list use different separators or none.
    We will try several separators in order.
    """
    # Try common dataset separator first
    if "___" in label:
        parts = label.split("___", 1)
        return parts[0].strip(), parts[1].strip()
    # Try double underscore with one or two underscores
    if "__" in label:
        parts = label.split("__", 1)
        return parts[0].strip(), parts[1].strip()
    # Try single underscore between crop and disease (less reliable)
    if "_" in label:
        parts = label.split("_", 1)
        return parts[0].strip(), parts[1].strip()
    # Fallback: treat whole label as crop and disease unknown
    return label.strip(), "unknown"

def populate_from_labels(labels: List[str] = None):
    """
    Insert crops and diseases from given label list.
    Now: **includes healthy entries** as disease rows as well.
    """
    if labels is None:
        labels = _LABELS

    conn = get_connection()
    c = conn.cursor()

    # default placeholders
    default_crop_info = "Crop information will be added later."
    default_disease_desc = "Details will be added later."
    default_cure = "Treatment information will be added later."

    # special healthy defaults (we will still insert them into diseases table)
    healthy_description = "Plant appears healthy."
    healthy_cure = "No treatment necessary."

    for label in labels:
        crop_name, disease_name = _normalize_label(label)

        # normalize strings (store exactly as requested, but strip whitespace)
        crop_name = crop_name.strip()
        disease_name = disease_name.strip()

        # ensure crop exists
        c.execute("SELECT id FROM crops WHERE crop_name = ?", (crop_name,))
        row = c.fetchone()
        if not row:
            try:
                c.execute(
                    "INSERT INTO crops (crop_name, info) VALUES (?, ?)",
                    (crop_name, default_crop_info)
                )
                crop_id = c.lastrowid
            except sqlite3.IntegrityError:
                # race or concurrent insertion; fetch it
                c.execute("SELECT id FROM crops WHERE crop_name = ?", (crop_name,))
                crop_id = c.fetchone()["id"]
        else:
            crop_id = row["id"]

        # Decide description/cure values
        if disease_name.lower() == "healthy":
            desc = healthy_description
            cure = healthy_cure
        else:
            desc = default_disease_desc
            cure = default_cure

        # Insert disease (including healthy) - use INSERT OR IGNORE to prevent duplicates
        try:
            c.execute(
                "INSERT OR IGNORE INTO diseases (crop_id, disease_name, description, cure) VALUES (?, ?, ?, ?)",
                (crop_id, disease_name, desc, cure)
            )
        except sqlite3.IntegrityError:
            pass  # ignore existing

    conn.commit()
    conn.close()

# -------------------------
# CRUD functions for crops
# -------------------------
def add_crop(crop_name: str, info: Optional[str] = None) -> bool:
    """Return True if added, False if exists or failed."""
    if info is None:
        info = "Crop information will be added later."
    conn = get_connection()
    c = conn.cursor()
    try:
        c.execute("INSERT INTO crops (crop_name, info) VALUES (?, ?)", (crop_name, info))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def delete_crop(crop_name: str) -> bool:
    """
    Delete a crop and its diseases. Return True if deleted, False if not found.
    """
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT id FROM crops WHERE crop_name = ?", (crop_name,))
    row = c.fetchone()
    if not row:
        conn.close()
        return False
    crop_id = row["id"]
    c.execute("DELETE FROM diseases WHERE crop_id = ?", (crop_id,))
    c.execute("DELETE FROM crops WHERE id = ?", (crop_id,))
    conn.commit()
    conn.close()
    return True

def update_crop_info(crop_name: str, new_info: str) -> bool:
    conn = get_connection()
    c = conn.cursor()
    c.execute("UPDATE crops SET info = ? WHERE crop_name = ?", (new_info, crop_name))
    updated = c.rowcount
    conn.commit()
    conn.close()
    return updated > 0

def get_crop_by_name(crop_name: str) -> Optional[Dict[str, Any]]:
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT * FROM crops WHERE crop_name = ?", (crop_name,))
    row = c.fetchone()
    conn.close()
    if not row:
        return None
    return dict(row)

def list_crops() -> List[Dict[str, Any]]:
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT * FROM crops ORDER BY crop_name COLLATE NOCASE")
    rows = c.fetchall()
    conn.close()
    return [dict(r) for r in rows]

# -------------------------
# CRUD functions for diseases
# -------------------------
def add_disease(crop_name: str, disease_name: str, description: Optional[str] = None, cure: Optional[str] = None) -> bool:
    """
    Add disease for a crop. Return True if added, False if crop not found or already exists.
    """
    if description is None:
        description = "Details will be added later."
    if cure is None:
        cure = "Treatment information will be added later."

    conn = get_connection()
    c = conn.cursor()
    # get crop id
    c.execute("SELECT id FROM crops WHERE crop_name = ?", (crop_name,))
    crop = c.fetchone()
    if not crop:
        conn.close()
        return False
    crop_id = crop["id"]
    try:
        c.execute(
            "INSERT INTO diseases (crop_id, disease_name, description, cure) VALUES (?, ?, ?, ?)",
            (crop_id, disease_name, description, cure)
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def delete_disease(crop_name: str, disease_name: str) -> bool:
    """
    Delete a disease under a crop. Return True if deleted, False if not found.
    """
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT id FROM crops WHERE crop_name = ?", (crop_name,))
    crop = c.fetchone()
    if not crop:
        conn.close()
        return False
    crop_id = crop["id"]
    c.execute("DELETE FROM diseases WHERE crop_id = ? AND disease_name = ?", (crop_id, disease_name))
    deleted = c.rowcount
    conn.commit()
    conn.close()
    return deleted > 0

def update_disease(crop_name: str, disease_name: str, description: Optional[str] = None, cure: Optional[str] = None) -> bool:
    """
    Update disease description/cure. Return True if updated, False otherwise.
    """
    if description is None and cure is None:
        return False
    conn = get_connection()
    c = conn.cursor()
    # join to find correct disease id
    c.execute("""
        UPDATE diseases
        SET description = COALESCE(?, description),
            cure = COALESCE(?, cure)
        WHERE id IN (
            SELECT d.id FROM diseases d
            JOIN crops c ON d.crop_id = c.id
            WHERE c.crop_name = ? AND d.disease_name = ?
        )
    """, (description, cure, crop_name, disease_name))
    updated = c.rowcount
    conn.commit()
    conn.close()
    return updated > 0

def get_diseases_by_crop(crop_name: str) -> List[Dict[str, Any]]:
    conn = get_connection()
    c = conn.cursor()
    c.execute("""
        SELECT d.* FROM diseases d
        JOIN crops c ON d.crop_id = c.id
        WHERE c.crop_name = ?
        ORDER BY d.disease_name COLLATE NOCASE
    """, (crop_name,))
    rows = c.fetchall()
    conn.close()
    return [dict(r) for r in rows]

def get_disease(crop_name: str, disease_name: str) -> Optional[Dict[str, Any]]:
    conn = get_connection()
    c = conn.cursor()
    c.execute("""
        SELECT d.* FROM diseases d
        JOIN crops c ON d.crop_id = c.id
        WHERE c.crop_name = ? AND d.disease_name = ?
    """, (crop_name, disease_name))
    row = c.fetchone()
    conn.close()
    return dict(row) if row else None

def list_all_diseases() -> List[Dict[str, Any]]:
    conn = get_connection()
    c = conn.cursor()
    c.execute("""
        SELECT d.id, c.crop_name, d.disease_name, d.description, d.cure
        FROM diseases d
        JOIN crops c ON d.crop_id = c.id
        ORDER BY c.crop_name COLLATE NOCASE, d.disease_name COLLATE NOCASE
    """)
    rows = c.fetchall()
    conn.close()
    return [dict(r) for r in rows]

# -------------------------
# Simple CLI for testing
# -------------------------
def _print_menu():
    print("\nSmart Agro DB CLI")
    print("------------------")
    print("1. Initialize DB and populate labels")
    print("2. List crops")
    print("3. List all diseases")
    print("4. Show diseases for a crop")
    print("5. Add crop")
    print("6. Delete crop")
    print("7. Add disease")
    print("8. Delete disease")
    print("9. Update crop info")
    print("10. Update disease info/cure")
    print("0. Exit")

def _cli():
    while True:
        _print_menu()
        choice = input("Choice: ").strip()
        if choice == "1":
            init_db()
            populate_from_labels()
            print("DB initialized and populated.")
        elif choice == "2":
            crops = list_crops()
            print(f"\nTotal crops: {len(crops)}")
            for c in crops:
                print(f"- {c['crop_name']} | info: {c['info']}")
        elif choice == "3":
            diseases = list_all_diseases()
            print(f"\nTotal disease records: {len(diseases)}")
            for d in diseases:
                print(f"- {d['crop_name']} :: {d['disease_name']}")
        elif choice == "4":
            crop = input("Crop name: ").strip()
            ds = get_diseases_by_crop(crop)
            if not ds:
                print("No diseases found (or crop not found).")
            else:
                for d in ds:
                    print(f"- {d['disease_name']} | desc: {d['description']} | cure: {d['cure']}")
        elif choice == "5":
            name = input("New crop name: ").strip()
            info = input("Info (or leave blank): ").strip()
            ok = add_crop(name, info or None)
            print("Added." if ok else "Already exists / failed.")
        elif choice == "6":
            name = input("Crop name to delete: ").strip()
            ok = delete_crop(name)
            print("Deleted." if ok else "Not found.")
        elif choice == "7":
            crop = input("Crop name: ").strip()
            disease = input("Disease name: ").strip()
            desc = input("Description (or blank): ").strip()
            cure = input("Cure (or blank): ").strip()
            ok = add_disease(crop, disease, desc or None, cure or None)
            print("Added disease." if ok else "Failed (crop missing or already exists).")
        elif choice == "8":
            crop = input("Crop name: ").strip()
            disease = input("Disease name to delete: ").strip()
            ok = delete_disease(crop, disease)
            print("Deleted." if ok else "Not found.")
        elif choice == "9":
            crop = input("Crop name: ").strip()
            info = input("New info text: ").strip()
            ok = update_crop_info(crop, info)
            print("Updated." if ok else "Not found.")
        elif choice == "10":
            crop = input("Crop name: ").strip()
            disease = input("Disease name: ").strip()
            desc = input("New description (or blank to skip): ").strip()
            cure = input("New cure (or blank to skip): ").strip()
            ok = update_disease(crop, disease, desc or None, cure or None)
            print("Updated." if ok else "Not found or no changes provided.")
        elif choice == "0":
            print("Bye.")
            break
        else:
            print("Invalid choice.")

# -------------------------
# Auto-run when executed
# -------------------------
if __name__ == "__main__":
    print("Smart Agro DB module executed directly.")
    print("If this is the first run, choose option 1 to initialize and populate the database.")
    _cli()
