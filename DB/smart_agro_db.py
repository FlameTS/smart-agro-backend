#!/usr/bin/env python3
"""
smart_agro_db.py

Supabase-backed database module for Smart Agro project.

- Uses Supabase REST API via supabase-py
- Provides functions to add/delete/list/search crops & diseases
- Schema and seed data are managed in Supabase (no local init needed)

Usage:
    python smart_agro_db.py   # runs CLI to test functions
"""

import os
from typing import List, Optional, Dict, Any
from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")

if not SUPABASE_URL or not SUPABASE_ANON_KEY:
    raise RuntimeError("SUPABASE_URL and SUPABASE_ANON_KEY must be set in .env")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)


# -------------------------
# CRUD functions for crops
# -------------------------
def add_crop(crop_name: str, info: Optional[str] = None) -> bool:
    """Return True if added, False if exists or failed."""
    if info is None:
        info = "Crop information will be added later."
    try:
        supabase.table("crops").insert({
            "crop_name": crop_name,
            "info": info
        }).execute()
        return True
    except Exception:
        return False


def delete_crop(crop_name: str) -> bool:
    """
    Delete a crop and its diseases (cascade handled by FK ON DELETE CASCADE).
    Return True if deleted, False if not found.
    """
    # Check if crop exists first
    result = supabase.table("crops").select("id").eq("crop_name", crop_name).execute()
    if not result.data:
        return False
    supabase.table("crops").delete().eq("crop_name", crop_name).execute()
    return True


def update_crop_info(crop_name: str, new_info: str) -> bool:
    result = (
        supabase.table("crops")
        .update({"info": new_info, "updated_at": "now()"})
        .eq("crop_name", crop_name)
        .execute()
    )
    return len(result.data) > 0


def get_crop_by_name(crop_name: str) -> Optional[Dict[str, Any]]:
    result = supabase.table("crops").select("*").eq("crop_name", crop_name).execute()
    if result.data:
        return result.data[0]
    return None


def list_crops() -> List[Dict[str, Any]]:
    result = supabase.table("crops").select("*").order("crop_name").execute()
    return result.data


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

    # Get crop id
    crop = supabase.table("crops").select("id").eq("crop_name", crop_name).execute()
    if not crop.data:
        return False
    crop_id = crop.data[0]["id"]

    try:
        supabase.table("diseases").insert({
            "crop_id": crop_id,
            "disease_name": disease_name,
            "description": description,
            "cure": cure
        }).execute()
        return True
    except Exception:
        return False


def delete_disease(crop_name: str, disease_name: str) -> bool:
    """
    Delete a disease under a crop. Return True if deleted, False if not found.
    """
    crop = supabase.table("crops").select("id").eq("crop_name", crop_name).execute()
    if not crop.data:
        return False
    crop_id = crop.data[0]["id"]

    result = supabase.table("diseases").select("id").eq("crop_id", crop_id).eq("disease_name", disease_name).execute()
    if not result.data:
        return False

    supabase.table("diseases").delete().eq("crop_id", crop_id).eq("disease_name", disease_name).execute()
    return True


def update_disease(crop_name: str, disease_name: str, description: Optional[str] = None, cure: Optional[str] = None) -> bool:
    """
    Update disease description/cure. Return True if updated, False otherwise.
    """
    if description is None and cure is None:
        return False

    crop = supabase.table("crops").select("id").eq("crop_name", crop_name).execute()
    if not crop.data:
        return False
    crop_id = crop.data[0]["id"]

    update_data = {}
    if description is not None:
        update_data["description"] = description
    if cure is not None:
        update_data["cure"] = cure
    update_data["updated_at"] = "now()"

    result = (
        supabase.table("diseases")
        .update(update_data)
        .eq("crop_id", crop_id)
        .eq("disease_name", disease_name)
        .execute()
    )
    return len(result.data) > 0


def get_diseases_by_crop(crop_name: str) -> List[Dict[str, Any]]:
    crop = supabase.table("crops").select("id").eq("crop_name", crop_name).execute()
    if not crop.data:
        return []
    crop_id = crop.data[0]["id"]

    result = (
        supabase.table("diseases")
        .select("*")
        .eq("crop_id", crop_id)
        .order("disease_name")
        .execute()
    )
    return result.data


def get_disease(crop_name: str, disease_name: str) -> Optional[Dict[str, Any]]:
    crop = supabase.table("crops").select("id").eq("crop_name", crop_name).execute()
    if not crop.data:
        return None
    crop_id = crop.data[0]["id"]

    result = (
        supabase.table("diseases")
        .select("*")
        .eq("crop_id", crop_id)
        .eq("disease_name", disease_name)
        .execute()
    )
    return result.data[0] if result.data else None


def list_all_diseases() -> List[Dict[str, Any]]:
    """List all diseases with their crop names (via join)."""
    # Supabase supports foreign key joins via select syntax
    result = (
        supabase.table("diseases")
        .select("id, disease_name, description, cure, crops(crop_name)")
        .order("disease_name")
        .execute()
    )
    # Flatten the result for compatibility
    diseases = []
    for row in result.data:
        diseases.append({
            "id": row["id"],
            "crop_name": row["crops"]["crop_name"] if row.get("crops") else "Unknown",
            "disease_name": row["disease_name"],
            "description": row["description"],
            "cure": row["cure"]
        })
    return diseases


# -------------------------
# Simple CLI for testing
# -------------------------
def _print_menu():
    print("\nSmart Agro DB CLI (Supabase)")
    print("----------------------------")
    print("1. List crops")
    print("2. List all diseases")
    print("3. Show diseases for a crop")
    print("4. Add crop")
    print("5. Delete crop")
    print("6. Add disease")
    print("7. Delete disease")
    print("8. Update crop info")
    print("9. Update disease info/cure")
    print("0. Exit")


def _cli():
    while True:
        _print_menu()
        choice = input("Choice: ").strip()
        if choice == "1":
            crops = list_crops()
            print(f"\nTotal crops: {len(crops)}")
            for c in crops:
                print(f"- {c['crop_name']} | info: {c['info']}")
        elif choice == "2":
            diseases = list_all_diseases()
            print(f"\nTotal disease records: {len(diseases)}")
            for d in diseases:
                print(f"- {d['crop_name']} :: {d['disease_name']}")
        elif choice == "3":
            crop = input("Crop name: ").strip()
            ds = get_diseases_by_crop(crop)
            if not ds:
                print("No diseases found (or crop not found).")
            else:
                for d in ds:
                    print(f"- {d['disease_name']} | desc: {d['description']} | cure: {d['cure']}")
        elif choice == "4":
            name = input("New crop name: ").strip()
            info = input("Info (or leave blank): ").strip()
            ok = add_crop(name, info or None)
            print("Added." if ok else "Already exists / failed.")
        elif choice == "5":
            name = input("Crop name to delete: ").strip()
            ok = delete_crop(name)
            print("Deleted." if ok else "Not found.")
        elif choice == "6":
            crop = input("Crop name: ").strip()
            disease = input("Disease name: ").strip()
            desc = input("Description (or blank): ").strip()
            cure = input("Cure (or blank): ").strip()
            ok = add_disease(crop, disease, desc or None, cure or None)
            print("Added disease." if ok else "Failed (crop missing or already exists).")
        elif choice == "7":
            crop = input("Crop name: ").strip()
            disease = input("Disease name to delete: ").strip()
            ok = delete_disease(crop, disease)
            print("Deleted." if ok else "Not found.")
        elif choice == "8":
            crop = input("Crop name: ").strip()
            info = input("New info text: ").strip()
            ok = update_crop_info(crop, info)
            print("Updated." if ok else "Not found.")
        elif choice == "9":
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
    print("Smart Agro DB module (Supabase) executed directly.")
    print("Schema and seed data are managed in Supabase.")
    _cli()
