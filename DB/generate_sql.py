import csv
import json

csv_path = 'DB/crop_disease_updates.csv'
sql_path = 'DB/seed.sql'

def escape_sql(val):
    if not val:
        return "''"
    return "'" + val.replace("'", "''") + "'"

with open(csv_path, mode='r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    rows = list(reader)

with open(sql_path, mode='w', encoding='utf-8') as f:
    # We will do this in two passes:
    # 1. Insert crops
    crops = {}
    for r in rows:
        name = r['crop_name'].strip()
        info = r['crop_info'].strip()
        if name not in crops:
            crops[name] = info

    for name, info in crops.items():
        f.write(f"INSERT INTO public.crops (crop_name, info) VALUES ({escape_sql(name)}, {escape_sql(info)}) ON CONFLICT (crop_name) DO UPDATE SET info = EXCLUDED.info;\n")

    f.write("\n")

    # 2. Insert diseases using a subquery to get crop_id
    for r in rows:
        c_name = r['crop_name'].strip()
        d_name = r['disease_name'].strip()
        desc = r['disease_description'].strip()
        cure = r['cure'].strip()
        
        # We assume the disease might exist, but disease table only has id as primary key (not disease_name + crop_id)
        # So we can't just ON CONFLICT DO UPDATE easily without a unique constraint.
        # Let's delete existing first to prevent duplicates, or we can just try to update, if 0 rows, insert.
        # A simple PLpgSQL block can do this gracefully:
        
        block = f"""
DO $$
DECLARE
    v_crop_id bigint;
    v_disease_id bigint;
BEGIN
    SELECT id INTO v_crop_id FROM public.crops WHERE crop_name = {escape_sql(c_name)};
    
    SELECT id INTO v_disease_id FROM public.diseases WHERE crop_id = v_crop_id AND disease_name = {escape_sql(d_name)};
    
    IF v_disease_id IS NOT NULL THEN
        UPDATE public.diseases SET description = {escape_sql(desc)}, cure = {escape_sql(cure)} WHERE id = v_disease_id;
    ELSE
        INSERT INTO public.diseases (crop_id, disease_name, description, cure) VALUES (v_crop_id, {escape_sql(d_name)}, {escape_sql(desc)}, {escape_sql(cure)});
    END IF;
END $$;
"""
        f.write(block)
