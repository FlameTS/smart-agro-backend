import os
import csv
from supabase import create_client, Client
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

url: str = os.environ.get("SUPABASE_URL")
key: str = os.environ.get("SUPABASE_SERVICE_ROLE_KEY") # Use service role key to bypass RLS if possible
if not key:
    key = os.environ.get("SUPABASE_ANON_KEY")

supabase: Client = create_client(url, key)

csv_path = 'DB/crop_disease_updates.csv'

def main():
    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        return

    with open(csv_path, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            crop_name = row['crop_name'].strip()
            disease_name = row['disease_name'].strip()
            crop_info = row['crop_info'].strip()
            disease_description = row['disease_description'].strip()
            cure = row['cure'].strip()

            print(f"Processing Crop: {crop_name}, Disease: {disease_name}")

            # 1. Handle Crop
            crop_data = supabase.table('crops').select("id").eq("crop_name", crop_name).execute()
            crop_id = None
            if crop_data.data:
                crop_id = crop_data.data[0]['id']
                # Update crop info
                supabase.table('crops').update({"info": crop_info}).eq("id", crop_id).execute()
            else:
                # Insert crop
                new_crop = supabase.table('crops').insert({
                    "crop_name": crop_name,
                    "info": crop_info
                }).execute()
                if new_crop.data:
                    crop_id = new_crop.data[0]['id']

            if not crop_id:
                print(f"Failed to get or create crop_id for {crop_name}")
                continue

            # 2. Handle Disease
            disease_data = supabase.table('diseases').select("id").eq("crop_id", crop_id).eq("disease_name", disease_name).execute()
            if disease_data.data:
                disease_id = disease_data.data[0]['id']
                # Update disease
                supabase.table('diseases').update({
                    "description": disease_description,
                    "cure": cure
                }).eq("id", disease_id).execute()
            else:
                # Insert disease
                supabase.table('diseases').insert({
                    "crop_id": crop_id,
                    "disease_name": disease_name,
                    "description": disease_description,
                    "cure": cure
                }).execute()

    print("Finished updating database.")

if __name__ == '__main__':
    main()
