import duckdb
import os
from huggingface_hub import get_token

# 1. Your Targets
TARGET_IDS = [
    "ygOB2sVGXN1UsRt_Ys_SwA_2",
    "Fsy3vcauEtcBmHEiRSbrBA_2",
    "UQ_0AJhUfqB1XvFU9qAB5A_0",
    "I51ZInKHYPUbKY91HA89Pw_2",
    "wOXtTCkJrmY9mMei2zc2SQ_0"
]

def main():
    print("Initializing DuckDB...")
    con = duckdb.connect()
    
    # 2. Setup (New Syntax for DuckDB v0.10+)
    con.sql("INSTALL httpfs; LOAD httpfs;")
    
    token = get_token()
    if not token:
        print("❌ Error: Not logged in. Run 'hf auth login' first.")
        return

    # THE FIX: Use CREATE SECRET instead of SET
    con.sql(f"CREATE SECRET hf_auth (TYPE HUGGINGFACE, TOKEN '{token}');")

    # 3. Fast Query
    # This URL pattern grabs all parquet files in the data folder
    dataset_url = "hf://datasets/josefbednar/prague-streetview-50k/data/*.parquet"
    
    print(f"Scanning remote dataset for {len(TARGET_IDS)} images...")
    print("(This scans only the IDs first, so it should be fast)")
    
    targets_sql = "', '".join(TARGET_IDS)
    
    # We select 'image' which is a struct containing bytes
    query = f"""
        SELECT image_id, image
        FROM '{dataset_url}' 
        WHERE image_id IN ('{targets_sql}')
    """
    
    results = con.sql(query).fetchall()
    
    if not results:
        print("❌ No matches found. Check your IDs.")
        return

    # 4. Save Images
    os.makedirs("retrieved_images", exist_ok=True)
    print(f"✅ Found {len(results)} matches! Saving to disk...")

    for row in results:
        img_id = row[0]
        img_data = row[1] # This is a dictionary: {'bytes': b'...', 'path': ...}
        
        save_path = f"retrieved_images/{img_id}.jpg"
        
        # Write the binary data directly to a file
        with open(save_path, "wb") as f:
            f.write(img_data['bytes'])
            
        print(f"   -> Saved {save_path}")

if __name__ == "__main__":
    main()