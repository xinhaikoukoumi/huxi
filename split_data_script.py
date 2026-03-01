import os
import pandas as pd
import numpy as np

def split_samples_and_create_dataset(source_dir, output_dir, sample_duration_sec=30):
    """
    Splits samples from CSV files based on filename info and fixed duration.
    
    Filename format assumption: AB, [Label], [Count].[Suffix].csv
    Example: 'AB，A，10.3.csv' -> Label: A, Sample Count: 10
    
    Each sample has a fixed duration of 30 seconds.
    """
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    csv_files = [f for f in os.listdir(source_dir) if f.lower().endswith('.csv')]
    
    total_extracted_samples = 0
    
    print(f"Found {len(csv_files)} CSV files. Starting processing...")
    
    for filename in csv_files:
        try:
            # 1. Parse filename to get label and expected sample count
            # Handle both Chinese and English commas
            clean_filename = filename.replace('，', ',')
            parts = clean_filename.split(',')
            
            if len(parts) < 3:
                print(f"Skipping {filename}: Does not match expected format 'AB,Label,Count...'")
                continue
                
            label = parts[1].strip()
            count_part = parts[2].strip()
            
            # Extract the integer part of the count (e.g., "10.3" -> 10)
            # Find the first number in the 3rd part
            try:
                # Expecting format like "10.3" or "10" or "10.3-..."
                # We'll try to convert the part before the first non-digit/dot if possible, 
                # or just look for the integer part before a dot
                num_str = count_part.split('.')[0] # "10.3" -> "10"
                # If there are other characters (like dates), we might need regex, 
                # but let's try simple split first as user example is "10.3"
                if '-' in num_str: # Handle case like "10-..." if exists
                     num_str = num_str.split('-')[0]
                expected_samples = int(num_str)
            except ValueError:
                print(f"Skipping {filename}: Could not parse sample count from '{count_part}'")
                continue

            file_path = os.path.join(source_dir, filename)
            
            # 2. Read the CSV file
            # Assuming header is on line 1 (0-indexed) based on previous context 
            # (Line 0: ,Ch1... Line 1: Time (s)...)
            try:
                df = pd.read_csv(file_path, header=1)
            except Exception as e:
                print(f"Error reading {filename}: {e}")
                continue

            # Check if 'Time (s)' column exists
            time_col = None
            for col in df.columns:
                if 'Time' in str(col) and '(s)' in str(col):
                    time_col = col
                    break
            
            if time_col is None:
                # Fallback: assume first column is time if checking fails
                time_col = df.columns[0]
                # print(f"Warning: 'Time (s)' column not found in {filename}, utilizing first column '{time_col}' as time.")

            # Ensure time is numeric
            df[time_col] = pd.to_numeric(df[time_col], errors='coerce')
            df = df.dropna(subset=[time_col])
            
            # 3. Split data into chunks of 30 seconds
            # We explicitly use the time column to slice, rather than assuming fixed row counts
            
            # Time starts usually at 0 or small value. 
            # We want [0, 30), [30, 60), ...
            
            extracted_count = 0
            
            for i in range(expected_samples):
                start_time = i * sample_duration_sec
                end_time = (i + 1) * sample_duration_sec
                
                # Filter data for this time window
                sample_df = df[(df[time_col] >= start_time) & (df[time_col] < end_time)]
                
                if sample_df.empty:
                    # If we run out of data before expected samples, stop for this file
                    # print(f"  Notice: {filename} stopped at sample {i+1}/{expected_samples} (No data in range {start_time}-{end_time}s)")
                    break
                
                # Normalize Time to start from 0 for the individual sample file (Optional, but good for consistent viewing)
                # sample_df_copy = sample_df.copy()
                # sample_df_copy[time_col] = sample_df_copy[time_col] - start_time
                
                # Construct output filename
                # Format: Label_OriginalFileIndex_SampleIndex.csv
                # Removing extension from original filename for cleaner name
                base_name = os.path.splitext(filename)[0].replace('，', '_').replace(',', '_')
                output_filename = f"{label}_{base_name}_sample{i+1}.csv"
                output_path = os.path.join(output_dir, output_filename)
                
                sample_df.to_csv(output_path, index=False)
                extracted_count += 1
                total_extracted_samples += 1
            
            print(f"Processed {filename}: Extracted {extracted_count}/{expected_samples} samples.")

        except Exception as e:
            print(f"An unexpected error occurred processing {filename}: {e}")

    print(f"\nProcessing complete. Total split samples created: {total_extracted_samples}")
    print(f"Files saved to: {output_dir}")

if __name__ == "__main__":
    # Source directory containing the original bulk CSVs
    raw_data_dir = r"d:\huxi\data"
    
    # Output directory for the individual split samples
    dataset_dir = r"d:\huxi\split_dataset"
    
    split_samples_and_create_dataset(raw_data_dir, dataset_dir)
