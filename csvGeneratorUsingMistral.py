import os
import time
import json
import pandas as pd
import base64
import re
import pickle
from PIL import Image
import io
from mistralai import Mistral
from mistralai import DocumentURLChunk, ImageURLChunk, TextChunk
from pathlib import Path
from datetime import datetime
import random
import concurrent.futures

def clean_price(price_str):
    """Convert price string to float, handling various formats."""
    if not price_str or price_str == "N/A":
        return 0.0
    
    if isinstance(price_str, (int, float)):
        return float(price_str)
    
    # Handle special text values
    if isinstance(price_str, str):
        if price_str.upper() in ["FREE", "COMP", "COMPLIMENTARY", "GRATIS", "N/C", "NC", "NO CHARGE"]:
            return 0.0
    
    # Remove currency symbols and non-numeric characters except decimal point
    clean_str = re.sub(r'[^\d.]', '', str(price_str))
    try:
        return float(clean_str) if clean_str else 0.0
    except ValueError:
        print(f"Warning: Could not convert '{price_str}' to float, using 0.0")
        return 0.0

def standardize_date_format(date_str):
    """Convert various date formats to DD-MM-YYYY format."""
    if not date_str:
        return ""
    
    # Common date formats to try
    date_formats = [
        '%Y-%m-%d',       # 2023-01-31
        '%m/%d/%Y',       # 01/31/2023
        '%d/%m/%Y',       # 31/01/2023
        '%m-%d-%Y',       # 01-31-2023
        '%d-%m-%Y',       # 31-01-2023
        '%B %d, %Y',      # January 31, 2023
        '%d %B %Y',       # 31 January 2023
        '%m/%d/%y',       # 01/31/23
        '%d/%m/%y',       # 31/01/23
        '%Y/%m/%d',       # 2023/01/31
        '%d.%m.%Y',       # 31.01.2023
        '%m.%d.%Y',       # 01.31.2023
    ]
    
    # Try to parse the date
    for fmt in date_formats:
        try:
            date_obj = datetime.strptime(date_str, fmt)
            return date_obj.strftime('%d-%m-%Y')  # Convert to DD-MM-YYYY
        except ValueError:
            continue
    
    # If we can't parse, try to extract date components with regex
    try:
        # Look for patterns like DD/MM/YYYY or YYYY-MM-DD
        date_pattern = r'(\d{1,4})[-./](\d{1,2})[-./](\d{1,4})'
        match = re.search(date_pattern, date_str)
        if match:
            part1, part2, part3 = match.groups()
            
            # Try to determine format based on values
            if len(part1) == 4:  # Likely YYYY-MM-DD
                year, month, day = part1, part2, part3
            elif len(part3) == 4:  # Likely DD/MM/YYYY
                day, month, year = part1, part2, part3
            else:  # Best guess based on US format MM/DD/YYYY
                month, day, year = part1, part2, part3
                
                # If day > 12, then it's likely DD/MM/YY(YY)
                if int(day) > 12:
                    day, month = month, day
                    
            # Fix two-digit years
            if len(year) == 2:
                if int(year) > 50:  # Assume 19xx for years > 50
                    year = f"19{year}"
                else:  # Assume 20xx for years <= 50
                    year = f"20{year}"
                    
            # Ensure day and month are 2 digits
            day = day.zfill(2)
            month = month.zfill(2)
            
            # Return formatted date
            return f"{day}-{month}-{year}"
    except Exception as e:
        print(f"Error parsing date '{date_str}': {e}")
    
    # If all parsing fails, return original
    print(f"Warning: Could not standardize date format for '{date_str}'")
    return date_str

def validate_json_structure(receipt_data):
    """Validate and fix JSON structure for consistent formatting."""
    required_keys = ['business_name', 'business_address', 'date', 'items', 'subtotal', 'tax', 'total']
    
    # Initialize missing keys with default values
    for key in required_keys:
        if key not in receipt_data:
            if key == 'items':
                receipt_data[key] = []
            elif key in ['subtotal', 'tax', 'total']:
                receipt_data[key] = 0.0
            else:
                receipt_data[key] = ""
    
    # Standardize date format
    if receipt_data['date']:
        receipt_data['date'] = standardize_date_format(receipt_data['date'])
    
    # Ensure items is a list of dictionaries with required fields
    if not isinstance(receipt_data['items'], list):
        receipt_data['items'] = []
    
    valid_items = []
    for item in receipt_data['items']:
        if isinstance(item, dict):
            # Ensure each item has required fields with correct types
            valid_item = {
                'description': str(item.get('description', '')).strip(),
                'quantity': item.get('quantity', 1),
                'price': clean_price(item.get('price', 0))
            }
            
            # Normalize quantity to numeric
            if not isinstance(valid_item['quantity'], (int, float)):
                try:
                    valid_item['quantity'] = float(re.sub(r'[^\d.]', '', str(valid_item['quantity'])))
                    # Convert to int if it's a whole number
                    if valid_item['quantity'].is_integer():
                        valid_item['quantity'] = int(valid_item['quantity'])
                except:
                    valid_item['quantity'] = 1
            
            # Only add items with a description
            if valid_item['description']:
                valid_items.append(valid_item)
    
    receipt_data['items'] = valid_items
    
    # Convert numeric fields to float
    receipt_data['subtotal'] = clean_price(receipt_data['subtotal'])
    receipt_data['tax'] = clean_price(receipt_data['tax'])
    receipt_data['total'] = clean_price(receipt_data['total'])
    
    return receipt_data

def resize_image_if_needed(img, max_size_mb=3.0):
    """Resize image if it's too large for API processing."""
    # Check current size by simulating conversion to bytes
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG', quality=95)
    size_mb = len(img_byte_arr.getvalue()) / (1024 * 1024)
    
    if size_mb <= max_size_mb:
        return img
    
    print(f"⚠️ Image size ({size_mb:.2f} MB) exceeds {max_size_mb} MB, resizing...")
    
    # Calculate scale factor to reduce to target size
    scale_factor = (max_size_mb / size_mb) ** 0.5  # Square root for 2D scaling
    new_width = int(img.width * scale_factor)
    new_height = int(img.height * scale_factor)
    
    # Resize image
    resized_img = img.resize((new_width, new_height), Image.LANCZOS)
    
    # Verify new size
    check_arr = io.BytesIO()
    resized_img.save(check_arr, format='JPEG', quality=90)
    new_size_mb = len(check_arr.getvalue()) / (1024 * 1024)
    print(f"✅ Image resized to {new_width}x{new_height} ({new_size_mb:.2f} MB)")
    
    return resized_img

def process_receipt(image_path, client, retry_count=0):
    """Process a receipt image and extract data using Mistral."""
    start_time = time.time()
    max_retries = 3
    
    try:
        # Open and encode the image
        with open(image_path, 'rb') as img_file:
            img = Image.open(img_file)
            
            # Resize image if too large
            img = resize_image_if_needed(img, max_size_mb=3.0)
            
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='JPEG', quality=90)
            img_byte_arr.seek(0)
            encoded_image = base64.b64encode(img_byte_arr.getvalue()).decode()
            base64_data_url = f"data:image/jpeg;base64,{encoded_image}"
        
        # Process with Mistral OCR
        print(f"Processing {image_path}...")
        try:
            # Set a timeout for the API request
            with concurrent.futures.ThreadPoolExecutor() as executor:
                ocr_future = executor.submit(client.ocr.process,
                    document=ImageURLChunk(image_url=base64_data_url),
                    model="mistral-ocr-latest"
                )
                # 60 second timeout for OCR
                image_response = ocr_future.result(timeout=60)  
                image_ocr_markdown = image_response.pages[0].markdown
                
        except concurrent.futures.TimeoutError:
            print("⏱️ OCR processing timed out after 60 seconds")
            raise TimeoutError("OCR processing timed out")
        except Exception as e:
            if "429" in str(e) and retry_count < max_retries:
                # Rate limit hit - wait and retry
                wait_time = (2 ** retry_count) * 30 + random.randint(1, 30)
                print(f"⚠️ Rate limit hit! Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
                return process_receipt(image_path, client, retry_count + 1)
            elif "400" in str(e) and retry_count < max_retries:
                # File too large - resize more aggressively and retry
                print(f"⚠️ File size error! Retrying with more aggressive resizing...")
                with open(image_path, 'rb') as img_file:
                    img = Image.open(img_file)
                    # More aggressive resize
                    img = resize_image_if_needed(img, max_size_mb=1.5 - (retry_count * 0.5))
                    # Try again
                    time.sleep(5)
                    return process_receipt(image_path, client, retry_count + 1)
            else:
                raise e
    
        # Process the OCR text to extract structured data
        print(f"Analyzing receipt data...")
        try:
            # Add progress indicator to show activity
            start_chat_time = time.time()
            print("Waiting for AI response", end="", flush=True)
            
            # Use ThreadPoolExecutor to implement timeout for chat
            with concurrent.futures.ThreadPoolExecutor() as executor:
                chat_future = executor.submit(
                    client.chat.complete,
                    model="ministral-8b-latest",
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                TextChunk(
                                    text=(
                                        f"This is image's OCR in markdown:\n\n{image_ocr_markdown}\n.\n"
                                        "Extract the following information from this receipt in JSON format:\n"
                                        "1. business_name: The name of the business or restaurant\n"
                                        "2. business_address: The full address of the business\n"
                                        "3. date: The receipt date in format DD-MM-YYYY\n"
                                        "4. items: Array of purchased items, each with description, quantity, and price\n"
                                        "5. subtotal: The subtotal amount before tax\n"
                                        "6. tax: The tax amount\n"
                                        "7. total: The total amount paid\n\n"
                                        "Return only valid JSON with these fields. For items, always include 'description', 'quantity', and 'price' fields."
                                    )
                                ),
                            ],
                        }
                    ],
                    response_format={"type": "json_object"},
                    temperature=0
                )
                
                # Progress indicator
                dots = 0
                while not chat_future.done():
                    dots = (dots % 3) + 1
                    print("." * dots + " " * (3 - dots) + "\b" * 4, end="", flush=True)
                    time.sleep(1)
                
                # 90 second timeout for chat
                chat_response = chat_future.result(timeout=90)
                
            print(f" - completed in {time.time() - start_chat_time:.1f}s")
            
        except concurrent.futures.TimeoutError:
            print("\n⏱️ AI analysis timed out after 90 seconds")
            raise TimeoutError("AI analysis timed out")
        except Exception as e:
            if "429" in str(e) and retry_count < max_retries:
                # Rate limit hit - wait and retry
                wait_time = (2 ** retry_count) * 30 + random.randint(1, 30)
                print(f"\n⚠️ Rate limit hit! Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
                return process_receipt(image_path, client, retry_count + 1)
            else:
                print(f"\n❌ Error during analysis: {str(e)}")
                raise e
        
        # Extract and validate the JSON response
        try:
            receipt_json = json.loads(chat_response.choices[0].message.content)
            receipt_json = validate_json_structure(receipt_json)
            
            # Create result dictionary
            result = {
                'file_name': os.path.basename(image_path),
                'business_name': receipt_json.get('business_name', ''),
                'business_address': receipt_json.get('business_address', ''),
                'date': receipt_json.get('date', ''),
                'items': receipt_json.get('items', []),
                'subtotal': receipt_json.get('subtotal', 0.0),
                'tax': receipt_json.get('tax', 0.0),
                'total': receipt_json.get('total', 0.0),
                'processing_time': time.time() - start_time,
                'raw_ocr': image_ocr_markdown,
                'extracted_json': json.dumps(receipt_json)
            }
            
            return result
            
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {str(e)}")
            return {
                'file_name': os.path.basename(image_path),
                'business_name': '',
                'business_address': '',
                'date': '',
                'items': [],
                'subtotal': 0.0,
                'tax': 0.0,
                'total': 0.0,
                'processing_time': time.time() - start_time,
                'raw_ocr': image_ocr_markdown,
                'extracted_json': '{}'
            }
            
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        return None

def format_items_for_csv(items):
    """Format items array into a readable string for CSV."""
    if not items:
        return ""
    
    items_str = "; ".join([
        f"{item.get('quantity', 1)}x {item.get('description', '')} (${float(item.get('price', 0)):.2f})" 
        for item in items if item.get('description')
    ])
    return items_str

def save_checkpoint(data, checkpoint_path):
    """Save checkpoint data to resume processing later."""
    with open(checkpoint_path, 'wb') as f:
        pickle.dump(data, f)
    print(f"✅ Checkpoint saved: {checkpoint_path}")

def load_checkpoint(checkpoint_path):
    """Load checkpoint data to resume processing."""
    try:
        with open(checkpoint_path, 'rb') as f:
            data = pickle.load(f)
        print(f"✅ Checkpoint loaded: {checkpoint_path}")
        return data
    except (FileNotFoundError, EOFError, pickle.UnpicklingError):
        print("No valid checkpoint found, starting fresh")
        return None

def append_to_csv(row, output_file, first_write=False):
    """Append a row to the CSV file, creating it if needed."""
    df = pd.DataFrame([row])
    if first_write:
        df.to_csv(output_file, sep='|', index=False, mode='w')
    else:
        df.to_csv(output_file, sep='|', index=False, mode='a', header=False)

def update_row_by_filename(rows, filename, new_row):
    """Update a row in the list of rows based on filename, or add it if not found."""
    for i, row in enumerate(rows):
        if row['File_Name'] == filename:
            rows[i] = new_row
            return
    # If not found, append
    rows.append(new_row)

def generate_dataset(start_idx=None, end_idx=None, output_filename=None, timeout=180):
    """Generate a CSV dataset from receipt images with checkpointing."""
    
    # Generate checkpoint and output filenames if not provided
    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)
    
    if output_filename is None:
        if start_idx is not None and end_idx is not None:
            output_filename = f'receipt_dataset_results_{start_idx}_to_{end_idx-1}.csv'
            checkpoint_path = checkpoint_dir / f'checkpoint_{start_idx}_to_{end_idx-1}.pkl'
        else:
            output_filename = 'final_receipt_dataset_results_formatted.csv'
            checkpoint_path = checkpoint_dir / 'checkpoint_full.pkl'
    else:
        checkpoint_path = checkpoint_dir / f'checkpoint_{os.path.basename(output_filename)}.pkl'
    
    # Load Mistral API key
    with open('.streamlit/secrets.toml', 'r') as f:
        for line in f:
            if 'MISTRAL_API_KEY' in line:
                api_key = line.split('=')[1].strip().strip('"').strip("'")
                break
    
    if not api_key:
        print("API key not found. Please set MISTRAL_API_KEY in .streamlit/secrets.toml")
        return
    
    # Initialize Mistral client
    client = Mistral(api_key=api_key)
    
    # Set up dataset directory path
    dataset_dir = Path('/home/nemilai/Desktop/ReceiptScanning/receiptDataset')
    
    # Get all image files from receiptDataset/
    image_files = sorted(list(dataset_dir.glob('*.jpg')) + 
                        list(dataset_dir.glob('*.jpeg')) + 
                        list(dataset_dir.glob('*.png')))
    
    if not image_files:
        print("No image files found in receiptDataset/ directory")
        return
    
    # Apply range filters if provided
    if start_idx is not None and end_idx is not None:
        start_idx = int(start_idx)
        end_idx = int(end_idx)
        image_files = image_files[start_idx:end_idx]
    
    # Create a mapping of filenames to their paths
    filename_to_path = {os.path.basename(f): f for f in image_files}
    
    # Check if output file exists and load data
    existing_rows = {}
    placeholder_files = []
    if os.path.exists(output_filename):
        try:
            df = pd.read_csv(output_filename, sep='|')
            for _, row in df.iterrows():
                filename = row['File_Name']
                # Check if this is a placeholder row (empty business name and subtotal=0)
                if pd.isna(row['Business_Name']) or row['Business_Name'] == '':
                    if filename in filename_to_path:
                        placeholder_files.append(filename)
                else:
                    # Regular completed row
                    existing_rows[filename] = row
            print(f"Found {len(existing_rows)} completed entries and {len(placeholder_files)} placeholders in existing CSV")
        except Exception as e:
            print(f"Error reading existing CSV: {e}")
    
    # Try to load checkpoint
    checkpoint_data = load_checkpoint(checkpoint_path)
    
    if checkpoint_data:
        # Resume from checkpoint
        results = checkpoint_data.get('results', [])
        skipped_receipts = checkpoint_data.get('skipped', [])
        
        # Update existing_rows from results if newer
        for result in results:
            filename = result['file_name']
            existing_rows[filename] = result
        
        print(f"Loaded {len(results)} results from checkpoint")
        
        # Process placeholder files and files not in the CSV yet
        files_to_process = [
            f for f in image_files 
            if os.path.basename(f) in placeholder_files or 
               os.path.basename(f) not in existing_rows
        ]
    else:
        # Fresh start
        results = []
        skipped_receipts = []
        
        # Process files that don't have completed rows
        files_to_process = [
            f for f in image_files 
            if os.path.basename(f) not in existing_rows or
               os.path.basename(f) in placeholder_files
        ]
    
    print(f"Found {len(files_to_process)} files to process")
    print(f"Timeout set to {timeout} seconds per receipt")
    
    # Load existing CSV data for appending
    all_rows = []
    if os.path.exists(output_filename):
        try:
            all_rows = pd.read_csv(output_filename, sep='|').to_dict('records')
        except pd.errors.EmptyDataError:
            # Handle the empty CSV file case
            print("Empty CSV file detected, creating with headers")
            # Create CSV with headers
            dummy_row = {
                'File_Name': '', 'Business_Name': '', 'Business_Address': '',
                'Date': '', 'Items': '', 'Subtotal': 0, 'Tax': 0, 'Total': 0,
                'Processing_Time_Seconds': 0, 'Raw_OCR': '', 'Extracted_JSON': '{}'
            }
            pd.DataFrame([dummy_row]).to_csv(output_filename, sep='|', index=False)
            # Initialize all_rows with the empty row
            all_rows = [dummy_row]
    else:
        # Create a new CSV file with headers
        dummy_row = {
            'File_Name': '', 'Business_Name': '', 'Business_Address': '',
            'Date': '', 'Items': '', 'Subtotal': 0, 'Tax': 0, 'Total': 0,
            'Processing_Time_Seconds': 0, 'Raw_OCR': '', 'Extracted_JSON': '{}'
        }
        pd.DataFrame([dummy_row]).to_csv(output_filename, sep='|', index=False)
        # Initialize all_rows with the empty row
        all_rows = [dummy_row]
    
    try:
        # Process each receipt
        for i, image_path in enumerate(files_to_process):
            file_name = os.path.basename(image_path)
            print(f"Processing {i+1}/{len(files_to_process)}: {file_name}")
            
            # Add a delay between API calls to avoid rate limiting
            if i > 0:
                time.sleep(random.randint(2, 5))
            
            # Use ThreadPoolExecutor to implement timeout for the entire process_receipt function
            start_time = time.time()
            try:
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(process_receipt, image_path, client)
                    result = future.result(timeout=timeout)  # This will raise TimeoutError if it takes too long
                    
                elapsed = time.time() - start_time
                if result:
                    results.append(result)
                    print(f"Completed in {elapsed:.2f} seconds")
                    
                    # Format result for CSV
                    items_str = format_items_for_csv(result['items'])
                    row = {
                        'File_Name': result['file_name'],
                        'Business_Name': result['business_name'],
                        'Business_Address': result['business_address'],
                        'Date': result['date'],
                        'Items': items_str,
                        'Subtotal': result['subtotal'],
                        'Tax': result['tax'],
                        'Total': result['total'],
                        'Processing_Time_Seconds': result['processing_time'],
                        'Raw_OCR': result['raw_ocr'],
                        'Extracted_JSON': result['extracted_json']
                    }
                    
                    # Update or add row in memory
                    update_row_by_filename(all_rows, file_name, row)
                    
                    # Filter out dummy rows before saving
                    filtered_rows = [r for r in all_rows if r['File_Name'] != '']
                    
                    # Save updated CSV
                    pd.DataFrame(filtered_rows).to_csv(output_filename, sep='|', index=False)
                    
                    # Save checkpoint after each successful processing
                    checkpoint = {
                        'results': results,
                        'skipped': skipped_receipts,
                        'last_processed': file_name,
                        'timestamp': datetime.now().isoformat()
                    }
                    save_checkpoint(checkpoint, checkpoint_path)
                else:
                    print(f"Failed to process {file_name}")
                    skipped_receipts.append(str(image_path))
                    
                    # Add placeholder row for the failed processing
                    placeholder_row = {
                        'File_Name': file_name,
                        'Business_Name': '',
                        'Business_Address': '',
                        'Date': '',
                        'Items': '',
                        'Subtotal': 0,
                        'Tax': 0,
                        'Total': 0,
                        'Processing_Time_Seconds': 0,
                        'Raw_OCR': '',
                        'Extracted_JSON': '{}'
                    }
                    # Update or add placeholder in memory
                    update_row_by_filename(all_rows, file_name, placeholder_row)
                    
                    # Filter out dummy rows before saving
                    filtered_rows = [r for r in all_rows if r['File_Name'] != '']
                    
                    # Save updated CSV
                    pd.DataFrame(filtered_rows).to_csv(output_filename, sep='|', index=False)
                    
            except (concurrent.futures.TimeoutError, TimeoutError) as e:
                elapsed = time.time() - start_time
                print(f"⏱️ TIMEOUT: Skipping {file_name} - processing exceeded {timeout} seconds (took {elapsed:.2f}s)")
                skipped_receipts.append(str(image_path))
                
                # Add placeholder row for the timed-out processing
                placeholder_row = {
                    'File_Name': file_name,
                    'Business_Name': '',
                    'Business_Address': '',
                    'Date': '',
                    'Items': '',
                    'Subtotal': 0,
                    'Tax': 0,
                    'Total': 0,
                    'Processing_Time_Seconds': 0,
                    'Raw_OCR': '',
                    'Extracted_JSON': '{}'
                }
                # Update or add placeholder in memory
                update_row_by_filename(all_rows, file_name, placeholder_row)
                
                # Filter out dummy rows before saving
                filtered_rows = [r for r in all_rows if r['File_Name'] != '']
                
                # Save updated CSV
                pd.DataFrame(filtered_rows).to_csv(output_filename, sep='|', index=False)
                
                # Save checkpoint after each skipped file too
                checkpoint = {
                    'results': results,
                    'skipped': skipped_receipts,
                    'last_processed': file_name,
                    'timestamp': datetime.now().isoformat()
                }
                save_checkpoint(checkpoint, checkpoint_path)
                continue
                
            except Exception as e:
                print(f"❌ ERROR: {str(e)}")
                skipped_receipts.append(str(image_path))
                
                # Add placeholder row for the error case
                placeholder_row = {
                    'File_Name': file_name,
                    'Business_Name': '',
                    'Business_Address': '',
                    'Date': '',
                    'Items': '',
                    'Subtotal': 0,
                    'Tax': 0,
                    'Total': 0,
                    'Processing_Time_Seconds': 0,
                    'Raw_OCR': '',
                    'Extracted_JSON': '{}'
                }
                # Update or add placeholder in memory
                update_row_by_filename(all_rows, file_name, placeholder_row)
                
                # Filter out dummy rows before saving
                filtered_rows = [r for r in all_rows if r['File_Name'] != '']
                
                # Save updated CSV
                pd.DataFrame(filtered_rows).to_csv(output_filename, sep='|', index=False)
                
                # Save checkpoint after each error too
                checkpoint = {
                    'results': results,
                    'skipped': skipped_receipts,
                    'last_processed': file_name,
                    'timestamp': datetime.now().isoformat()
                }
                save_checkpoint(checkpoint, checkpoint_path)
                continue
            
            # Show progress
            total_processed = len(results) + len(skipped_receipts)
            success_rate = (len(results) / total_processed) * 100 if total_processed > 0 else 0
            print(f"Progress: {total_processed}/{len(files_to_process)} ({success_rate:.1f}% success rate)")
    
    except KeyboardInterrupt:
        print("\n⚠️ Processing interrupted by user")
        print("Saving current progress...")
        
        # One final checkpoint
        checkpoint = {
            'results': results,
            'skipped': skipped_receipts,
            'last_processed': os.path.basename(image_path) if 'image_path' in locals() else None,
            'timestamp': datetime.now().isoformat(),
            'interrupted': True
        }
        save_checkpoint(checkpoint, checkpoint_path)
    
    # Save list of skipped receipts if any
    if skipped_receipts:
        skipped_file = f"{os.path.splitext(output_filename)[0]}_skipped.txt"
        with open(skipped_file, 'w') as f:
            f.write("\n".join(skipped_receipts))
        print(f"List of {len(skipped_receipts)} skipped receipts saved to {skipped_file}")
    
    print(f"Results saved to {output_filename}")
    print(f"Total receipts processed: {len(results)} of {len(files_to_process)}")
    print(f"Skipped receipts: {len(skipped_receipts)}")
    if results:
        print(f"Average processing time: {sum(r['processing_time'] for r in results)/len(results):.2f} seconds")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate a CSV dataset from receipt images')
    parser.add_argument('--start', type=int, help='Starting index for the image files')
    parser.add_argument('--end', type=int, help='Ending index for the image files')
    parser.add_argument('--output', type=str, help='Output filename')
    parser.add_argument('--timeout', type=int, default=180, help='Timeout in seconds for processing each receipt (default: 180)')
    
    args = parser.parse_args()
    
    generate_dataset(args.start, args.end, args.output, args.timeout)