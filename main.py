import streamlit as st
import requests
import json
import re
import time
import pandas as pd
from PIL import Image
import io

st.set_page_config(page_title="Receipt Scanner", layout="wide")

st.title("Receipt Scanner")
st.write("Upload a receipt image to extract information")

# Helper function to convert price strings to float
def clean_price(price_str):
    # Remove currency symbols and any non-numeric characters except decimal point
    return float(re.sub(r'[^\d.]', '', str(price_str)))

def process_receipt(image_file):
    # Start timing
    start_time = time.time()
    
    url = 'https://app.nanonets.com/api/v2/OCR/Model/4c962449-fc20-4d5d-9750-426f18be10fe/LabelFile/?async=false'
    
    # Create a files dictionary with the uploaded file
    files = {'file': image_file}
    
    response = requests.post(
        url, 
        auth=requests.auth.HTTPBasicAuth('23bd9bfa-0af9-11f0-826f-66ecaf5113f5', ''), 
        files=files
    )
    
    # Parse the JSON response
    receipt_data = json.loads(response.text)
    
    # Extract the prediction data
    prediction = receipt_data['result'][0]['prediction']
    
    # Initialize variables
    business_name = None
    date = None
    tax = None
    total = None
    items = []
    
    # Extract information from the prediction
    for field in prediction:
        if field['label'] == 'Merchant_Name':
            business_name = field['ocr_text']
        elif field['label'] == 'Date':
            date = field['ocr_text']
        elif field['label'] == 'Tax_Amount':
            tax = clean_price(field['ocr_text'])
        elif field['label'] == 'Total_Amount':
            total = clean_price(field['ocr_text'])
        elif field['label'] == 'table':
            # Extract items from the table
            for cell in field['cells']:
                if cell['label'] == 'Description':
                    row = cell['row']
                    description = cell['text']
                    
                    # Find corresponding price and quantity for this row
                    price = next((c['text'] for c in field['cells'] if c['label'] == 'Price' and c['row'] == row), None)
                    quantity = next((c['text'] for c in field['cells'] if c['label'] == 'Quantity' and c['row'] == row), None)
                    
                    if description and price and quantity:
                        items.append({
                            'description': description,
                            'price': clean_price(price),
                            'quantity': int(quantity)
                        })
    
    # Calculate subtotal
    subtotal = sum(item['price'] for item in items)
    
    # Calculate processing time
    processing_time = time.time() - start_time
    
    return {
        'business_name': business_name,
        'date': date,
        'items': items,
        'subtotal': subtotal,
        'tax': tax,
        'total': total,
        'processing_time': processing_time
    }

# File uploader
uploaded_file = st.file_uploader("Choose a receipt image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Uploaded Receipt")
        image = Image.open(uploaded_file)
        st.image(image, width=300)
    
    # Reset file pointer to beginning
    uploaded_file.seek(0)
    
    with st.spinner('Processing receipt...'):
        # Process the receipt
        result = process_receipt(uploaded_file)
    
    with col2:
        # Display receipt information in a card-like container
        with st.container():
            st.success("Receipt successfully processed!")
            
            # Business info
            st.subheader(f"📍 {result['business_name'] or 'Unknown Business'}")
            st.write(f"📅 Date: {result['date'] or 'Unknown'}")
            
            # Items
            st.write("### Items Purchased")
            if result['items']:
                # Convert items to DataFrame for better display
                items_df = pd.DataFrame([
                    {
                        "Qty": item['quantity'],
                        "Description": item['description'],
                        "Price": f"${item['price']:.2f}",
                        "Total": f"${item['quantity'] * item['price']:.2f}"
                    } for item in result['items']
                ])
                st.table(items_df)
            else:
                st.write("No items detected")
            
            # Summary
            st.write("### Summary")
            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Subtotal", f"${result['subtotal']:.2f}")
            col_b.metric("Tax", f"${result['tax']:.2f}" if result['tax'] else "N/A")
            col_c.metric("Total", f"${result['total']:.2f}" if result['total'] else "N/A")
            
            # Processing time
            st.info(f"⏱️ Processing time: {result['processing_time']:.2f} seconds")

else:
    # Show a sample image or instructions when no file is uploaded
    st.info("Please upload an image of a receipt to extract information.")