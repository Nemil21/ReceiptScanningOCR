import streamlit as st
import time
import pandas as pd
import json
import re
from PIL import Image
import io
import base64
from pathlib import Path
from mistralai import Mistral
from mistralai import DocumentURLChunk, ImageURLChunk, TextChunk

st.set_page_config(page_title="Receipt Scanner", layout="wide")

st.title("Receipt Scanner")
st.write("Upload a receipt image to extract information using Mistral OCR")

# Helper function to convert price strings to float
def clean_price(price_str):
    # Remove currency symbols and any non-numeric characters except decimal point
    return float(re.sub(r'[^\d.]', '', str(price_str)))

def process_receipt(image_file):
    # Start timing
    start_time = time.time()
    
    # Set up Mistral client
    api_key = st.secrets.get("MISTRAL_API_KEY", None)
    if not api_key:
        api_key = st.text_input("Enter your Mistral API key:", type="password")
        if not api_key:
            st.error("API key is required")
            return None
    
    client = Mistral(api_key=api_key)
    
    # Convert the uploaded image to base64
    img = Image.open(image_file)
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    encoded_image = base64.b64encode(img_byte_arr.getvalue()).decode()
    base64_data_url = f"data:image/png;base64,{encoded_image}"
    
    # Process with Mistral OCR
    with st.spinner('Extracting text from receipt...'):
        # Process image with OCR
        image_response = client.ocr.process(
            document=ImageURLChunk(image_url=base64_data_url),
            model="mistral-ocr-latest"
        )
        
        # Extract OCR text
        image_ocr_markdown = image_response.pages[0].markdown
    
    # Process the OCR text to extract structured data
    with st.spinner('Analyzing receipt data...'):
        # Get structured response from model
        chat_response = client.chat.complete(
            model="ministral-8b-latest",  # You can also use pixtral-12b-latest if you prefer
            messages=[
                {
                    "role": "user",
                    "content": [
                        TextChunk(
                            text=(
                                f"This is image's OCR in markdown:\n\n{image_ocr_markdown}\n.\n"
                                "Extract the following information from this receipt in JSON format: "
                                "business_name, date, items (array of objects with description, quantity, price), "
                                "subtotal, tax, and total. "
                                "The output should be strictly be json with no extra commentary."
                            )
                        ),
                    ],
                }
            ],
            response_format={"type": "json_object"},
            temperature=0,
        )
        
        # Extract the JSON response
        receipt_json = json.loads(chat_response.choices[0].message.content)
    
    # Process the parsed JSON
    result = {
        'business_name': receipt_json.get('business_name'),
        'date': receipt_json.get('date'),
        'items': receipt_json.get('items', []),
        'subtotal': receipt_json.get('subtotal'),
        'tax': receipt_json.get('tax'),
        'total': receipt_json.get('total'),
        'processing_time': time.time() - start_time,
        'raw_ocr': image_ocr_markdown
    }
    
    return result

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
    
    # Process the receipt
    result = process_receipt(uploaded_file)
    
    if result:
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
                            "Qty": item.get('quantity', 1),
                            "Description": item.get('description', ''),
                            "Price": f"${float(item.get('price', 0)):.2f}",
                            "Total": f"${float(item.get('quantity', 1)) * float(item.get('price', 0)):.2f}"
                        } for item in result['items']
                    ])
                    st.table(items_df)
                else:
                    st.write("No items detected")
                
                # Summary
                st.write("### Summary")
                col_a, col_b, col_c = st.columns(3)
                
                subtotal = float(result['subtotal']) if result['subtotal'] else 0
                col_a.metric("Subtotal", f"${subtotal:.2f}")
                
                tax = float(result['tax']) if result['tax'] else 0
                col_b.metric("Tax", f"${tax:.2f}")
                
                total = float(result['total']) if result['total'] else 0
                col_c.metric("Total", f"${total:.2f}")
                
                # Processing time
                st.info(f"⏱️ Processing time: {result['processing_time']:.2f} seconds")
                
                # Raw OCR text (collapsible)
                with st.expander("View raw OCR text"):
                    st.markdown(result['raw_ocr'])
                # Add JSON display
                with st.expander("View extracted JSON data"):
                    # Create a clean version of the result for display
                    display_json = {
                        'business_name': result['business_name'],
                        'date': result['date'],
                        'items': result['items'],
                        'subtotal': result['subtotal'],
                        'tax': result['tax'],
                        'total': result['total']
                    }
                    
                    # Display formatted JSON
                    st.json(display_json)
                    
                    # Add a button to copy JSON to clipboard
                    json_str = json.dumps(display_json, indent=2)
                    st.code(json_str, language="json")
                    
                    # Add download button
                    st.download_button(
                        label="Download JSON",
                        data=json_str,
                        file_name="receipt_data.json",
                        mime="application/json"
                    )

else:
    # Show instructions when no file is uploaded
    st.info("Please upload an image of a receipt to extract information.")