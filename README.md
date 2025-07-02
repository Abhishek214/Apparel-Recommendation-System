import json
import re
from typing import Dict, List, Any

def process_br_accounts_output(output_json: Dict) -> Dict:
    """
    Post-process the existing BR output to convert accounts string into separate tables.
    This works with your current output format.
    """
    
    print("🔄 Processing BR accounts output...")
    
    if not output_json or "values" not in output_json:
        print("❌ No values found in output")
        return output_json
    
    # Find the accounts field in the existing output
    accounts_item = None
    accounts_index = None
    
    for i, value_item in enumerate(output_json["values"]):
        if "kv_pairs" in value_item:
            kv_values = value_item["kv_pairs"].get("values", [])
            for j, kv_pair in enumerate(kv_values):
                if kv_pair.get("label_name", "").lower() == "accounts":
                    accounts_item = kv_pair
                    accounts_index = (i, j)
                    print(f"✅ Found accounts field at position {accounts_index}")
                    break
    
    if not accounts_item:
        print("❌ No accounts field found")
        return output_json
    
    accounts_value = accounts_item.get("label_value", "")
    print(f"📝 Accounts value length: {len(accounts_value)}")
    print(f"📝 Accounts preview: {accounts_value[:200]}...")
    
    if not accounts_value or accounts_value.strip() == "":
        print("❌ Accounts field is empty")
        return output_json
    
    # Parse the accounts JSON string
    try:
        # Clean up the accounts string (remove extra escaping)
        cleaned_accounts = accounts_value.replace('\\"', '"')
        
        # Try to parse as JSON
        accounts_data = json.loads(cleaned_accounts)
        print(f"✅ Successfully parsed accounts JSON")
        print(f"📊 Accounts data type: {type(accounts_data)}")
        
        if isinstance(accounts_data, list):
            print(f"📊 Found {len(accounts_data)} account(s)")
        elif isinstance(accounts_data, dict):
            accounts_data = [accounts_data]
            print(f"📊 Converted single account to list")
        else:
            print(f"❌ Unexpected accounts data type: {type(accounts_data)}")
            return output_json
        
        # Extract tables from accounts data
        tables = extract_tables_from_accounts(accounts_data, accounts_item)
        
        if tables:
            print(f"✅ Created {len(tables)} tables")
            
            # Remove the original accounts field
            i, j = accounts_index
            output_json["values"][i]["kv_pairs"]["values"].pop(j)
            
            # Add the new tables to the output
            output_json["values"].extend(tables)
            
            print("✅ BR accounts processing complete")
        else:
            print("❌ No tables created from accounts data")
        
    except json.JSONDecodeError as e:
        print(f"❌ Failed to parse accounts JSON: {e}")
        print(f"❌ Problematic string: {accounts_value[:100]}...")
    
    return output_json

def extract_tables_from_accounts(accounts_data: List[Dict], original_accounts_item: Dict) -> List[Dict]:
    """Extract table structures from accounts data"""
    
    tables = []
    source = original_accounts_item.get("source", "DCREST AI")
    exchange_token = original_accounts_item.get("exchange_token", "")
    
    print(f"🔍 Processing {len(accounts_data)} account(s) for table extraction")
    
    for account_idx, account in enumerate(accounts_data):
        print(f"📋 Processing account {account_idx + 1}")
        print(f"   Account keys: {list(account.keys()) if isinstance(account, dict) else 'Not a dict'}")
        
        if not isinstance(account, dict):
            continue
        
        # Process each field in the account
        for field_name, field_value in account.items():
            print(f"   🔍 Processing field: {field_name} (type: {type(field_value)})")
            
            # Skip simple fields (these would be handled as key-value pairs)
            if field_name in ["Account Number", "Account Type"]:
                continue
            
            # Process complex fields as tables
            if isinstance(field_value, list) and field_value:
                print(f"   📊 Creating table for {field_name} with {len(field_value)} items")
                
                table = create_table_from_list(field_name, field_value, source, exchange_token)
                if table:
                    tables.append(table)
                    print(f"   ✅ Table created: {field_name}")
                else:
                    print(f"   ❌ Failed to create table: {field_name}")
    
    return tables

def create_table_from_list(table_name: str, data_list: List[Dict], source: str, exchange_token: str) -> Dict:
    """Create a table structure from a list of dictionaries"""
    
    if not data_list or not isinstance(data_list[0], dict):
        print(f"   ❌ Invalid data for table {table_name}")
        return None
    
    # Get headers from the first item
    headers = list(data_list[0].keys())
    print(f"   📋 Table headers: {headers}")
    
    # Create table rows
    table_rows = []
    
    # Add header row
    header_row = []
    for header in headers:
        header_row.append({
            "value": header,
            "confidence_score": None,
            "bbox": None
        })
    table_rows.append(header_row)
    
    # Add data rows
    for row_idx, row_data in enumerate(data_list):
        data_row = []
        for header in headers:
            value = str(row_data.get(header, "N/A"))
            data_row.append({
                "value": value,
                "confidence_score": None,  # You can add confidence calculation here
                "bbox": None
            })
        table_rows.append(data_row)
        print(f"   📝 Added row {row_idx + 1}: {[row_data.get(h, 'N/A') for h in headers]}")
    
    # Create the table structure
    table = {
        "table": {
            "data_title": table_name,
            "source": source,
            "values": table_rows,
            "confidence_score": None,
            "bbox": None,
            "exchange_token": exchange_token
        }
    }
    
    print(f"   ✅ Created table '{table_name}' with {len(table_rows)} rows")
    return table

# Usage: Add this to your extraction process AFTER getting the current output

def apply_br_post_processing(output_json: Dict, doc_type: str) -> Dict:
    """Apply BR post-processing if it's a BR document"""
    
    if doc_type.upper() == "BR":
        print("🎯 Applying BR post-processing...")
        return process_br_accounts_output(output_json)
    else:
        print(f"ℹ️  Skipping BR post-processing for doc_type: {doc_type}")
        return output_json

# Integration example for your extraction_client.py:

def updated_get_extraction_with_br_postprocessing(
    doc_type,
    session,
    azure_token,
    imageBlobID,
    prompt_store,
    background_tasks,
    confidence: bool = True,
    confidence_method: Literal[
        "product", "average", "first", "sum", "weighted_avg"
    ] = "product",
    multimodal: bool = False,
    image_detail: Literal["low", "high"] = "low",
    search_type: Literal["SEARCH", "NO-SEARCH", "IMAGE"] = "SEARCH",
):
    """Updated extraction function that applies BR post-processing"""
    
    try:
        # ... your existing extraction logic ...
        
        # Get the standard output (what you're currently getting)
        json_output = extract_output(
            extracted_entities,
            table_exchange=table_exchanges[0],
            vector=session,
            source="DCREST AI",
            Title=File_Name,
        )
        
        # Apply BR post-processing if it's a BR document
        if doc_type.upper() == "BR":
            print(f"🎯 Applying BR post-processing for {doc_type}")
            json_output = process_br_accounts_output(json_output)
        
        return json_output
        
    except Exception as e:
        logger.exception(e)
        return None

# Quick test function to test with your existing output:

def test_br_postprocessor_with_real_data():
    """Test the BR post-processor with data that matches your actual output"""
    
    # Sample that matches your actual output format
    sample_output = {
        "subsection_title": "TEST_FIXES",
        "values": [
            {
                "kv_pairs": {
                    "data_title": "",
                    "values": [
                        {
                            "label_name": "Is the meeting resolved",
                            "label_value": "Yes",
                            "source": "DCREST AI",
                            "confidence_score": 84,
                            "bbox": None,
                            "exchange_token": "test-token-1"
                        },
                        {
                            "label_name": "Accounts",
                            "label_value": '[{"Account Number": "All Accounts", "Account Type": "Current Accounts and Savings Accounts", "Amended Information": [{"Name": "Paul Simon Rhodes", "Position": "Company Director", "Effective Date": "N/A", "Group": "Group A", "Action": "add", "Remarks": "N/A", "Government ID Number": "N/A"}, {"Name": "Roman Te Kloots", "Position": "Authorized Person", "Effective Date": "N/A", "Group": "Group B", "Action": "add", "Remarks": "N/A", "Government ID Number": "N/A"}], "Group wise Signing Authorities": [{"Name": "Paul Simon Rhodes", "Group": "Group A"}, {"Name": "Roman Te Kloots", "Group": "Group B"}], "Signing Instructions": [{"Required_signatories": "1 from Group A and 1 from Group B", "Amount_threshold_Instruction": "HKD >= 20K", "Currency": "USD"}]}]',
                            "source": "DCREST AI",
                            "confidence_score": 81,
                            "bbox": None,
                            "exchange_token": "test-token-2"
                        }
                    ]
                }
            }
        ],
        "vector": "test-vector"
    }
    
    print("🧪 Testing BR post-processor with real data format...")
    result = process_br_accounts_output(sample_output)
    
    print("\n📊 RESULTS:")
    print(f"Number of value items: {len(result.get('values', []))}")
    
    for i, item in enumerate(result.get('values', [])):
        if 'table' in item:
            table = item['table']
            print(f"Table {i+1}: {table['data_title']}")
            print(f"  Rows: {len(table['values'])}")
            if table['values']:
                headers = [cell['value'] for cell in table['values'][0]]
                print(f"  Headers: {headers}")

if __name__ == "__main__":
    test_br_postprocessor_with_real_data()
