import json
from typing import Dict, List

def process_br_accounts_output(output_json: Dict) -> Dict:
    """
    Post-process the existing BR output to convert accounts string into nested structure.
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
        
        # Extract accounts structure as list of lists
        accounts_list = extract_accounts_as_list_of_lists(accounts_data, accounts_item)
        
        if accounts_list:
            print(f"✅ Created accounts list structure with {len(accounts_list)} account(s)")
            
            # Update the original accounts field with list of lists structure
            i, j = accounts_index
            output_json["values"][i]["kv_pairs"]["values"][j] = {
                "label_name": "accounts",
                "label_value": accounts_list,  # This is now a list of lists
                "source": accounts_item.get("source", "DCREST AI"),
                "confidence_score": accounts_item.get("confidence_score", 95),
                "bbox": accounts_item.get("bbox"),
                "exchange_token": accounts_item.get("exchange_token", "")
            }
            
            print("✅ BR accounts processing complete")
        else:
            print("❌ No accounts structure created from accounts data")
        
    except json.JSONDecodeError as e:
        print(f"❌ Failed to parse accounts JSON: {e}")
        print(f"❌ Problematic string: {accounts_value[:100]}...")
    
    return output_json

def extract_accounts_as_list_of_lists(accounts_data: List[Dict], original_accounts_item: Dict) -> List[List[Dict]]:
    """Extract accounts structure as list of lists - each inner list contains one account's fields"""
    
    accounts_list = []
    source = original_accounts_item.get("source", "DCREST AI")
    confidence_score = original_accounts_item.get("confidence_score", 95)
    exchange_token = original_accounts_item.get("exchange_token", "")
    
    print(f"🔍 Processing {len(accounts_data)} account(s) for list of lists structure")
    
    for account_idx, account in enumerate(accounts_data):
        print(f"📋 Processing account {account_idx + 1}")
        print(f"   Account keys: {list(account.keys()) if isinstance(account, dict) else 'Not a dict'}")
        
        if not isinstance(account, dict):
            continue
        
        # Create a list for this account's fields
        account_fields = []
        
        # Process each field in the account
        for field_name, field_value in account.items():
            print(f"   🔍 Processing field: {field_name} (type: {type(field_value)})")
            
            # Handle simple account fields with schema format
            if field_name in ["Account Number", "Account Type"]:
                camel_case_name = field_name.replace(" ", "").replace("A", "a", 1) if field_name.startswith("A") else field_name.replace(" ", "")
                account_field_item = {
                    "label_name": camel_case_name,
                    "label_value": {"type": "string"},  # Schema format for simple fields
                    "source": source,
                    "confidence_score": {"type": "integer"},
                    "bbox": {"type": "array"},
                    "exchange_token": exchange_token
                }
                account_fields.append(account_field_item)
                print(f"   ✅ Added simple field: {camel_case_name}")
            
            # Handle complex fields as table structures
            elif isinstance(field_value, list) and field_value:
                print(f"   📊 Creating table structure for {field_name} with {len(field_value)} items")
                
                # Convert field name to camelCase
                camel_case_table_name = field_name.replace(" ", "").replace("A", "a", 1) if field_name.startswith("A") else field_name.replace(" ", "")
                
                # Create table structure with real headers and schema data rows
                table_structure = create_table_structure_with_schema(field_value, field_name)
                
                if table_structure:
                    table_field_item = {
                        "table": {
                            "dataTitle": field_name,
                            "source": source,
                            "values": table_structure,
                            "confidence_score": None,
                            "bbox": None,
                            "exchange_token": exchange_token
                        }
                    }
                    account_fields.append(table_field_item)
                    print(f"   ✅ Added table: {field_name}")
        
        if account_fields:
            accounts_list.append(account_fields)
            print(f"   ✅ Added account {account_idx + 1} with {len(account_fields)} fields")
    
    print(f"✅ Created accounts list with {len(accounts_list)} account(s)")
    return accounts_list

def create_table_structure_with_schema(data_list: List[Dict], table_name: str) -> List[List[Dict]]:
    """Create table structure with real headers and schema type data rows"""
    
    if not data_list or not isinstance(data_list[0], dict):
        print(f"   ❌ Invalid data for table {table_name}")
        return []
    
    # Get headers from the first item
    headers = list(data_list[0].keys())
    print(f"   📋 Table headers: {headers}")
    
    # Create table rows
    table_rows = []
    
    # Add header row with real header names
    header_row = []
    for header in headers:
        header_row.append({
            "value": header,  # Real header name
            "confidence_score": None,
            "bbox": None
        })
    table_rows.append(header_row)
    print(f"   📝 Added header row: {headers}")
    
    # Add data rows with schema types (matching your image format)
    for row_idx, row_data in enumerate(data_list):
        data_row = []
        for header in headers:
            data_row.append({
                "value": {"type": "string"},  # Schema format for data rows
                "confidenceScore": {"type": "integer"}, 
                "bbox": {"type": "array"}
            })
        table_rows.append(data_row)
        print(f"   📝 Added schema data row {row_idx + 1}")
    
    print(f"   ✅ Created table structure with {len(table_rows)} rows")
    return table_rows

# Usage: Add this to your extraction process AFTER getting the current output

def apply_br_post_processing(output_json: Dict, doc_type: str) -> Dict:
    """Apply BR post-processing if it's a BR document"""
    
    if doc_type.upper() == "BR":
        print("🎯 Applying BR post-processing...")
        return process_br_accounts_output(output_json)
    else:
        print(f"ℹ️  Skipping BR post-processing for doc_type: {doc_type}")
        return output_json

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
        if 'kv_pairs' in item:
            kv_pairs = item['kv_pairs']['values']
            for kv_pair in kv_pairs:
                if kv_pair.get('label_name') == 'accounts':
                    print(f"Accounts structure:")
                    print(f"  Type: nested list with {len(kv_pair.get('value', []))} items")
                    for nested_item in kv_pair.get('value', []):
                        if 'label_value' in nested_item:
                            print(f"    {nested_item.get('label_name')}: {nested_item.get('label_value')}")
                        elif 'value' in nested_item and isinstance(nested_item['value'], list):
                            print(f"    {nested_item.get('label_name')}: table with {len(nested_item['value'])} rows")
                            if nested_item['value']:
                                headers = [cell['value'] for cell in nested_item['value'][0]]
                                print(f"      Headers: {headers}")

if __name__ == "__main__":
    test_br_postprocessor_with_real_data()
