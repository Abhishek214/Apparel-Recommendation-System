# Test script to verify BR parser works with sample data
import json
from source.br_parser import br_json_parser

def test_br_parser():
    # Sample JSON that should come from LLM for BR documents
    sample_br_json = {
        "Meeting Date": "2024-01-15",
        "Company Name in English": "Test Company Ltd",
        "Company Name in Chinese": "测试公司",
        "accounts": [
            {
                "Account Number": "All Accounts",
                "Account Type": "Current Accounts and Savings Accounts",
                "Amended Information": [
                    {
                        "Name": "John Doe",
                        "Position": "Director",
                        "Effective Date": "2024-01-01",
                        "Group": "Group A",
                        "Action": "add",
                        "Remarks": "N/A",
                        "Government ID Number": "ID123456"
                    },
                    {
                        "Name": "Jane Smith",
                        "Position": "Authorized Person",
                        "Effective Date": "2024-01-01",
                        "Group": "Group B",
                        "Action": "add",
                        "Remarks": "N/A",
                        "Government ID Number": "ID789012"
                    }
                ],
                "Group wise Signing Authorities": [
                    {
                        "Name": "John Doe",
                        "Group": "Group A"
                    },
                    {
                        "Name": "Jane Smith",
                        "Group": "Group B"
                    }
                ],
                "Signing Instructions": [
                    {
                        "Required_signatories": "1 from Group A and 1 from Group B",
                        "Amount_threshold_Instruction": "HKD >= 20K",
                        "Currency": "HKD"
                    }
                ]
            }
        ]
    }
    
    # Convert to JSON string for testing
    sample_br_json_str = json.dumps(sample_br_json)
    
    # Mock response object
    mock_response = {
        "response": "Test response",
        "document_content": "Test document",
        "documents_used": [{"@search.score": 0.8}, {"@search.score": 0.9}]
    }
    
    try:
        entity_keys, entity_values, table_names, table_arrays = br_json_parser(
            json_str=sample_br_json_str,
            query="Extract account information",
            response=mock_response,
            exchange="test-exchange-token"
        )
        
        print("=== BR Parser Test Results ===")
        print(f"Entity Keys: {entity_keys}")
        print(f"Entity Values: {[v.value for v in entity_values]}")
        print(f"Table Names: {table_names}")
        print(f"Table Arrays Count: {len(table_arrays)}")
        
        for i, table in enumerate(table_arrays):
            print(f"\nTable {i+1} ({table_names[i]}): {len(table)} rows x {len(table[0]) if table else 0} cols")
            if table:
                # Print first row (headers)
                headers = [cell.value for cell in table[0]]
                print(f"Headers: {headers}")
                
                # Print first data row if exists
                if len(table) > 1:
                    first_row = [cell.value for cell in table[1]]
                    print(f"First row: {first_row}")
        
        return True
        
    except Exception as e:
        print(f"BR Parser Test Failed: {e}")
        import traceback
        traceback.print_exc()
        return False

# Run this test to verify the parser works
if __name__ == "__main__":
    test_br_parser()
n
