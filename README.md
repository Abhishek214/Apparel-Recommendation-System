import json
import statistics
from typing import List, Dict, Union, Optional, Literal, Tuple
from source.constants import EntityValue, ExtractedEntities
from source.log_handler import logger


def br_json_parser(
    json_str: str,
    query: str,
    response: Dict,
    exchange: str,
    logprobs: Optional[List[Dict[str, Union[bytes, str, float]]]] = None,
    confidence_method: Literal[
        "product", "average", "first", "sum", "weighted_avg"
    ] = "product",
) -> Tuple[List[str], List[EntityValue], List[str], List[List[List[EntityValue]]]]:
    """
    Custom parser for BR (Bank Resolution) documents that handles complex nested structures.
    
    Args:
        json_str: JSON string representing the entity
        query: The query/prompt used
        response: The response from the RAG system
        exchange: Exchange token
        logprobs: Logprobs for each string token
        confidence_method: Method for computing confidence scores
    
    Returns:
        Tuple of:
        - entity_keys: List of simple key-value pair names
        - entity_values: List of simple key-value pair values
        - table_names: List of table names extracted from nested structures
        - table_arrays: List of tables (each table is a list of rows, each row is a list of EntityValues)
    """
    
    try:
        if not json_str or json_str.strip() == "":
            logger.warning("Empty JSON string provided to BR parser")
            return [], [], [], []
            
        # Clean the JSON string
        json_str = json_str.strip()
        if not json_str.startswith('{'):
            logger.warning(f"JSON string doesn't start with '{{': {json_str[:50]}...")
            return [], [], [], []
            
        entity_data = json.loads(json_str)
        logger.info(f"BR Parser - Successfully parsed JSON with keys: {list(entity_data.keys())}")
        
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON in br_json_parser: {e}")
        logger.error(f"Problematic JSON string: {json_str}")
        return [], [], [], []
    
    # Lists to store simple key-value pairs
    entity_keys: List[str] = []
    entity_values: List[EntityValue] = []
    
    # Lists to store table data
    table_names: List[str] = []
    table_arrays: List[List[List[EntityValue]]] = []
    
    # Track logprobs usage
    last_used_idx = 0
    
    def create_entity_value(value: str, logprobs_slice: Optional[List] = None) -> EntityValue:
        """Helper function to create EntityValue objects"""
        return EntityValue(
            value=value,
            confidence=None,
            bbox=None,
            context_relevance_score=None,
            query=query,
            answer=response.get("response", ""),
            document_content=response.get("document_content", ""),
            rag_similarity_maen=statistics.mean(
                [doc["@search.score"] for doc in response.get("documents_used", [])]
            ) if response.get("documents_used") else 0,
            logprobs=logprobs_slice,
            exchange=exchange,
        )
    
    def extract_table_from_list(table_data: List[Dict], table_name: str) -> List[List[EntityValue]]:
        """Extract table structure from list of dictionaries"""
        logger.info(f"extract_table_from_list called for '{table_name}' with {len(table_data) if table_data else 0} items")
        
        if not table_data or not isinstance(table_data, list) or len(table_data) == 0:
            logger.warning(f"Empty or invalid table data for {table_name}")
            return []
        
        # Check if first item is a dictionary
        if not isinstance(table_data[0], dict):
            logger.warning(f"First item in {table_name} is not a dict: {type(table_data[0])}")
            return []
        
        # Get headers from first dictionary
        headers = list(table_data[0].keys())
        logger.info(f"Table '{table_name}' headers: {headers}")
        
        # Create table structure
        table_rows = []
        
        # Add header row
        header_row = []
        for header in headers:
            header_row.append(create_entity_value(header))
        table_rows.append(header_row)
        logger.info(f"Added header row with {len(header_row)} columns")
        
        # Add data rows
        for i, row_data in enumerate(table_data):
            if not isinstance(row_data, dict):
                logger.warning(f"Row {i} in {table_name} is not a dict, skipping")
                continue
                
            data_row = []
            for header in headers:
                value = str(row_data.get(header, "N/A"))
                data_row.append(create_entity_value(value))
            table_rows.append(data_row)
            logger.info(f"Added data row {i+1} with values: {[row_data.get(h, 'N/A') for h in headers]}")
        
        logger.info(f"Successfully created table '{table_name}' with {len(table_rows)} total rows")
        return table_rows
    
    def process_accounts_structure(accounts_data: Union[List[Dict], Dict, str]) -> None:
        """Process the complex accounts structure and extract tables"""
        nonlocal table_names, table_arrays, entity_keys, entity_values
        
        try:
            logger.info(f"process_accounts_structure called with type: {type(accounts_data).__name__}")
            
            # Handle different input formats
            if isinstance(accounts_data, str):
                if not accounts_data.strip():
                    logger.warning("Empty accounts string")
                    return
                    
                logger.info(f"Attempting to parse accounts string: {accounts_data[:100]}...")
                try:
                    accounts_data = json.loads(accounts_data)
                    logger.info(f"Successfully parsed accounts string as JSON, result type: {type(accounts_data).__name__}")
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse accounts string as JSON: {e}")
                    logger.error(f"Problematic string: {accounts_data}")
                    return
            
            # Normalize to list format
            if isinstance(accounts_data, dict):
                logger.info("Converting single account dict to list")
                accounts_data = [accounts_data]
            elif not isinstance(accounts_data, list):
                logger.warning(f"Unexpected accounts data type: {type(accounts_data)}")
                return
            
            logger.info(f"Processing {len(accounts_data)} account(s)")
            
            for i, account in enumerate(accounts_data):
                logger.info(f"Processing account {i+1}")
                if not isinstance(account, dict):
                    logger.warning(f"Skipping non-dict account: {type(account)}")
                    continue
                
                logger.info(f"Account {i+1} has keys: {list(account.keys())}")
                
                # Handle simple account fields as key-value pairs
                for key, value in account.items():
                    logger.info(f"  Processing account field: {key} (type: {type(value).__name__})")
                    
                    if key in ["Account Number", "Account Type"]:
                        # Convert to camelCase
                        camel_key = key.replace(" ", "").replace("A", "a", 1) if key.startswith("A") else key
                        entity_keys.append(camel_key)
                        entity_values.append(create_entity_value(str(value)))
                        logger.info(f"    → Added account field: {camel_key} = {value}")
                    
                    # Handle complex nested structures as tables
                    elif key == "Amended Information" and isinstance(value, list):
                        logger.info(f"    → Processing Amended Information with {len(value)} items")
                        table_names.append("Amended Information")
                        table_result = extract_table_from_list(value, "Amended Information")
                        table_arrays.append(table_result)
                        logger.info(f"    → Added table: Amended Information with {len(table_result)} rows")
                    
                    elif key == "Group wise Signing Authorities" and isinstance(value, list):
                        logger.info(f"    → Processing Group wise Signing Authorities with {len(value)} items")
                        table_names.append("Group wise Signing Authorities")
                        table_result = extract_table_from_list(value, "Group wise Signing Authorities")
                        table_arrays.append(table_result)
                        logger.info(f"    → Added table: Group wise Signing Authorities with {len(table_result)} rows")
                    
                    elif key == "Signing Instructions" and isinstance(value, list):
                        logger.info(f"    → Processing Signing Instructions with {len(value)} items")
                        # Handle signing instructions which may have mixed structure
                        signing_instructions = []
                        for instruction in value:
                            if isinstance(instruction, dict):
                                signing_instructions.append(instruction)
                            elif isinstance(instruction, str):
                                # Handle standalone string values
                                signing_instructions.append({"Instruction": instruction})
                        
                        if signing_instructions:
                            table_names.append("Signing Instructions")
                            table_result = extract_table_from_list(signing_instructions, "Signing Instructions")
                            table_arrays.append(table_result)
                            logger.info(f"    → Added table: Signing Instructions with {len(table_result)} rows")
                    
                    elif "Intra-group" in key and isinstance(value, (list, dict)):
                        logger.info(f"    → Processing {key}")
                        # Handle intra-group fund transfer instructions
                        if isinstance(value, dict):
                            intra_group_data = [value]
                        else:
                            intra_group_data = value
                        
                        if intra_group_data:
                            table_names.append("Intra-group Companies Fund Transfer Instructions")
                            table_result = extract_table_from_list(intra_group_data, "Intra-group Companies Fund Transfer Instructions")
                            table_arrays.append(table_result)
                            logger.info(f"    → Added table: Intra-group Companies Fund Transfer Instructions with {len(table_result)} rows")
                    
                    else:
                        logger.info(f"    → Skipping field {key} (not a recognized account field)")
        
        except Exception as e:
            logger.error(f"Error in process_accounts_structure: {e}")
            import traceback
            traceback.print_exc()
    
    # Process each key-value pair in the main JSON
    for key, value in entity_data.items():
        logger.info(f"Processing main field: {key} (type: {type(value).__name__})")
        
        if key.lower() == "accounts":
            # Check if accounts field has content
            if value and str(value).strip() != "":
                logger.info(f"Processing accounts field: {type(value).__name__}, value: {str(value)[:100]}...")
                process_accounts_structure(value)
            else:
                logger.warning("Accounts field is empty")
                # Add as simple key-value pair to show it was processed
                entity_keys.append("accounts")
                entity_values.append(create_entity_value("No account information found"))
        
        elif key != "table":  # Skip table key as it's handled separately
            # Convert key to camelCase for consistency
            camel_key = key.replace(" ", "")
            if camel_key and camel_key[0].isupper():
                camel_key = camel_key[0].lower() + camel_key[1:]
            
            entity_keys.append(camel_key)
            entity_values.append(create_entity_value(str(value)))
            logger.info(f"Added simple field: {camel_key} = {str(value)[:50]}...")
    
    logger.info(f"BR Parser completed - Entity Keys: {len(entity_keys)}, Table Names: {len(table_names)}")
    return entity_keys, entity_values, table_names, table_arrays


def br_post_extraction_processing(
    session_token: str,
    azure_token: str,
    background_tasks,
    prompt_data: Dict,
    responses: List[Dict],
    confidence_method: Literal[
        "product", "average", "first", "sum", "weighted_avg"
    ] = "product",
) -> Tuple[ExtractedEntities, List[str]]:
    """
    Custom post-processing function for BR documents that uses the BR parser.
    
    This function replaces the standard post_extraction_processing for BR document type.
    """
    from source.ragClient import exchangeToken, exchangeTok_patch
    from source.content_moderation import (
        non_blocking_docSearch_evaluation,
        non_blocking_response_evaluation,
    )
    from source.utils import extract_json
    
    # Initialize result containers
    params_keys = []
    params_values = []
    param_table = []
    param_table_values = []
    table_exchange = []
    
    # Process each response
    for items, response in zip(prompt_data.values(), responses):
        try:
            exchange_token = exchangeToken(session_token, azure_token, items["prompt"])
            answer = response.get("answer", "")
            logprobs = response.get("logprobs")
            
            if "?" in answer:
                text = extract_json(answer)
                if text:
                    try:
                        if items.get("table") == True:
                            # Use standard table parser for explicit table prompts
                            from source.utils import table_parser
                            table_names, table_values = table_parser(
                                table_json_str=text,
                                logprobs=logprobs,
                                confidence_method=confidence_method,
                                query=items["prompt"],
                                response=response,
                            )
                            param_table.extend(table_names)
                            param_table_values.extend(table_values)
                            table_exchange.extend([exchange_token for _ in table_names])
                        else:
                            # Use BR-specific parser for regular entity extraction
                            entity_keys, entity_values, br_table_names, br_table_arrays = br_json_parser(
                                json_str=text,
                                query=items["prompt"],
                                response=response,
                                exchange=exchange_token,
                                logprobs=logprobs,
                                confidence_method=confidence_method,
                            )
                            
                            # Add simple key-value pairs
                            params_keys.extend(entity_keys)
                            params_values.extend(entity_values)
                            
                            # Add extracted tables
                            param_table.extend(br_table_names)
                            param_table_values.extend(br_table_arrays)
                            table_exchange.extend([exchange_token for _ in br_table_names])
                            
                    except Exception as e:
                        logger.error(f"Error in BR JSON parsing: {e}")
                        # Fallback to treating as simple entity
                        params_keys.append(items["entity_name"])
                        params_values.append(
                            EntityValue(
                                value=text,
                                confidence=None,
                                bbox=None,
                                query=items["prompt"],
                                answer=answer,
                                document_content=response.get("document_content", ""),
                                rag_similarity_maen=statistics.mean(
                                    [doc["@search.score"] for doc in response.get("documents_used", [])]
                                ) if response.get("documents_used") else 0,
                                exchange=exchange_token,
                                logprobs=logprobs,
                            )
                        )
            else:
                # Handle non-JSON responses
                params_keys.append(items["entity_name"])
                params_values.append(
                    EntityValue(
                        value=answer,
                        confidence=None,
                        bbox=None,
                        query=items["prompt"],
                        answer=answer,
                        document_content=response.get("document_content", ""),
                        rag_similarity_maen=statistics.mean(
                            [doc["@search.score"] for doc in response.get("documents_used", [])]
                        ) if response.get("documents_used") else 0,
                        exchange=exchange_token,
                        logprobs=logprobs,
                    )
                )
            
            # Add background tasks
            background_tasks.add_task(
                exchangeTok_patch, azure_token, exchange_token, answer
            )
            background_tasks.add_task(
                non_blocking_docSearch_evaluation,
                azure_token,
                exchange_token,
                items["prompt"],
                answer,
                response.get("document_content", ""),
            )
            background_tasks.add_task(
                non_blocking_response_evaluation,
                azure_token,
                exchange_token,
                items["prompt"],
                answer,
                response.get("document_content", ""),
            )
            
        except Exception as e:
            logger.error(f"Error processing BR document response: {e}")
            continue
    
    # Create ExtractedEntities object
    extracted_entities = ExtractedEntities(
        params_keys=params_keys,
        params_values=params_values,
        param_table=param_table,
        param_table_values=param_table_values,
    )
    
    return extracted_entities, table_exchange
