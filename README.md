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
        entity_data = json.loads(json_str)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON in br_json_parser: {e}")
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
        if not table_data or not isinstance(table_data[0], dict):
            return []
        
        # Get headers from first dictionary
        headers = list(table_data[0].keys())
        
        # Create table structure
        table_rows = []
        
        # Add header row
        header_row = []
        for header in headers:
            header_row.append(create_entity_value(header))
        table_rows.append(header_row)
        
        # Add data rows
        for row_data in table_data:
            data_row = []
            for header in headers:
                value = str(row_data.get(header, "N/A"))
                data_row.append(create_entity_value(value))
            table_rows.append(data_row)
        
        return table_rows
    
    def process_accounts_structure(accounts_data: List[Dict]) -> None:
        """Process the complex accounts structure and extract tables"""
        nonlocal table_names, table_arrays
        
        for account in accounts_data:
            # Handle simple account fields as key-value pairs
            for key, value in account.items():
                if key in ["Account Number", "Account Type"]:
                    # Convert to camelCase
                    camel_key = key.replace(" ", "").replace("A", "a", 1) if key.startswith("A") else key
                    entity_keys.append(camel_key)
                    entity_values.append(create_entity_value(str(value)))
                
                # Handle complex nested structures as tables
                elif key == "Amended Information" and isinstance(value, list):
                    table_names.append("Amended Information")
                    table_arrays.append(extract_table_from_list(value, "Amended Information"))
                
                elif key == "Group wise Signing Authorities" and isinstance(value, list):
                    table_names.append("Group wise Signing Authorities")
                    table_arrays.append(extract_table_from_list(value, "Group wise Signing Authorities"))
                
                elif key == "Signing Instructions" and isinstance(value, list):
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
                        table_arrays.append(extract_table_from_list(signing_instructions, "Signing Instructions"))
                
                elif "Intra-group" in key and isinstance(value, (list, dict)):
                    # Handle intra-group fund transfer instructions
                    if isinstance(value, dict):
                        intra_group_data = [value]
                    else:
                        intra_group_data = value
                    
                    table_names.append("Intra-group Companies Fund Transfer Instructions")
                    table_arrays.append(extract_table_from_list(intra_group_data, "Intra-group Companies Fund Transfer Instructions"))
    
    # Process each key-value pair in the main JSON
    for key, value in entity_data.items():
        if key.lower() == "accounts" and isinstance(value, str):
            # Parse the accounts string as JSON
            try:
                accounts_json = json.loads(value)
                if isinstance(accounts_json, list):
                    process_accounts_structure(accounts_json)
                else:
                    # Handle single account object
                    process_accounts_structure([accounts_json])
            except json.JSONDecodeError:
                # If can't parse as JSON, treat as regular key-value pair
                entity_keys.append(key)
                entity_values.append(create_entity_value(str(value)))
        
        elif key != "table":  # Skip table key as it's handled separately
            # Convert key to camelCase for consistency
            camel_key = key.replace(" ", "")
            if camel_key and camel_key[0].isupper():
                camel_key = camel_key[0].lower() + camel_key[1:]
            
            entity_keys.append(camel_key)
            entity_values.append(create_entity_value(str(value)))
    
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
