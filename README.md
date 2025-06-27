#!/usr/bin/env python3
"""
Simple RAG Client - Basic document upload and search functionality
"""

import json
import time
import httpx
import asyncio
from typing import Dict, List, Optional, Any, Literal


class SimpleRAGClient:
    """Simple RAG client for document processing and search"""
    
    def __init__(self, rag_service_url: str, document_uploader_url: str, ingest_status_url: str, app_id: str):
        self.rag_service_url = rag_service_url
        self.document_uploader_url = document_uploader_url
        self.ingest_status_url = ingest_status_url
        self.app_id = app_id
        self.timeout = 120
    
    async def upload_document(
        self,
        file_path: str,
        file_name: str,
        azure_token: str,
        multimodal: bool = True,
        chunking_strategy: Literal["BY_PAGE", "BY_SECTION", "BY_PARAGRAPH", "SEMANTIC"] = "BY_PAGE"
    ) -> Optional[Dict[str, Any]]:
        """Upload document with specified chunking strategy"""
        
        headers = {
            "accept": "application/json",
            "Authorization": azure_token
        }
        
        extension = file_name.split(".")[-1].lower()
        
        # Generate a simple session ID for this upload
        session_id = f"dc_auto_{int(time.time())}"
        
        querystring = {
            "title": file_name,
            "classification": "PUBLIC",
            "extension": extension,
            "session_id": session_id,
            "save_document_blob": False,
            "save_document_metadata": False,
            "is_universal_document": False,
            "multimodal": multimodal,
            "chunk_method": chunking_strategy,
        }
        
        try:
            url = f"{self.document_uploader_url}/{self.app_id}"
            
            async with httpx.AsyncClient(timeout=self.timeout, verify=False) as client:
                with open(file_path, "rb") as file:
                    files = {"file": (file_name, file, self._get_content_type(extension))}
                    response = await client.post(
                        url,
                        headers=headers,
                        files=files,
                        params=querystring,
                    )
                
                response.raise_for_status()
                result = response.json()
                # Add session_id to result for future queries
                result["session_id"] = session_id
                return result
                
        except Exception as e:
            print(f"Document upload failed: {e}")
            return None
    
    def _get_content_type(self, extension: str) -> str:
        """Get content type based on file extension"""
        content_types = {
            "pdf": "application/pdf",
            "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "txt": "text/plain",
            "jpg": "image/jpeg",
            "jpeg": "image/jpeg",
            "png": "image/png"
        }
        return content_types.get(extension, "application/octet-stream")
    
    async def get_document_status(self, document_id: str, azure_token: str) -> str:
        """Get document processing status"""
        try:
            headers = {
                "accept": "application/json",
                "Authorization": azure_token
            }
            
            params = {
                "parent_document_id": document_id,
                "paginate_output": "true",
                "is_universal_document": "false",
                "size": "50",
            }
            
            url = f"{self.ingest_status_url}/{self.app_id}"
            
            async with httpx.AsyncClient(timeout=60, verify=False) as client:
                response = await client.get(url, params=params, headers=headers)
                response.raise_for_status()
                
                if response.status_code != 200:
                    return "FAILED"
                
                items = response.json().get("items", [])
                if len(items) == 0:
                    return "PENDING"
                else:
                    return items[0].get("ingest_status", "UNKNOWN")
                    
        except Exception as e:
            print(f"Status check failed: {e}")
            return "FAILED"
    
    async def search_documents(
        self,
        query: str,
        document_id: str,
        azure_token: str,
        system_prompt: str = "You are an intelligent AI assistant",
        model_type: str = "gpt-4o-mini",
        temperature: float = 0.1,
        max_documents: int = 10,
        search_type: Literal["SEARCH", "NO-SEARCH", "IMAGE"] = "SEARCH",
        multimodal: bool = False,
        image_detail: Literal["low", "high"] = "low"
    ) -> Optional[Dict[str, Any]]:
        """Search documents using RAG"""
        
        headers = {
            "accept": "application/json",
            "Authorization": azure_token
        }
        
        # Generate session ID from document ID for search context
        session_id = f"search_{document_id}_{int(time.time())}"
        
        json_data = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ],
            "model": model_type,
            "encoding_model": "cl100k_base",
            "embedding_model": "text-embedding-ada-002",
            "temperature": temperature,
            "vector_store_index": "lydia-embeddings-3",
            "search_text": query,
            "use_tools": False,
            "max_number_of_documents": max_documents,
            "search_universal": "false",
            "save_response": "false",
            "session_id": session_id,
            "model_logprobs": 0,
            "multimodal": multimodal,
            "search_type": search_type,
        }
        
        if multimodal:
            json_data["image_detail"] = image_detail
        
        try:
            url = f"{self.rag_service_url}/{self.app_id}"
            
            async with httpx.AsyncClient(timeout=self.timeout, verify=False) as client:
                retry_count = 0
                max_retries = 3
                
                while retry_count < max_retries:
                    response = await client.post(url, headers=headers, json=json_data)
                    
                    if response.status_code == 429:
                        # Rate limited, wait and retry
                        await asyncio.sleep(2 ** retry_count)
                        retry_count += 1
                        continue
                    
                    response.raise_for_status()
                    result = response.json()
                    
                    # Check if we got meaningful results
                    document_content = result.get("document_content", "")
                    if document_content or search_type != "SEARCH":
                        return result
                    
                    # No meaningful results, retry
                    retry_count += 1
                    if retry_count < max_retries:
                        await asyncio.sleep(2)
                
                # Return even if no documents found
                return result
                
        except Exception as e:
            print(f"Document search failed: {e}")
            return None
    
    async def delete_document(self, document_id: str, azure_token: str) -> bool:
        """Delete document from the RAG system"""
        try:
            headers = {
                "accept": "application/json",
                "Authorization": azure_token
            }
            
            querystring = {"parent_document_id": document_id}
            delete_url = f"{self.document_uploader_url}/{self.app_id}"
            
            async with httpx.AsyncClient(timeout=self.timeout, verify=False) as client:
                response = await client.delete(delete_url, headers=headers, params=querystring)
                response.raise_for_status()
                return response.status_code == 202
                
        except Exception as e:
            print(f"Document deletion failed: {e}")
            return False
