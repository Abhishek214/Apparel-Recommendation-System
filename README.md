# Only showing the updated get_entity endpoint - other endpoints remain the same

@ExtractionServer.post("/entity-extract", status_code=201)
async def get_entity(
    background_tasks: BackgroundTasks,
    DCREST_JWT_TOKEN: str = Header(),
    X_HSBC_Request_Correlation_Id: str = Header(),
    azure_token: str = Header(),
    DCREST_DOC_TYPE: str = Form(),
    DCREST_SESSION: str = Form(),
    imageBlobID: str = Form(default=None),
    prompt_store=Depends(get_prompt_store),
    confidence: bool = True,
    confidence_method: Literal[
        "product", "average", "first", "sum", "weighted_avg"
    ] = Form(default="product"),
    multimodal: bool = Form(default=False),
    image_detail: Literal["low", "high"] = Form(default="low"),
    search_type: Literal["SEARCH", "NO-SEARCH", "IMAGE"] = Form(default="SEARCH"),
):
    """
    Extract all entities from the given document.
    
    Args:
        DCREST_JWT_TOKEN: auth user token
        X_HSBC_Request_Correlation_Id: request correlation id
        azure_token: azure token for authentication to azure services
        DCREST_DOC_TYPE: document type (BR documents will use specialized parser)
        DCREST_SESSION: session id for which the document was uploaded. Only one active session can be there.
        imageBlobID: image id, if the request relates to only a single page of document
        prompt_store: prompts to be used for entities extraction
        confidence: whether to compute and return the response confidence estimation
        confidence_method: method to compute the confidence estimation, using log probs from LLM responses as likelihood estimates.
        multimodal: if true, the model will create image blobs of document pages along with the text part. It also enables multimodal processing of document.
        image_details: resolution of saved image blobs (image blobs for multimodal). Both or to have
        search_type: whether to search text-only, image-only, or both or to not search at all.
    
    Returns:
        extracted key-value pair and tables using provided prompts for the requested document type.
        For BR documents, complex nested structures are automatically converted to separate tables.
    """
    
    logger.info(f"Extraction {X_HSBC_Request_Correlation_Id=} for doc_type={DCREST_DOC_TYPE}")
    
    verify = auth(DCREST_JWT_TOKEN)
    
    try:
        output_json = await get_extraction(
            doc_type=DCREST_DOC_TYPE,  # Pass the document type
            session=DCREST_SESSION,
            azure_token=azure_token,
            imageBlobID=imageBlobID,
            prompt_store=prompt_store,
            confidence=confidence,
            confidence_method=confidence_method,
            multimodal=multimodal,
            image_detail=image_detail,
            search_type=search_type,
            background_tasks=background_tasks,
        )
        
        logger.info(f"Extraction {X_HSBC_Request_Correlation_Id=} complete for {DCREST_DOC_TYPE}...")
        return jsonable_encoder(output_json)
    except Exception as e:
        logger.error(f"Error in extraction for {DCREST_DOC_TYPE}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Sorry, Azure service seems down. please try again in sometime !",
        )
