from configurations.params import DOC_TAGGING
from source.conf_scores import CONFIDENCE_SCORE_TOOLS
from source.conf_scores.confidence_score import (
    ConfidenceScoreResponseConsistency,
    compute_confidence_score,
)
from source.conf_scores.logprobs import LogProbs
from source.content_moderation import (
    non_blocking_docSearch_evaluation,
    non_blocking_que_moderation,
    non_blocking_response_evaluation,
)
from source.log_handler import logger
from source.ragClient import ask_rag, exchangeTok_patch, exchangeToken
from source.utils import (
    EntityValue,
    ExtractedEntities,
    extract_json,
    extract_output,
    json_parser,
    table_parser,
)
# Import the new BR parser
from source.br_parser import br_post_extraction_processing

################################

headers = {"accept": "application/json", "Content-Type": "multipart/form-data"}


def classify_document(document, url=DOC_TAGGING, headers=headers):
    try:
        print("requesting doc tag")
        data = {"file": open(document, "rb")}
        response = requests.post(url, files=data, verify=False)
        print("doc tag response", response)
        if response.status_code == 201:
            response = response.json()
            doc_tag = response["doctype"]
            time_taken = response["time_taken"]
            print("file name", document, "doc tag", doc_tag, "time taken", time_taken)
            # TODO: log doc tag response in db or in log
            return response
        else:
            return {"message": "error in document tagging"}
    except Exception as e:
        logger.exception(e)
        return None


async def get_extraction(
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
    try:
        repeate_requests = 1
        for mth in CONFIDENCE_SCORE_TOOLS.methods:
            if isinstance(mth, ConfidenceScoreResponseConsistency):
                repeate_requests = mth.number_of_repeates
                break
        assert (
            repeate_requests > 0
        ), f"There should be at least 1 request requested. Found {repeate_requests}"

        prompts_dict, File_Name = getPrompts(doc_type, prompt_store)
        task = []
        for _ in range(repeate_requests):
            task.append(
                run_batch_prompts(
                    prompt_data=prompts_dict,
                    session_id=session,
                    azure_token=azure_token,
                    imageBlobID=imageBlobID,
                    background_tasks=background_tasks,
                    confidence=confidence,
                    confidence_method=confidence_method,
                    multimodal=multimodal,
                    image_detail=image_detail,
                    search_type=search_type,
                    doc_type=doc_type,  # Pass doc_type to determine parser
                )
            )

        responses = await asyncio.gather(*task)
        answers, table_exchanges = [], []
        for resp in responses:
            answers.append(resp[0])
            table_exchanges.append(resp[1])

        extracted_entities = await compute_confidence_score(
            extracted_entities=answers,
            computation_tools=CONFIDENCE_SCORE_TOOLS,
            azure_token=azure_token,
        )

        json_output = extract_output(
            extracted_entities,
            table_exchange=table_exchanges[0],
            vector=session,
            source="DCREST AI",
            Title=File_Name,
        )
        return json_output
    except Exception as e:
        logger.exception(e)
        return None


async def run_batch_prompts(
    prompt_data,
    session_id,
    azure_token,
    imageBlobID,
    background_tasks,
    confidence: bool = True,
    confidence_method: Literal[
        "product", "average", "first", "sum", "weighted_avg"
    ] = "product",
    multimodal: bool = False,
    image_detail: Literal["low", "high"] = "low",
    search_type: Literal["SEARCH", "NO-SEARCH", "IMAGE"] = "SEARCH",
    doc_type: str = "",  # Add doc_type parameter
    postprocessing_fn: Callable[
        [
            Dict[str, Union[str, bool]],
            List[Dict[str, Union[str, List[Dict[str, Union[bytes, str, float]]]]]],
            Literal["product", "average", "first", "sum", "weighted_avg"],
        ],
        Union[ExtractedEntities, Any],
    ] = None,  # Make postprocessing_fn optional
) -> Union[ExtractedEntities, Any]:

    try:
        async with httpx.AsyncClient(timeout=None) as client:
            task = []
            for keys, items in prompt_data.items():

                if "system_prompt" in list(items.keys()):
                    system_prompt = items["system_prompt"]
                else:
                    system_prompt = "you are an intelligent AI assistant"

                task.append(
                    ask_rag(
                        client=client,
                        session=session_id,
                        azure_token=azure_token,
                        question=items["prompt"],
                        search_text=None,
                        system_prompt=system_prompt,
                        imageBlobID=imageBlobID,
                        model_type=items["model_type"],
                        temperature=items["temperature"],
                        logprobs=int(confidence),
                        multimodal=multimodal,
                        image_detail=image_detail,
                        search_type=search_type,
                    )
                )

            answers = await asyncio.gather(*task)

        # Determine which post-processing function to use based on doc_type
        if doc_type.upper() == "BR":
            # Use BR-specific parser
            processing_fn = br_post_extraction_processing
        else:
            # Use standard parser
            processing_fn = postprocessing_fn or post_extraction_processing
        
        return processing_fn(
            session_token=session_id,
            azure_token=azure_token,
            background_tasks=background_tasks,
            prompt_data=prompt_data,
            responses=answers,
            confidence_method=confidence_method,
        )
    except Exception as e:
        logger.exception(e)
        return None


def post_extraction_processing(
    session_token,
    azure_token,
    background_tasks,
    prompt_data,
    responses,
    confidence_method,
):
    """
    Standard post-processing function for non-BR documents.
    This is the existing implementation.
    """
    params_keys = []
    params_values = []
    param_table = []
    param_table_values = []
    table_exchange = []
    
    for items, response in zip(prompt_data.values(), responses):
        exchange_token = exchangeToken(session_token, azure_token, items["prompt"])
        answer = response["answer"]
        logprobs = response["logprobs"] if "logprobs" in response else None
        
        if "?" in answer:
            text = extract_json(answer)
            try:
                if items["table"] == True:
                    # print("----------------inside table parser------------------------")
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
                    # print("extract json output --->", text)
                    entity_keys, entity_values = json_parser(
                        text,
                        logprobs=logprobs,
                        confidence_method=confidence_method,
                        query=items["prompt"],
                        response=response,
                        exchange=exchange_token,
                    )
                    params_keys.extend(entity_keys)
                    params_values.extend(entity_values)
            except Exception as e:
                params_keys.append(items["entity_name"])
                logger.error(f"error in loading json: {e}")
                params_values.append(
                    EntityValue(
                        value=text,
                        confidence=None,
                        bbox=None,
                        query=items["prompt"],
                        answer=answer,
                        document_content=response["document_content"],
                        rag_similarity_maen=statistics.mean(
                            [doc["@search.score"] for doc in response["documents_used"]]
                        ),
                        exchange=exchange_token,
                        logprobs=logprobs,
                    )
                )
        else:
            if logprobs is None:
                params_keys.append(items["entity_name"])
                params_values.append(
                    EntityValue(
                        value=answer,
                        confidence=None,
                        bbox=None,
                        query=items["prompt"],
                        answer=answer,
                        document_content=response["document_content"],
                        rag_similarity_maen=statistics.mean(
                            [doc["@search.score"] for doc in response["documents_used"]]
                        ),
                        exchange=exchange_token,
                    )
                )
            else:
                continue
        
        try:
            compute_logprob = LogProbs()
            values_dict = {
                items["entity_name"]: EntityValue(
                    value=answer,
                    confidence=None,
                    bbox=None,
                    query=items["prompt"],
                    answer=answer,
                    document_content=response["document_content"],
                    rag_similarity_maen=statistics.mean(
                        [doc["@search.score"] for doc in response["documents_used"]]
                    ),
                    exchange=exchange_token,
                    logprobs=logprobs,
                )
            }
            values_dict = compute_logprob.compute_joint_logprobs(
                model_answer_dict=values_dict,
                logprobs=logprobs,
                method=confidence_method,
            )
            params_keys.append(list(values_dict.keys())[0])
            params_values.append(list(values_dict.values())[0])
        except Exception as e:
            logger.error(f"Error in computing logprobs for string response: {e}")

        background_tasks.add_task(
            exchangeTok_patch, azure_token, exchange_token, answer
        )
        # background_tasks.add_task(non_blocking_que_moderation, azure_token, exchange_token)
        background_tasks.add_task(
            non_blocking_docSearch_evaluation,
            azure_token,
            exchange_token,
            items["prompt"],
            answer,
            response["document_content"],
        )
        background_tasks.add_task(
            non_blocking_response_evaluation,
            azure_token,
            exchange_token,
            items["prompt"],
            answer,
            response["document_content"],
        )

    extracted_entities = ExtractedEntities(
        params_keys=params_keys,
        params_values=params_values,
        param_table=param_table,
        param_table_values=param_table_values,
    )
    return extracted_entities, table_exchange
