import logging
import argparse
import json
from argparse import RawTextHelpFormatter
import requests
from typing import Optional
import warnings
import numpy as np
import pandas as pd
import os
try:
    from langflow.load import upload_file
except ImportError:
    warnings.warn("Langflow provides a function to help you upload files to the flow. Please install langflow to use it.")
    upload_file = None

BASE_API_URL = os.getenv("BASE_API_URL", "http://127.0.0.1:7860")
FLOW_ID = os.getenv("FLOW_ID", "a4c8da6b-3eeb-44fb-ba84-ce25269b9096")
ENDPOINT = os.getenv("FLOW_ENDPOINT", "") # You can set a specific endpoint name in the flow settings

# You can tweak the flow by adding a tweaks dictionary
# e.g {"OpenAI-XXXXX": {"model_name": "gpt-4"}}
TWEAKS = {
    "ChatInput-yhtxo": {},
    "AstraDB-DeKzR": {},
    "OpenAIEmbeddings-Ogk96": {},
    "ParseData-oKvYV": {},
    "ParseData-NEnlp": {},
    "CustomComponent-Vonyf": {},
    "CombineText-mnlMF": {},
    "ParseData-ivxGY": {},
    "Prompt-D1N4g": {},
    "OpenAIModel-TMuXZ": {},
    "CombineText-ip0er": {},
    "CombineText-oeaYO": {},
    "CustomComponent-08EfX": {},
    "ParseData-btJZD": {},
    "TextOutput-F0ntr": {}
}

def run_flow(message: str,
             endpoint: str,
             output_type: str = "text",
             input_type: str = "chat",
             tweaks: Optional[dict] = None,
             api_key: Optional[str] = None) -> dict:
    """
    Run a flow with a given message and optional tweaks.

    :param api_key:
    :param input_type:
    :param output_type:
    :param message: The message to send to the flow
    :param endpoint: The ID or the endpoint name of the flow
    :param tweaks: Optional tweaks to customize the flow
    :return: The JSON response from the flow
    """
    api_url = f"{BASE_API_URL}/api/v1/run/{endpoint}"

    payload = {
        "input_value": message,
        "output_type": output_type,
        "input_type": input_type,
    }
    headers = None
    #if tweaks:
    #    payload["tweaks"] = tweaks
    if api_key:
        headers = {"x-api-key": api_key}
    payload = json.loads(json.dumps(payload))
    logging.debug(payload)
    response = requests.post(api_url, json=payload, headers=headers)
    return response.json()

def extract_passage_text(d):
    for key, value in d.items():
        if key == 'passage_text':
            return value
    return None

def dict_to_json_compatible(d):
    # Convert any ndarray to list within the dictionary
    for key, value in d.items():
        if isinstance(value, np.ndarray):
            d[key] = value.tolist()
    return json.dumps(d)

def iterate_rows_parquet(input_file_path: str,
                 query_column: str,
                 gt_column: str,
                 endpoint: str = FLOW_ID,
                 tweaks: Optional[dict] = None,
                 application_token:Optional[str] = None,
                 output_type: str = "text",
                 input_type: str = "chat"
                 ):

    # Read the file into a DataFrame
    df = pd.read_parquet(input_file_path)

    # Iterate over each row in the DataFrame
    for _, row in df.iterrows():
        query_value = row[query_column]
        answers_value = row[gt_column]
        concatenated_answers = ""
        for answer_str in answers_value:
            if concatenated_answers != "":
                concatenated_answers = concatenated_answers.join(",")
            concatenated_answers = concatenated_answers.join(answer_str)
        answers_value = concatenated_answers
        logging.debug(answers_value)

        # Create a JSON object with the specified format
        json_object = {
            "query": query_value,
            "gt_answer": answers_value
        }
        json_object = dict_to_json_compatible(json_object)

        try:
            response = run_flow(
                json_object,
                endpoint,
                output_type,
                input_type,
                tweaks,
                application_token
            )
            logging.info(json.dumps(response, indent=2))
        except Exception as e:
            logging.error(f"An error occurred: {e}")


def iterate_rows_json(input_file_path: str,
                      query_column: str,
                      gt_column: str,
                      endpoint: str = FLOW_ID,
                      tweaks: Optional[dict] = None,
                      application_token:Optional[str] = None,
                      output_type: str = "text",
                      input_type: str = "chat"
                      ):
    try:
        with open(input_file_path, 'r') as file:
            data = json.load(file)
            for record in data:
                if query_column in record and gt_column in record:
                    query_value = record[query_column]
                    answers_value = record[gt_column]
                    concatenated_answers = ""
                    for answer_str in answers_value:
                        if concatenated_answers != "":
                            concatenated_answers = concatenated_answers.join(",")
                        concatenated_answers = concatenated_answers.join(answer_str)
                    answers_value = concatenated_answers
                    logging.debug(answers_value)

                    # Create a JSON object with the specified format
                    json_object = {
                        "query": query_value,
                        "gt_answer": answers_value
                    }
                    json_object = dict_to_json_compatible(json_object)

                    try:
                        response = run_flow(
                            json_object,
                            endpoint,
                            output_type,
                            input_type,
                            tweaks,
                            application_token
                        )
                        logging.info(json.dumps(response, indent=2))
                    except Exception as e:
                        logging.error(f"An error occurred: {e}")
                else:
                    logging.error(f"Error: Missing columns {query_column} or {gt_column} in the input data")
    except FileNotFoundError:
        logging.error(f"Error: File not found at {input_file_path}")
    except json.JSONDecodeError:
        logging.error("Error: Failed to decode JSON. Ensure the file contains valid JSON.")


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Process a parquet dataset", formatter_class=RawTextHelpFormatter)
    parser.add_argument("--input_file_path", type=str, help="Path to the input file containing the input data")
    parser.add_argument("--query_column", type=str, default="query", help="the column containing the LLM query")
    parser.add_argument("--gt_column", type=str, default="answers", help="the column containing the Ground Truth response to the query")
    parser.add_argument("--endpoint", type=str, default=ENDPOINT or FLOW_ID, help="The ID or the endpoint name of the flow")
    parser.add_argument("--tweaks", type=str, help="JSON string representing the tweaks to customize the flow", default=json.dumps(TWEAKS))
    parser.add_argument("--api_key", type=str, default=None, help="Application Token for authentication")
    parser.add_argument("--output_type", type=str, default="text", help="The output type")
    parser.add_argument("--input_type", type=str, default="chat", help="The input type")
    parser.add_argument("--upload_file", type=str, help="Path to the file to upload", default=None)
    parser.add_argument("--components", type=str, help="Components to upload the file to", default=None)
    parser.add_argument("--format", type=str, help="input file format", default="parquet")
    args = parser.parse_args()

    # Configure logging with a specific level and format
    logging.basicConfig(
        level=getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper()),
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    try:
        tweaks = json.loads(args.tweaks)
    except json.JSONDecodeError:
        raise ValueError("Invalid tweaks JSON string")

    if args.upload_file:
        if not upload_file:
            raise ImportError("Langflow is not installed. Please install it to use the upload_file function.")
        elif not args.components:
            raise ValueError("You need to provide the components to upload the file to.")
        tweaks = upload_file(file_path=args.upload_file, host=BASE_API_URL, flow_id=ENDPOINT, components=args.components, tweaks=tweaks)

    # Run the function with the specified file path and column names
    if args.format == "parquet":
        iterate_rows_parquet(args.input_file_path, args.query_column, args.gt_column, args.endpoint, args.tweaks, args.api_key, args.output_type, args.input_type)
    elif args.format == "json":
        iterate_rows_json(args.input_file_path, args.query_column, args.gt_column, args.endpoint, args.tweaks, args.api_key, args.output_type, args.input_type)
    else:
        raise ValueError(f"Unsupported file format: {args.format}")

if __name__ == "__main__":
    main()
