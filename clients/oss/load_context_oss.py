import argparse
import json
import logging
from argparse import RawTextHelpFormatter
import requests
from typing import Optional
import warnings
import os
import pandas as pd
try:
    from langflow.load import upload_file
except ImportError:
    warnings.warn("Langflow provides a function to help you upload files to the flow. Please install langflow to use it.")
    upload_file = None

BASE_API_URL = os.getenv("BASE_API_URL", "http://127.0.0.1:7860")
FLOW_ID = os.getenv("FLOW_ID", "d0461e9c-9cc7-4305-b2db-86d3f4786c01")
ENDPOINT = os.getenv("FLOW_ENDPOINT", "context") # The endpoint name of the flow

# You can tweak the flow by adding a tweaks dictionary
# e.g {"OpenAI-XXXXX": {"model_name": "gpt-4"}}
TWEAKS = {
    "TextInput-sY41S": {},
    "MessagetoData-iqECO": {},
    "SplitText-z4ecM": {},
    "AstraDB-r0UJG": {},
    "OpenAIEmbeddings-gkjVl": {}
}

def run_flow(message: str,
             endpoint: str,
             output_type: str,
             input_type: str,
             tweaks: Optional[dict],
             api_key: Optional[str]) -> dict:
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
    if tweaks:
        payload["tweaks"] = tweaks
    if api_key:
        headers = {"x-api-key": api_key}
    response = requests.post(api_url, json=payload, headers=headers)
    return response.json()

def extract_passage_text(d, subfield):
    for key, value in d.items():
        if key == subfield:
            return value
    return None

def iterate_context_parquet(context_file_path, context_column, endpoint, tweaks, api_key, output_type, input_type, context_subfield):
    # Read the file into a DataFrame
    df = pd.read_parquet(context_file_path)

    # Iterate over each row in the DataFrame
    for _, row in df.iterrows():
        passages_val = extract_passage_text(row[context_column], context_subfield)
        logging.debug(passages_val)
        for item in passages_val:
            logging.debug(item)
            try:
                response = run_flow(
                    message=item,
                    endpoint=endpoint,
                    output_type=output_type,
                    input_type=input_type,
                    tweaks=tweaks,
                    api_key=api_key
                )
                logging.info(json.dumps(response, indent=2))
            except Exception as e:
                logging.error(f"An error occurred: {e}")

def iterate_context_json(context_file_path, context_column, endpoint, tweaks, api_key, output_type, input_type):
    try:
        with open(context_file_path, 'r') as file:
            data = json.load(file)
        for record in data:
            if context_column in record:
                logging.debug(f"Contexts for record ID {record['id']}:")
                for context in record[context_column]:
                    response = run_flow(
                        message=context,
                        endpoint=endpoint,
                        output_type=output_type,
                        input_type=input_type,
                        tweaks=tweaks,
                        api_key=api_key
                    )
                    logging.info(json.dumps(response, indent=2))
    except FileNotFoundError:
        logging.error(f"Error: File not found at {context_file_path}")
    except json.JSONDecodeError:
        logging.error("Error: Failed to decode JSON. Ensure the file contains valid JSON.")

def main():
    parser = argparse.ArgumentParser(description="Process a context file", formatter_class=RawTextHelpFormatter)
    parser.add_argument("--context_file_path", type=str, default=".", help="Path to the input file containing the contextual data")
    parser.add_argument("--context_column", type=str, default="passages", help="The column containing the context data")
    parser.add_argument("--context_subfield", type=str, default="passage_text", help="The subfield containing the context data")
    parser.add_argument("--endpoint", type=str, default=ENDPOINT or FLOW_ID, help="The ID or the endpoint name of the flow")
    parser.add_argument("--tweaks", type=str, help="JSON string representing the tweaks to customize the flow", default=json.dumps(TWEAKS))
    parser.add_argument("--api_key", type=str, help="API key for authentication", default=None)
    parser.add_argument("--output_type", type=str, default="text", help="The output type")
    parser.add_argument("--input_type", type=str, default="text", help="The input type")
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
        tweaks = upload_file(file_path=args.upload_file, host=BASE_API_URL, flow_id=args.endpoint, components=[args.components], tweaks=tweaks)

    # Run the function with the specified file path and column names
    if args.format == "parquet":
        iterate_context_parquet(args.context_file_path, args.context_column, args.endpoint, tweaks, args.api_key, args.output_type, args.input_type, args.context_subfield)
    elif args.format == "json":
        iterate_context_json(args.context_file_path, args.context_column, args.endpoint, tweaks, args.api_key, args.output_type, args.input_type)
    else:
        raise ValueError(f"Unsupported file format: {args.format}")

if __name__ == "__main__":
    main()
