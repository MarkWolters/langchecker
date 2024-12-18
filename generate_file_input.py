import argparse
import os
import json
import logging
from astrapy import DataAPIClient

def read_body_blob_and_extract_results(table_name, keyspace, api_endpoint, environment, output_file, results_column):
    client = DataAPIClient(os.environ["ASTRA_DB_APPLICATION_TOKEN"], environment=environment)
    db = client.get_database_by_api_endpoint(api_endpoint)
    logging.info(db.info())

    coll = db.get_collection(table_name, keyspace=keyspace)
    try:
        with open(output_file, 'w') as file:
            file.write("{\n\"results\": [\n")
            first_entry = True
            for (i, doc) in enumerate(coll.find()):
                try:
                    body_blob = doc.get(results_column)
                    if body_blob:
                        json_data = json.loads(body_blob)
                        try:
                            json_content = json.loads(json_data["data"]["content"])
                            results = json_content["results"]
                            if not first_entry:  # Write a comma before subsequent entries
                                file.write(",\n")
                            file.write(json.dumps(results, indent=4))
                            first_entry = False
                        except KeyError:
                            logging.error("Expected JSON structure not found in row:", body_blob)
                    else:
                        logging.error("No results found in row:", doc)
                except Exception as e:
                    logging.error("Error processing row:", doc, e)
            file.write("\n]\n}")
            logging.info(f"Data successfully written to {output_file}.")
    except Exception as e:
        logging.error("Error writing data to file:", e)

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Read data from a DataStax Astra table.")
    parser.add_argument('--table_name', required=True, help="The name of the table to read from.")
    parser.add_argument('--keyspace', required=True, help="The keyspace where the table is located.")
    parser.add_argument('--api_endpoint', required=True, help="Data API endpoint to use.")
    parser.add_argument('--environment', required=True, help="The environment to use.")
    parser.add_argument('--output_file', required=True, help="Where to write the extracted results.")
    parser.add_argument('--results_column', required=True, help="The column containing the json results.")

    args = parser.parse_args()

    # Call the function with parsed arguments
    read_body_blob_and_extract_results(
        table_name=args.table_name,
        keyspace=args.keyspace,
        api_endpoint=args.api_endpoint,
        environment=args.environment,
        output_file=args.output_file,
        results_column=args.results_column
    )

if __name__ == "__main__":
    main()
