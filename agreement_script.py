import csv
import re
from ollama_api import OllamaClient

# Initialize the Ollama client
client = OllamaClient(base_url='http://192.168.31.151:11434')

# Function to query the LLM
def query_llm(prompt, model="phi3.5"):
    try:
        response = client.generate_completion(model=model, prompt=prompt, stream=False)
        return response['response']
    except Exception as e:
        print(f"Error querying LLM: {e}")
        return None

# Function to extract the first number from the LLM's response
def extract_number(text):
    """Extracts the first number in the text."""
    match = re.search(r'[-+]?\d*\.\d+|\d+', text)
    if match:
        return float(match.group())
    else:
        return 0.0

# Function to create the prompt for the agreement score
def create_agreement_prompt(row):
    """
    Generate a prompt that asks the LLM to assess the stock condition and the predicted return,
    and return an agreement score between -10 and +10.
    """
    prompt = (
        f"Based on the following information about a stock in the Indian financial markets:\n\n"
        f"Valuation Score: {row['valuation_score']}\n"
        f"Growth Score: {row['growth_score']}\n"
        f"Momentum Score: {row['momentum_score']}\n"
        f"Peer Analysis Score: {row['peer_analysis_score']}\n"
        f"Sector Score: {row['sector_score']}\n"
        f"Market Sentiment Score: {row['market_sentiment_score']}\n"
        f"Headline Score: {row['headline_score']}\n\n"
        f"Predicted Next Day Return from an ML system: {row['predicted_next_day_return']}\n\n"
        f"Description: {row['description']}\n"
        f"Headline: {row['headline']}\n"
        f"Summary: {row['summary']}\n\n"
        f"In the context of the Indian financial markets and considering current economic conditions in India, "
        f"intelligently assess the condition of the stock and the predicted next day return from the ML system. "
        f"Provide an agreement score between -10 (strong disagreement) and +10 (strong agreement) on how strongly you agree with the predicted return. "
        f"Only return the score; do not include any text."
    )
    return prompt

def process_csv_with_agreement(input_csv, output_csv):
    """
    Process the CSV file, generate prompts for agreement scores, query the LLM,
    and save the results to a new output CSV file.
    """
    # Read the entire CSV into memory
    with open(input_csv, mode='r') as infile:
        reader = csv.DictReader(infile)
        rows = list(reader)

    with open(output_csv, mode='w', newline='') as outfile:
        # Dynamically generate fieldnames to include the agreement_score
        fieldnames = reader.fieldnames + ["agreement_score"]
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            try:
                print(f"Processing row: {row['ticker']}")

                # Generate the prompt based on the row
                prompt = create_agreement_prompt(row)

                # Query the LLM and capture the score
                raw_score = query_llm(prompt)
                try:
                    # Extract the number from the LLM's response
                    agreement_score = extract_number(raw_score) if raw_score is not None else 0.0
                except Exception as e:
                    print(f"Error extracting number: {e}")
                    agreement_score = 0.0

                # Append the agreement score to the row
                row.update({
                    "agreement_score": agreement_score
                })

                print(f"Processed row with agreement score: {row['agreement_score']}")
                writer.writerow(row)

            except Exception as e:
                print(f"Error processing row {row.get('ticker', 'unknown')}: {e}")
                # Optionally write incomplete rows with default agreement score
                row.update({"agreement_score": 0.0})
                writer.writerow(row)

    print(f"Processing complete. Agreement scores saved to {output_csv}")

# Run the script
if __name__ == "__main__":
    input_csv = "scored_data.csv"  # The CSV output from the previous script
    output_csv = "agreement_scored_data.csv"  # File where results will be saved
    process_csv_with_agreement(input_csv, output_csv)
