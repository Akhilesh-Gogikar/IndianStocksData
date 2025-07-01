import csv
import json

from ollama_api import OllamaClient

client = OllamaClient(base_url='http://192.168.31.151:11434')

# Function to query the LLM
def query_llm(prompt, model="llama3.2"):
    try:
        response = client.generate_completion(model=model, prompt=prompt, stream=False)
        return response['response']
    except Exception as e:
        print(f"Error querying LLM: {e}")
        return None

# Function to create prompts for each column type
def create_prompts(row):
    """
    Generate a list of prompts tailored to the row's data, leveraging selected numeric, categorical, and text columns.
    """
    prompts = []

    # **Fundamental Factors**
    # Valuation, Growth, and Financial Performance
    prompts.append(
        f"Based on the following description: '{row['description']}', Considering the following fundamental factors: RSI: {row['rsi']}, Beta: {row['beta']}, PE: {row['pe']}, ROE: {row['roe']}, Dividend Yield: {row['divYield']}, PBR: {row['pbr']}, EPS: {row['eps']}, Revenue: {row['financial_year_1_revenue']}, and other available financial metrics, evaluate the valuation, growth, and financial performance of this stock. The stock belongs to the '{row['gic_sector']}' sector, '{row['gic_industry']}' industry, and '{row['gic_subindustry']}' sub-industry in the Indian financial markets. Provide a score from -10 (poor) to +10 (excellent). Only return the score; do not include any text."
    )

    # **Peer Analysis**
    # Relative Performance Compared to Peers
    prompts.append(
        f"Based on the following description: '{row['description']}', Considering the current stock's Market Cap = {row['marketCap']}, TTM PE = {row['ttmPe']}, and the Market Cap and TTM PE ratios of Peer 1: Market Cap = {row['peer_1_marketCap']}, TTM PE = {row['peer_1_ttmPe']}; Peer 2: Market Cap = {row['peer_2_marketCap']}, TTM PE = {row['peer_2_ttmPe']}; Peer 3: Market Cap = {row['peer_3_marketCap']}, TTM PE = {row['peer_3_ttmPe']}, evaluate the stock's relative performance compared to its peers. The stock belongs to the '{row['gic_sector']}' sector, '{row['gic_industry']}' industry, and '{row['gic_subindustry']}' sub-industry in the Indian financial markets. Provide a score from -10 to +10. Only return the score; do not include any text."
    )

    # **Market and Sentiment Factors**
    # Market, Industry, and Headline Sentiment Analysis
    prompts.append(
        f"Based on the following description: '{row['description']}', Given the stock is traded in the '{row['gic_sector']}' sector, '{row['gic_industry']}' industry, and '{row['gic_subindustry']}' sub-industry on the '{row['exchange']}' exchange, and considering the following summary: '{row['summary']}', and headline: '{row['headline']}', analyze the overall market, industry, and sentiment for investment in the context of the Indian financial markets. Rate the sentiment from -10 (very negative) to +10 (very positive). Only return the score; do not include any text."
    )

    return prompts


def process_csv(input_csv, output_csv):
    """
    Process the top and bottom 100 rows based on 'predicted_next_day_return' column,
    generate prompts, query the LLM for scores, and save the results to a new output CSV file.
    """
    # Read the entire CSV into memory
    with open(input_csv, mode='r') as infile:
        reader = csv.DictReader(infile)
        rows = list(reader)

    # Sort rows by 'predicted_next_day_return' (convert to float for sorting)
    sorted_rows = sorted(rows, key=lambda x: float(x['predicted_next_day_return']), reverse=True)

    # Get top 100 and bottom 100 rows
    top_100 = sorted_rows[:100]
    bottom_100 = sorted_rows[-100:]
    filtered_rows = top_100 + bottom_100

    with open(output_csv, mode='w', newline='') as outfile:
        # Dynamically generate fieldnames to include all score columns
        fieldnames = reader.fieldnames + [
            "fundamental_score",
            "peer_analysis_score",
            "market_sentiment_score"
        ]
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()

        for row in filtered_rows:
            try:
                print(f"Processing row: {row['ticker']}")

                # Generate prompts based on the row
                prompts = create_prompts(row)
                scores = []

                for prompt in prompts:
                    # Query the LLM and capture the score
                    raw_score = query_llm(prompt)
                    try:
                        # Normalize the score to a float, default to 0 if parsing fails
                        score = float(raw_score) if raw_score is not None else 0.0
                    except ValueError:
                        score = 0.0
                    scores.append(score)

                # Map scores to their corresponding fields
                row.update({
                    "fundamental_score": scores[0],
                    "peer_analysis_score": scores[1],
                    "market_sentiment_score": scores[2]
                })

                print(f"Processed row with scores: {row}")
                writer.writerow(row)

            except Exception as e:
                print(f"Error processing row {row.get('ticker', 'unknown')}: {e}")
                # Optionally write incomplete rows with default scores
                writer.writerow({**row, **{key: 0.0 for key in fieldnames if key not in row}})

    print(f"Processing complete. Scores saved to {output_csv}")


# Run the script
if __name__ == "__main__":
    input_csv = "updated_data_with_predictions.csv"  # Replace with the path to your CSV file
    output_csv = "scored_data.csv"  # File where results will be saved
    process_csv(input_csv, output_csv)
