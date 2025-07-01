import pandas as pd

# Load the scored data
scored_data_file = "updated_data_with_predictions.csv"  # Replace with the path to your scored data file
scored_data = pd.read_csv(scored_data_file)

# scored_data_sorted = scored_data.sort_values(by="days_from_last")

# # Calculate the number of records corresponding to the top 5%
# top_5_percent_count = int(len(scored_data_sorted) * 0.05)

# # Select the top 5% of records with the lowest 'days_from_last' values
# top_5_percent_records = scored_data_sorted.head(top_5_percent_count)

# # Print the 'headline' and 'days_from_last' columns of these records
# headlines_and_days = top_5_percent_records[["headline", "days_from_last"]]

# # Display the result
# print(headlines_and_days)
print(scored_data['predicted_next_day_return'].describe())

