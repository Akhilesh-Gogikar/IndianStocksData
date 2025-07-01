import torch
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import re

class DataProcessor:
    def __init__(self):
        # Initialize Vectorizer for text columns
        self.vectorizer = TfidfVectorizer(max_features=1000)
        
        # Define numerical columns
        self.numerical_columns = [
            '3mAvgVol', '4wpct', '52wHigh', '52wLow', '52wpct', 'beta', 'bps', 
            'divYield', 'eps', 'inddy', 'indpb', 'indpe', 'marketCap', 'mrktCapRank', 
            'pb', 'pe', 'roe', 'nShareholders', 'lastPrice', 'ttmPe', '12mVol', 
            'mrktCapf', 'apef', 'pbr', 'etfLiq', 'financial_year_*_revenue', 
            'financial_year_*_profit', 'financial_statement_*_incTrev', 
            'financial_statement_*_incCrev', 'financial_statement_*_incGpro', 
            'financial_statement_*_incOpc', 'financial_statement_*_incRaw', 
            'financial_statement_*_incPfc', 'financial_statement_*_incEpc', 
            'financial_statement_*_incSga', 'financial_statement_*_incOpe', 
            'financial_statement_*_incEbi', 'financial_statement_*_incDep', 
            'financial_statement_*_incPbi', 'financial_statement_*_incIoi', 
            'financial_statement_*_incPbt', 'financial_statement_*_incToi', 
            'financial_statement_*_incNinc', 'financial_statement_*_incEps', 
            'financial_statement_*_incDps', 'financial_statement_*_incPyr'
        ]

        # Expand the numerical columns
        self.expanded_numerical_columns = self._expand_columns(self.numerical_columns)

        self.num_scaler = StandardScaler()
        # Add a variable to track if the scaler has been fitted
        self._num_scaler_fitted = False
        self._initialized = False

    def _expand_columns(self, columns_list):
        # Expand columns with wildcard patterns like 'financial_year_*_revenue'
        expanded_columns = []
        for col in columns_list:
            if '*' in col:
                # Adjust the range as per your data (e.g., 1 to 5)
                for i in range(1, 6):
                    expanded_columns.append(col.replace('*', str(i)))
            else:
                expanded_columns.append(col)
        return expanded_columns


    def _initialize_vectorizer_and_encoders(self, data_frame):
        # Combine text columns for BoW vectorizer initialization
        text_columns = [
            'headline', 'summary', 'feed_type', 'publisher', 'tag',
            'name', 'description', 'sector', 'gic_sector', 
            'gic_industrygroup', 'gic_industry', 'gic_subindustry', 'gic_short',
            'share_holding_*_title', 'share_holding_*_message', 'share_holding_*_description', 
            'share_holding_*_mood', 'commentary_*_title', 'commentary_*_message', 
            'commentary_*_description', 'commentary_*_mood', 
            'holdings_commentary_*_*_title', 'holdings_commentary_*_*_message', 
            'holdings_commentary_*_*_description', 'holdings_commentary_*_*_mood', 
            'financial_statement_*_reporting', 'financial_statement_*_description',
            'marketCapLabel', 'etfLiqLabel', 'peer_*_sector', 'peer_*_name'
        ]
        
        # Concatenate text columns for vectorization
        text_columns = [col for col in text_columns if col in data_frame.columns]  # Ensure columns exist in DataFrame
        data_frame['combined_text'] = data_frame[text_columns].fillna('').astype(str).agg(' '.join, axis=1)
        self.vectorizer.fit(data_frame['combined_text'])

        # Fit the numerical scaler
        num_columns = [col for col in self.expanded_numerical_columns if col in data_frame.columns]
        if num_columns:
            self.num_scaler.fit(data_frame[num_columns].fillna(0))
            self._num_scaler_fitted = True
        self._initialized = True

    def process_row(self, data_row):
        # Check if 'next_day_return' is null
        if pd.isnull(data_row['next_day_return']):
            #print(f"Skipping row due to null next_day_return: {data_row}")
            return None

        if not self._initialized:
            raise RuntimeError("DataProcessor not initialized. Please call _initialize_vectorizer_and_encoders first.")
        
        # Expand and concatenate text columns
        text_columns = [
            'headline', 'summary', 'feed_type', 'publisher', 'tag',
            'name', 'description', 'sector', 'gic_sector', 
            'gic_industrygroup', 'gic_industry', 'gic_subindustry', 'gic_short',
            'share_holding_*_title', 'share_holding_*_message', 'share_holding_*_description', 
            'share_holding_*_mood', 'commentary_*_title', 'commentary_*_message', 
            'commentary_*_description', 'commentary_*_mood', 
            'holdings_commentary_*_*_title', 'holdings_commentary_*_*_message', 
            'holdings_commentary_*_*_description', 'holdings_commentary_*_*_mood', 
            'financial_statement_*_reporting', 'financial_statement_*_description',
            'marketCapLabel', 'etfLiqLabel', 'peer_*_sector', 'peer_*_name'
        ]

        expanded_columns = []
        for col in text_columns:
            if re.search(r'_\*_', col):
                for i in range(1, 6):
                    expanded_columns.append(col.replace('*', str(i)))
            else:
                expanded_columns.append(col)
        
        expanded_columns = [col for col in expanded_columns if col in data_row.index]  # Ensure columns exist in DataRow
        text_data = ' '.join(str(data_row[col]) if pd.notnull(data_row[col]) else '' for col in expanded_columns)
        bow_vector = self.vectorizer.transform([text_data]).toarray()[0]
        bow_vector = torch.tensor(bow_vector, dtype=torch.float32).unsqueeze(0)  # Shape: [1, bow_features]


        # Process numerical data
        num_data = torch.zeros((1, len(self.expanded_numerical_columns)), dtype=torch.float32)
        num_values = []
        for idx, col in enumerate(self.expanded_numerical_columns):
            if col in data_row.index and pd.notnull(data_row[col]):
                num_values.append(float(data_row[col]))
            else:
                num_values.append(0.0)
        if self._num_scaler_fitted:
            num_values = self.num_scaler.transform([num_values])[0]
        num_data[0, :] = torch.tensor(num_values, dtype=torch.float32)

        # Target variable
        target_value = data_row['next_day_return'] if 'next_day_return' in data_row.index else 0
        if pd.notnull(target_value):
            # Clip to a reasonable range, e.g., between -100% and +100%
            target_value = max(min(target_value, 100.0), -100.0)
        else:
            target_value = 0.0
        target = torch.tensor(target_value, dtype=torch.float32)

        return bow_vector, num_data, target

    def process_dataframe(self, data_frame):
        if not self._initialized:
            self._initialize_vectorizer_and_encoders(data_frame)
        
        features_list, targets_list = [], []
        for _, row in data_frame.iterrows():
            features, target = self.process_row(row)
            features_list.append(features)
            targets_list.append(target)
        
        return torch.stack(features_list), torch.tensor(targets_list, dtype=torch.float32)

    def get_input_size(self):
        # Use the actual vocabulary size
        bow_size = len(self.vectorizer.get_feature_names())
        num_size = len(self.expanded_numerical_columns)
        return bow_size + num_size


