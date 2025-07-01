import torch
import torch.nn as nn
import torch.nn.functional as F

class MLModel(nn.Module):
    def __init__(self, text_input_size, num_input_size, embedding_dim=16, num_layers=2):
        super(MLModel, self).__init__()
        
        # Text data processing
        self.text_linear = nn.Linear(text_input_size, embedding_dim)
        
        # Numerical data processing
        self.num_linear = nn.Linear(num_input_size, embedding_dim)
        
        # Batch normalization on the combined embedding
        self.combined_bn = nn.BatchNorm1d(embedding_dim)
        
        # Transformer encoder with customizable layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim, 
            nhead=2, 
            dim_feedforward=32, 
            dropout=0.05,        
            activation='relu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output layer
        self.output_linear = nn.Linear(embedding_dim, 1)
            
    def forward(self, text_input, num_input):
        # Text embedding
        text_embedded = F.relu(self.text_linear(text_input))  # Shape: (batch_size, embedding_dim)
        
        # Numerical embedding
        num_embedded = F.relu(self.num_linear(num_input))  # Shape: (batch_size, embedding_dim)
        
        # Remove any extra dimensions if they exist
        text_embedded = text_embedded.view(text_embedded.size(0), -1)  # Ensure 2D: (batch_size, embedding_dim)
        num_embedded = num_embedded.view(num_embedded.size(0), -1)  # Ensure 2D: (batch_size, embedding_dim)
        
        # Stack embeddings along a new dimension
        combined_embedded = torch.stack([text_embedded, num_embedded], dim=1)  # Shape: (batch_size, 2, embedding_dim)
        
        # Reshape for BatchNorm1d
        batch_size, seq_len, emb_dim = combined_embedded.shape
        combined_embedded = combined_embedded.view(batch_size * seq_len, emb_dim)
        
        # Batch normalization
        combined_embedded = self.combined_bn(combined_embedded)
        
        # Reshape back to (batch_size, seq_len, embedding_dim)
        combined_embedded = combined_embedded.view(batch_size, seq_len, emb_dim)
        
        # Transformer encoder
        combined_embedded = combined_embedded.permute(1, 0, 2)  # Shape: (seq_len, batch_size, embedding_dim)
        transformer_output = self.transformer_encoder(combined_embedded)
        transformer_output = transformer_output.permute(1, 0, 2)  # Shape: (batch_size, seq_len, embedding_dim)
        
        # Aggregate the output
        aggregated_output = transformer_output.mean(dim=1)  # Shape: (batch_size, embedding_dim)
        
        # Final output
        output = self.output_linear(aggregated_output)
        return output.squeeze()




