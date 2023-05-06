import torch
import torch.nn as nn
import torch.nn.functional as F

class RelationNetwork(nn.Module):
    def __init__(self, embedding_dim, hidden_dim):
        super(RelationNetwork, self).__init__()
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Transformer encoder
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=4)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=2)
        
        # MLP layers for processing relations
        self.mlp1 = nn.Linear(2 * embedding_dim, hidden_dim)
        self.mlp2 = nn.Linear(hidden_dim, 1)
        
    def forward(self, inputs):
        # Embed input tokens
        embeddings = self.embedding(inputs)
        
        # Pass embeddings through transformer encoder
        encoded = self.encoder(embeddings)
        
        # Compute all pairwise combinations of encoded tokens
        pairs = []
        for i in range(encoded.shape[0]):
            for j in range(i+1, encoded.shape[0]):
                pairs.append(torch.cat([encoded[i], encoded[j]]))
        pairs = torch.stack(pairs)
        
        # Pass pairs through MLP layers
        x = F.relu(self.mlp1(pairs))
        x = self.mlp2(x)
        
        # Reshape output into a matrix
        n = encoded.shape[0]
        out = x.view(n, n)
        
        return out
