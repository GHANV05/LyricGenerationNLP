import torch
import torch.nn as nn

# Encoder Baseline Class
# Process an input sequence and convert it into a fixed-size internal representation (hidden & cell state).
# Representation will be pased to Decoder to generate a target sequence

class Encoder(nn.Module):
    """
    Initializes the Encoder module.
    
    Args:
        input_dim (int): Size of the input vocabulary (number of unique tokens).
        emb_dim (int): Size of the word embeddings.
        hid_dim (int): Number of features in the hidden state of the LSTM.
        dropout (float): Dropout rate to prevent overfitting.
    """
    
    def __init(self, input_dim, emb_dim, hid_dim, dropout):
        super().__init__()

        # Embedding layer: maps token IDs to dense vectors of size emb_dim
        self.embedding = nn.Embedding(input_dim, emb_dim)

        # LSTM layer: processes embedded tokens sequentially
        self.rnn = nn.LSTM(emb_dim, hid_dim)

        # Dropout: applied to embeddings to reduce overfitting
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        """
        Forward pass of the encoder.

        Args:
            src (Tensor): Input sequence tensor of shape [src_len, batch_size], 
                          where each element is a token ID.

        Returns:
            hidden (Tensor): Final hidden state of the LSTM [1, batch_size, hid_dim]
            cell (Tensor): Final cell state of the LSTM [1, batch_size, hid_dim]
        """
        # src: [src_len, batch_size]
        # Example src prompt: 
        # src = torch.tensor([[3, 17, 42], [12, 5, 9], [7, 3, 1]])  # Shape: [src_len=3, batch_size=3]

        
        # Step 1: Embed the input sequence and apply dropout
        # embedded: [src_len, batch_size, emb_dim]
        embedded = self.dropout(self.embedding(src)) # [src_len, batch_size, emb_dim]

        # Step 2: Pass through the LSTM
        # outputs: [src_len, batch_size, hid_dim] (not used here)
        # hidden: [1, batch_size, hid_dim] — final hidden state
        # cell: [1, batch_size, hid_dim] — final cell state
        outputs, (hidden, cell) = self.rnn(embedded)

        # Step 3: Return hidden and cell state to be used by the decoder
        return hidden, cell

# Decoder Baseline Class
# Takes hidden and cell states from encoder and usese them to
# generate a sequence of output tokens one at a time

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, dropout):
        """
        Intializes the Decoder module

        Args:
            output_dim (int_: Size of the ouput vocabulary
            emb_dim (int): Size of the token embeddings
            hid_dim (int): Number of features in the hidden state o the LSTM
            dropout (float): Dropout rate to prevent overfitting
        """
        super().__init__()

        # Embedding layer: turns token indices into dense vectors
        self.embedding = nn.Embedding(output_dim, emb_dim)

        # LSTM layer : takes embedded input + previous hidden/cell state and returns new states
        self.rnn = nn.LSTM(emb_dim, output_dim)

        # Lienar output layer: maps LSTM output to vocab space
        self.fc_out = nn.Linear(hid_dim, output_dim)

        # Dropout to reduce overfitting
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_token, hidden, cell):
        """
        Forward pass of the decoder for one time step

        Args:
            input_token (Tensor): The current input token [batch_size]
            hidden (Tensor): The hidden state from the previous time step [1, batch_size, hid_dim]
            cell (Tensor): The cell sate from the previous time step [1, batch_size, hid_dim]

        Returns:
            prediction (Tensor): Logits for the next token prediciton [batch_size, output_dim]
            hidden (Tensor): Updated hidden state
            cell (Tensor): Updated cell state
        """
        # Step 1: Add time dimension to input token for embedding
        # input_token: [batch_size] --> [1, batch_size]
        input_token = input_token.unsqueeze(0)

        # Step 2: Embed the input token and apply dropout
        # embeddedd: [1, batch_size, emb_dim]
        embedded = self.dropout(self.embedding(input_token))

        # Step 3: Run the LSTM for one step with the embedded input
        # output: [1, batch_size, hid_dim]
        output, (hidden, cell) = self.rnn(embedded, (hidden,cell))

        # Step 4: Remove time dimension and apply linear layer to get vocab logits
        # prediction: [batch_size, output_dim]
        prediction = self.fc_out(output.squeeze(0))

        return prediction, hidden cell

