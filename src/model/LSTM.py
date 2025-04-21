# libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import random

# LSTM decoder with random initial hidden state
class LSTMLyrics_by_Genre(nn.Module):
    # constructor parameters
    def __init__(self, vocab_size, embed_size, hidden_size, genre_embed_size, num_layers=1, bidirectional=False, teacher_forcing_ratio=0.5):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.genre_embedding = nn.Embedding(num_genres, genre_embed_size) #for genre embedding

        self.lstm = nn.LSTM(input_size = embed_size + genre_embed_size, 
                            hidden_size = hidden_size,
                            num_layers = num_layers,
                            bidirectional = bidirectional,
                            batch_first = True)

        self.fc = nn.Linear(hidden_size * self.num_directions, vocab_size)
        self.teacher_forcing_ratio = teacher_forcing_ratio
        # self.hidden_1 = nn.Linear(lstm_hidden_size * 2, lstm_hidden_size)
        # self.hidden_2 = nn.Linear(lstm_hidden_size, 1)

#forward pass def
    def forward(self, lyrics_input, genre_input, targets=None):
        batch_size, seq_len = lyrics_input.size()

        # 1. Embed lyrics (shape: [batch_size, seq_len, embed_size])
        lyric_embeds = self.embedding(lyrics_input)

        # 2. Embed genre (shape: [batch_size, genre_embed_size])
        genre_embeds = self.genre_embedding(genre_input)

        # 3. Repeat genre embedding for each timestep and concatenate
        genre_embeds = genre_embeds.unsqueeze(1).repeat(1, seq_len, 1)
        combined_input = torch.cat((lyric_embeds, genre_embeds), dim=2)
        # combined_input shape: [batch_size, seq_len, embed_size + genre_embed_size]

        # 4. Initialize hidden state (random for now)
        h_0 = torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_size).to(lyrics_input.device)
        c_0 = torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_size).to(lyrics_input.device)

        # 5. Run through LSTM
        lstm_out, _ = self.lstm(combined_input, (h_0, c_0))
        # lstm_out shape: [batch_size, seq_len, hidden_size * num_directions]

        # 6. Pass each timestep's output through final FC layer
        output_logits = self.fc(lstm_out)
        # output_logits shape: [batch_size, seq_len, vocab_size] batch first

        return output_logits

# training
    def train(model, dataloader, optimizer, criterion, device):
        model.train()

        for batch in dataloader:
            inputs, targets, genres = batch
            inputs = inputs.to(device)         # shape: [B, T]
            targets = targets.to(device)       # shape: [B, T]
            genres = genres.to(device)         # shape: [B]

            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs, genres)    # shape: [B, T, V]

            # Flatten the predictions and targets for loss calc
            outputs = outputs.view(-1, outputs.shape[-1])   # [B*T, V]
            targets = targets.view(-1)                      # [B*T]

            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
