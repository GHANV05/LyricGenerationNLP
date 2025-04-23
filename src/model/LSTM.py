# libraries
#pytorch is used for implementing the model
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import nltk
from nltk.tokenize import word_tokenize
from collections import Counter
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm

#tokenize the data
#nltk.download('punkt') #sentance tokenizer

class LyricsTokenizer:
    def __init__(self, min_freq=1, max_length=500):
        self.min_freq = min_freq #minimum freq to be in the vocab (really unique words removed)
        self.max_length = max_length #max length of song
        self.special_tokens = ['<PAD>', '<UNK>', '<EOS>', '<BOS>']
        self.word2id = {} #map words to token ids 
        self.id2word = {} #map ids back to words

    #build a vocab of unique words, filter rare words, and assign word ids
    def build_vocab(self, songs):
        #first make a clean list of token lists, lower case and split with word_tokenize
        tokenized = []
        for song in songs:
            song = song.lower()
            tokens = word_tokenize(song)
            tokenized.append(tokens)
        #list of allllll tokens not in individual lists
        all_tokens = []
        for song_tokens in tokenized:
            for token in song_tokens:
                all_tokens.append(token)
        #count for each token in the list
        freq = Counter(all_tokens)
        #remove tokens with 2 or less occurrances
        vocab = []
        for token, count in freq.items():
            if count > self.min_freq:
                vocab.append(token)
        #add special tokens 
        #(pad tokens for same length, unk for rare words, BOS start of lyric, eos end of lyric)
        full_vocab = self.special_tokens + sorted(vocab)
        self.vocab = full_vocab
        self.tokenized_songs = tokenized

        self.word2id = {} #map token to index
        self.id2word = {} #map index to token
        for idx in range(len(full_vocab)):
            token = full_vocab[idx]
            self.word2id[token] = idx
            self.id2word[idx] = token
        assert self.word2id['<PAD>'] == 0


    def tokenize(self, examples):
        example_ids = []
        misses = 0
        total = 0

        for example in examples:
            #clean the tokens
            example = example.lower()
            tokens = word_tokenize(example)
            ids = []

            for token in tokens:
                #check for word in dictionary
                if token in self.word2id:
                    ids.append(self.word2id[token])
                #else add to misses and add an unk char
                else:
                    misses += 1
                    ids.append(self.word2id['<UNK>'])
                total += 1 #count of all tokens
            #truncate ids if its too long
            if len(ids) >= self.max_length:
                ids = ids[:self.max_length]
                length = self.max_length
            #else add pad tokens to achieve max length
            else:
                length = len(ids)
                ids.extend([self.word2id['<PAD>']] * (self.max_length - len(ids)))
                
            #turn numerical ids into pytorch tensors
            tensor_ids = torch.tensor(ids, dtype=torch.long)
            #tuples of the real length of the song before padding, 
            example_ids.append((tensor_ids, length))
            
        print('Missed {} out of {} words -- {:.2f}%'.format(misses, total, 100 * misses / total))
        return example_ids

    def vocab_size(self):
        return len(self.word2id)

    
    def untokenize(self, token_ids):
        words = []
        for idx in token_ids:
            word = self.id2word.get(idx.item() if isinstance(idx, torch.Tensor) else idx, '<UNK>')
            # Skip padding
            if word == '<PAD>':
                continue
            words.append(word)
        return ' '.join(words)


#dataset design

class LyricsGenreDataset(Dataset):
    #initialize the dataset
    def __init__(self, tokenized_lyrics, genres, genre2idx):
        self.tokenized_lyrics = tokenized_lyrics  # list of (tensor, length) see above
        self.genre_ids = []

        for genre in genres:
            #transform the genre into a pytorch tensor
            genre_tensor = torch.tensor(genre2idx[genre], dtype=torch.long)
            #add to the list
            self.genre_ids.append(genre_tensor)
    #pass size of data set to pytorch
    def __len__(self):
        return len(self.tokenized_lyrics)
    #returns a specifiec data point as needed
    def __getitem__(self, idx):
        lyrics_tensor, length = self.tokenized_lyrics[idx]
        genre_id = self.genre_ids[idx]
        return lyrics_tensor, genre_id


#get a list of tuples (tensor, length, genre) and store them temporarily
def collate_fn(batch): 
    lyrics_batch = []
    genres = []

    for item in batch:
        lyrics_tensor, genre_id = item
        lyrics_batch.append(lyrics_tensor)
        genres.append(genre_id)
    #stack all the tensors into one tensor, (batch_size, seq_length)
    lyrics_batch = torch.stack(lyrics_batch)
    #make the genres one tensor (batch_size)
    genres_tensor = torch.stack(genres)
    #this is my teacher forcing in collation
    inputs = lyrics_batch[:, :-1] #all tokens except the last one
    targets = lyrics_batch[:, 1:] #all tokens except the first one
    #outputs 4 tensors created above
    return inputs, targets, genres_tensor


# LSTM decoder with zeros initial hidden state
class LSTMLyrics_by_Genre(nn.Module):
    # constructor parameters with nn.Module as base case
    def __init__(self, vocab_size, embed_dim, hidden_size, genre_embed_size, num_layers=1, bidirectional=False, teacher_forcing_ratio=0.5, num_genres=0):
        # vocab size = size of unique dictionary of tokens
        # embed dim = size of word embeddings
        # hidden size = size of hidden state
        # genre embed size = size of genre embedding vector
        # num layers = number of stacked lstm layers
        # not bidirectional = reads only forwad - could change this?
        # teacher forcing ratio = probability of using ground truth during training
        # num genres = how many genres - i think we have 10 
        super().__init__()
        #embedding vector for each token in vocab
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        #embedding vector for each genre
        self.genre_embedding = nn.Embedding(num_genres, genre_embed_size)
        #core lstm set up
        self.lstm = nn.LSTM(input_size = embed_dim + genre_embed_size, 
                            hidden_size = hidden_size,
                            num_layers = num_layers,
                            bidirectional = bidirectional,
                            batch_first = True)
        #some additional settings for the forward pass
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_directions = 2 if bidirectional else 1
        #linear layer for mapping outputs/hidden state to the dictionary
        self.fc = nn.Linear(hidden_size * self.num_directions, vocab_size)
        #how often should the model use the ground truth token instead of its own token
        self.teacher_forcing_ratio = teacher_forcing_ratio
        # trying out genre embedding hidden state beginnings
        self.genre_to_hidden = nn.Linear(genre_embed_size, self.num_layers * self.num_directions * self.hidden_size)
        self.genre_to_cell = nn.Linear(genre_embed_size, self.num_layers * self.num_directions * self.hidden_size)


#forward pass def
    def forward(self, lyrics_input, genre_input, targets=None):
        #targets are the ground truth indices for teacher forcing
        batch_size, seq_len = lyrics_input.size()
        #ensure that the code won't have issues with CPU and GPU mismatches
        device = lyrics_input.device 
        #tensor for each output at each timestep shape: (batch size, seq length, vocab size)
        outputs = torch.zeros(batch_size, seq_len, self.fc.out_features).to(device)

        # # Initial hidden state is all zeros for now
        # h_0 = torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_size).to(device)
        # c_0 = torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_size).to(device)
        # Embed genre into vectors
        genre_embeds = self.genre_embedding(genre_input)  # shape: [B, G]
        #unsqueeze to concatenated to lyrics at each timestep
        genre_embeds = genre_embeds.unsqueeze(1)  # shape: [B, 1, G]

        # genre based embedding changes
        hidden_flat = self.genre_to_hidden(genre_embeds.squeeze(1))  # [B, L*D*H]
        cell_flat = self.genre_to_cell(genre_embeds.squeeze(1))      # [B, L*D*H]

        # Reshape to match LSTM expected shape: [num_layers * num_directions, batch_size, hidden_size]
        h_0 = hidden_flat.view(self.num_layers * self.num_directions, batch_size, self.hidden_size).contiguous()
        c_0 = cell_flat.view(self.num_layers * self.num_directions, batch_size, self.hidden_size).contiguous()

        #first token from each lyric in batch
        input_token = lyrics_input[:, 0]  # start with first token (usually BOS)
        #hidden state is tuple for loop purposes
        hidden = (h_0, c_0)
        #loop through each timestep word by word
        for t in range(seq_len):
            #embed the input and add dim for lstm stds
            lyric_embed = self.embedding(input_token).unsqueeze(1)  # [B, 1, E]
            #already correct, occurs at each time step
            genre_expand = genre_embeds  # [B, 1, G]
            #concat step btween lyric and genre
            lstm_input = torch.cat([lyric_embed, genre_expand], dim=2)  # [B, 1, E+G]
            #in goes the concat input and hidden state, out comes the updated hidden and output
            lstm_out, hidden = self.lstm(lstm_input, hidden)  # [B, 1, H]
            #remove unnecesary dim, convert hidden state to logits
            output_logits = self.fc(lstm_out.squeeze(1))  # [B, V]
            #store predicted logits
            outputs[:, t, :] = output_logits
            #TEACHER FORCING: sometimes the model uses ground truth, otherwise prediction, based on supplied probability
            if targets is not None and random.random() < self.teacher_forcing_ratio:
                input_token = targets[:, t]  # Use ground truth
            else:
                input_token = output_logits.argmax(dim=1)  # Use model prediction
        #return all the logits for each position in sequence
        return outputs


# training
def train_model(model, dataloader, optimizer, criterion, device):
    model.train() #sets model into training mode
    total_loss = 0.0 #var to keep track of loss through whole run
    # model = LSTM duh
    # dataloader = batches of data, lyrics and genre in this case
    # optimizer = adam i think
    # criterion = loss function, cross entropy loss

    #loop through the dataloader to get a batch of data in the needed format
    for batch in tqdm(dataloader, desc="Training", leave=False):
        inputs, targets, genres = batch
        #stay in the right device to avoid errors!
        inputs, targets, genres = inputs.to(device), targets.to(device), genres.to(device)
        #clear gradients to avoid accumulation
        optimizer.zero_grad()
        #forward pass
        outputs = model(inputs, genres, targets)
        #reshape the outputS and target for cross entropy loss
        #outputs predictionf over vocab for time step
        outputs = outputs.view(-1, outputs.size(-1))  # [B*T, V]
        #vector of target indices
        targets = targets.reshape(-1)                 # [B*T]
        #compute the loss, difference btween predition and real
        loss = criterion(outputs, targets)
        #backpropogate the gradients
        loss.backward()
        #update model parameters
        optimizer.step()
        #add to loss for tracking
        total_loss += loss.item()

    return total_loss / len(dataloader)

def generate_lyrics(model, tokenizer, genre_str, genre2idx, max_len=100, device='cpu'):
    model.eval() #disable dropout and gradient tracking
    #convert genre to int and wrap in tensor
    genre_id = genre2idx[genre_str]
    genre_tensor = torch.tensor([genre_id], dtype=torch.long, device=device)
    #embed genre for concatting
    genre_embed = model.genre_embedding(genre_tensor).unsqueeze(1)
    #starts with BOS token
    input_token = torch.tensor([[tokenizer.word2id["<BOS>"]]], dtype=torch.long, device=device)

    #initialize hidden state
    hidden = (torch.zeros(1, 1, model.hidden_size).to(device),
              torch.zeros(1, 1, model.hidden_size).to(device))
    #list to store tokens
    generated = ["<BOS>"]
    
    #generation loop: up to max size
    for step in range(max_len):
        # #embed current token (BOS at first time step)
        # lyric_embed = model.embedding(input_token).unsqueeze(1)  # [1, 1, E]
        # lyric_embed = lyric_embed.squeeze(1) 
        # # print(f"lyric_embed shape: {lyric_embed.shape}")
        # # print(f"genre_embed shape: {genre_embed.shape}")

        # #concatts the genre embedding to the token
        # lstm_input = torch.cat([lyric_embed, genre_embed], dim=2)  # [1, 1, E+G]
        # #passes through the model and gets next state
        # output, hidden = model.lstm(lstm_input, hidden)
        # #converts into vector of vocab-size logits, remove timestep dim bcs we're doing one token at a time
        # logits = model.fc(output.squeeze(1))  # [1, V]
        # #uses greedy decoding and selects the index with the highest score
        # predicted_id = logits.argmax(dim=-1).item()
        # #transformes into a python int
        # word = tokenizer.id2word[predicted_id]

        token_embed = model.embedding(input_token)  # shape: (1, 1, embed_dim)

        # Concatenate genre embedding
        lstm_input = torch.cat((token_embed, genre_embed), dim=2)  # shape: (1, 1, embed_dim + genre_dim)

        output, hidden = model.lstm(lstm_input, hidden)
        logits = model.fc(output.squeeze(1))  # shape: (1, vocab_size)

        topk = torch.topk(logits, 5)
        top_probs = torch.nn.functional.softmax(topk.values, dim=-1)
        # print(f"Top tokens: {[tokenizer.id2word[i.item()] for i in topk.indices[0]]}")
        # print(f"Probs: {top_probs[0].tolist()}")

        #use softmax to get probabilities
        probs = torch.nn.functional.softmax(logits, dim=-1).squeeze()
        
        #sort the probabilities in descending order
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        
        #fin where prob exceeds threshold (like .9 or smth)
        top_p_threshold = 0.9
        cutoff = cumulative_probs > top_p_threshold
        if torch.any(cutoff):
            last_index = torch.where(cutoff)[0][0]
            sorted_probs = sorted_probs[:last_index + 1]
            sorted_indices = sorted_indices[:last_index + 1]
        
        #pick one from the filtered distribution
        sorted_probs = sorted_probs / sorted_probs.sum()  # re-normalize
        next_token_id = sorted_indices[torch.multinomial(sorted_probs, 1).item()].item()
        next_token = tokenizer.id2word[next_token_id]

        #debugging code
        # print(f"[Step {step}] Predicted token: {next_token}")
        # print("Decoded:", " ".join([tokenizer.id2word[token.item()] for token in tokenized_lyrics[0]]))

        #song is over at eos
        if next_token == "<EOS>":
            break
        if next_token == "<UNK>" or next_token == "<PAD>":
            continue
        #add the predicted word to the list
        generated.append(next_token)
    
        # prepare this to be the next input
        input_token = torch.tensor([[next_token_id]], dtype=torch.long, device=device)
    #return the final predictions
    return " ".join(generated[1:])  # remove <BOS> for clean output

