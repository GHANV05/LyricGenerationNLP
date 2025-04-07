from transformers import BertTokenizer, BertForSequenceClassification, AdamW, BertConfig
import numpy as np
import torch
from torch.utils.data import TensorDataset, random_split,  DataLoader, RandomSampler, SequentialSampler

#setup tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

#import data
song_lyrics = np.array([])
song_labels = np.array([])
num_genres = 0

#find longest song length (lyrically)
max_len = 0
for song in song_lyrics:
    input_ids = tokenizer.encode(song, add_special_tokens=True)
    max_len = max(max_len, len(input_ids))
print('Max Song Length: ', max_len)

#tokenize sentances and map tokens to wordID
input_ids = []
attention_masks = []

for song in song_lyrics:
    encoded_dict = tokenizer.encode_plus(
        song,                               #the song to encode
        add_special_tokens = True,          #prepend '[CLS]' and '[SEP]' to start and end of song (respectively)
        max_length = max_len,               #pad and truncate all songs to a length of max_len
        pad_to_max_length = True,
        return_attention_mask = True,       #construct attention masks
        return_tensors = 'pt',              #return pytorch tensors
    )
    input_ids.append(encoded_dict['input_ids'])
    attention_masks.append(encoded_dict['attention_mask'])


#convert lists into torch tensors
input_ids = torch.cat(input_ids, dim=0)
attention_masks = torch.cat(attention_masks, dim=0)
labels = torch.tensor(song_labels)

#Combine into tensor dataset
dataset = TensorDataset(input_ids, attention_masks, labels)


#Split data into training and validation (90/10)
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])


##Create Dataloaders to save memory during training, so data doesnt have to be loaded to sample
batch_size = 32

train_dataloader = DataLoader(
    train_dataset,
    sampler = RandomSampler(train_dataset), ##We want to train randomly on data
    batch_size=batch_size
)

validation_dataloader = DataLoader(
    val_dataset,
    sampler = SequentialSampler(val_dataset), ##For Validation, order doesnt matter
    batch_size=batch_size
)

#Load BertForSequenceClassification (base bert with a linear layer for classification)
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased", ##12 Layer Bert Model, uncased Vocab
    num_labels = num_genres,
    output_attentions = False,
    output_hidden_states = False,

)