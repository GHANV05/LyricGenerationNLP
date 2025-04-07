from transformers import BertTokenizer, BertForSequenceClassification, AdamW, BertConfig, get_linear_schedule_with_warmup
import numpy as np
import torch
from torch.utils.data import TensorDataset, random_split,  DataLoader, RandomSampler, SequentialSampler
import time
import datetime
import random
import pandas as pd

##HYPERPARAMETERS
batch_size = 32
learning_rate = 2e-5
epochs = 5

device = '' ##CONNECT TO GPU HERE

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

#setup optimizer
optimizer = AdamW(
    model.parameters(),
    lr = learning_rate,
    eps = 1e-8
)

#calculate steps for training total
total_steps = len(train_dataloader) * epochs

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps = 0,
    num_training_steps = total_steps
)

#takes in a list of predicted genres, and the labels that are in the dataset
#returns the proportion of examples we labelled correctly
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

#convert elapsed seconds into minutes
def format_time(elapsed):
    elapsed_round = int(round(elapsed))
    return str(datetime.timedelta(seconds=elapsed_round))

#information on training
training_stats = []

#start time for training
total_t0 = time.time()

#training loop
for epoch_i in range(0, epochs):
    print("")
    print('========== Epoch {:} / {:} =========='.format(epoch_i+1, epochs))
    print('Training...')

    #set start time for this epoch
    t0 = time.time()

    #reset training loss
    total_train_loss = 0

    #put model into training mode
    model.train()

    #loop over batches
    for step, batch in enumerate(train_dataloader):
        
        #report on progress
        if step % 40 == 0 and not step == 0:
            elapsed = format_time(time.time()-t0)

            print('Batch {:>5,} of {:>5,}. Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

        #unpack training batch from dataloader
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

        #clear gradients before calculating new ones
        model.zero_grad()

        #do forward pass, return loss and logits
        loss, logits = model(
            b_input_ids,
            token_type_ids = None,
            attention_mask = b_input_mask,
            labels=b_labels
        )

        #accumulate training loss
        total_train_loss += loss.item()

        #do backward pass to get gradients
        loss.backward()
        
        ##Clip gradients to fix exploding gradient problem
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        #update parameters and take a step using new gradients
        optimizer.step()
        
        #update the learning rate
        scheduler.step()

    avg_train_loss = total_train_loss / len(train_dataloader)

    training_time = format_time(time.time()-t0)

    print("")
    print("Average Training Loss: {0:.2f}".format(avg_train_loss))
    print("Training Epoch took : {:}".format(training_time))

    print("")
    print("Running Validation...")

    t0 = time.time()

    model.eval()

    total_eval_accuracy = 0
    total_eval_loss = 0
    nb_eval_steps = 0

    for batch in validation_dataloader:

        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

        with torch.no_grad():
            (loss, logits) = model(
                b_input_ids,
                token_type_ids = None,
                attention_mask = b_input_mask,
                labels = b_labels
            )
        
        total_eval_loss += loss.item

        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        total_eval_accuracy += flat_accuracy(logits, label_ids)

    avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
    print(" Accuracy: {0:.2f}".format(avg_val_accuracy))

    avg_val_loss = total_eval_loss / len(validation_dataloader)

    validation_time = format_time(time.time()-t0)

    print(" Validation Loss: {0:.2f}".format(avg_val_loss))
    print(" Validation took: {:}".format(validation_time))

    training_stats.append(
        {
            'epoch': epoch_i + 1,
            'Training Loss': avg_train_loss,
            'Valid. Loss': avg_val_loss,
            'Valid. Accur.': avg_val_accuracy,
            'Training Time': training_time,
            'Valid. Time': validation_time
        }
    )

print("")
print("Training Complete!")
print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))

pd.set_option('precision', 2)

df_stats = pd.DataFrame(data=training_stats)
df_stats = df_stats.set_index('epoch')

df_stats

