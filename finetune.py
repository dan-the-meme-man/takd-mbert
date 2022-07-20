import os
import torch
from datetime import datetime
from torch import nn
from transformers import logging
from transformers import BertTokenizer
from datasets import load_dataset
import numpy as np
import matplotlib.pyplot as plt

# stops console spam
logging.set_verbosity_error()

# paths
proj_dir = os.path.join('/home', 'ddegenaro', 'CAMeMBERT')
trained = os.path.join(proj_dir, 'trained')
linears = os.path.join(proj_dir, 'linears')

print('Getting tokenizer.')
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case=False)
print('Done.')

print('Getting fine-tuning data.') # same fine-tuning data used by Jiao et al. 2021
train_data = load_dataset('xnli', split='train', language='en').shuffle(seed=69)
print('Done.')

CELoss = torch.nn.CrossEntropyLoss()

# hyperparameters as defined by Jiao et al. 2021
epochs = 3
batch_size = 32
max_length = 128
lr = 2e-5
device = 'cuda:1'
vocab_size = 119547

length = max_length * vocab_size

class FFN(nn.Module):
    def __init__(self):
        super(FFN, self).__init__()
        self.linear = torch.nn.Linear(length, 3) # project to 3 classes
    def forward(self, x):
        x = torch.reshape(x, (1, length)) # flatten logits of model
        return self.linear(x)

def probs(sentence1, sentence2):
    inputs = tokenizer(sentence1, sentence2, return_tensors='pt', padding='max_length', max_length=max_length, truncation=True)
    inputs.to(device)

    with torch.set_grad_enabled(True):
        bert_outputs = model(**inputs).prediction_logits
        final_outputs = ffn(bert_outputs) 
    return final_outputs

steps_per_ta = 122414 / batch_size

for ta_num in reversed(range(6, 12)): # fine-tune all the TAs
    
    print('Getting model.')
    model_path = os.path.join(trained, 'ta_' + str(ta_num) + '.bin')
    model = torch.load(model_path)
    model.train() # ensure training mode
    for param in model.parameters(): # do not compute gradients for base model
        param.requires_grad = False
    model.to(device) # put to GPU
    model.zero_grad()

    # freeze embeddings
    for name, param in model.named_parameters():
        if 'embed' in name:
            param.requires_grad = False
    print('Done.')
    
    ffn = FFN() # create fine-tuning layer
    ffn.train() # ensure training mode
    ffn.to(device) # put to GPU
    ffn.zero_grad()
    
    # optimizer for fine-tuning the underlying model
    optimizer_b = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=lr/100, weight_decay=0)
    optimizer_b.zero_grad(set_to_none=True) # ensure grad = 0 at start

    # optimizer for the fine-tuning layer
    optimizer_f = torch.optim.Adam(ffn.parameters(), lr=lr, betas=(0.9, 0.999), eps=lr/100, weight_decay=0)
    optimizer_f.zero_grad(set_to_none=True) # ensure grad = 0 at start

    batch_indices = []
    losses = []

    with torch.set_grad_enabled(True): # calculate gradients during fine-tuning

        for epoch in range(epochs):
            
            loss = 0 # initialize useful loop variables
            i = 0
            start = datetime.now()
            print(f'Fine-tuning TA {ta_num}. Begin epoch {epoch + 1}/{epochs}:', start)

            for example in train_data: # go through all fine-tuning examples (English XNLI)
                premise = example['premise'] # get premise
                hypothesis = example['hypothesis'] # get hypothesis
                
                label = torch.tensor(example['label'], device=device) # get true label
                pred = probs(premise, hypothesis)[0] # get model prediction logits

                loss += CELoss(pred, label) / batch_size # add to loss, norm by batch size
                i += 1

                if i % batch_size == 0: # at the end of the batch
                    print()
                    print(pred, label)
                    batch_idx = int(i / batch_size)
                    avg_time = (datetime.now() - start) / (batch_idx + 1)
                    eta = datetime.now() + (steps_per_ta - batch_idx + (epochs - epoch - 1 + epochs * (ta_num - 6)) * steps_per_ta) * avg_time
                    print(f'Batches completed: {batch_idx:5}/122414. Average time per batch: {avg_time}. ETA: {eta}. Loss: {loss}.')
                    batch_indices.append(batch_idx)
                    losses.append(loss)
                    loss.backward() # backprop
                    optimizer_b.step() # update params
                    optimizer_f.step()
                    optimizer_b.zero_grad(set_to_none=True) # reset grad
                    optimizer_b.zero_grad(set_to_none=True)
                    loss = 0 # reset loss

    print('Saving FFN. Trained on {i} examples.')
    torch.save(ffn, os.path.join(linears, 'ta_' + str(ta_num) + '.bin'))
    print('Done.')
    
    # summary statistics
    plt.figure(0)
    batch_indices = np.array(batch_indices)
    losses = (torch.tensor(losses)).detach().numpy()
    plt.plot(batch_indices, losses)
    plt.title(f'Fine-tuning loss. Max loss: batch {np.argmax(losses)}. Min loss: batch {np.argmin(losses)}.')
    plt.xlabel('Batch index')
    plt.ylabel('Loss')
    graph_loc = os.path.join(linears, 'finetune_loss_' + str(ta_num) + '.png')
    plt.savefig(graph_loc, format='png')
    print(f'Saved loss graph to {graph_loc}.')
   
