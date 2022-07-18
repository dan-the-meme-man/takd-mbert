import os
import torch
from datetime import datetime
from torch import nn
from transformers import logging
from transformers import BertConfig, BertForPreTraining, BertTokenizer
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

class FFN(nn.Module):
    def __init__(self):
        super(FFN, self).__init__()
        self.linear = torch.nn.Sequential(torch.nn.LazyLinear(3)) # project to 3 classes
    def forward(self, x):
        x = torch.reshape(x, (1, max_length*119547)) # flatten logits of model
        return self.linear(x)

def probs(sentence1, sentence2):
    inputs = tokenizer(sentence1, sentence2, return_tensors='pt', padding='max_length', max_length=max_length, truncation=True)
    inputs.to(device)

    with torch.no_grad():
        bert_outputs = model(**inputs).prediction_logits

    with torch.set_grad_enabled(True):
        final_outputs = ffn(bert_outputs)

    return final_outputs

for ta_num in reversed(range(6, 12)): # fine-tune all the TAs
    
    print('Getting model.')
    model_path = os.path.join(trained, 'ta' + str(ta_num) + '.bin')
    model = torch.load(model_path)
    model.train() # ensure training mode
    for param in model.parameters(): # do not compute gradients for base model
        param.requires_grad = False
    model.to(device) # put to GPU
    print('Done.')

    ffn = FFN() # create fine-tuning layer
    ffn.train() # ensure training mode
    ffn.to(device) # put to GPU
    
    # optimizer for the fine-tuning layer
    optimizer = torch.optim.Adam(ffn.parameters(), lr=lr, betas=(0.9, 0.999), eps=lr/100, weight_decay=0)
    optimizer.zero_grad(set_to_none=True) # ensure grad = 0 at start
    
    batch_indices = []
    losses = []

    with torch.set_grad_enabled(True): # calculate gradients during fine-tuning

        for j in range(epochs):
            
            loss = 0 # initialize useful loop variables
            i = 0
            start = datetime.now()
            print(f'Fine-tuning TA {ta_num}. Begin epoch {j}:', start)

            for example in train_data: # go through all fine-tuning examples (English XNLI)
                premise = example['premise'] # get premise
                hypothesis = example['hypothesis'] # get hypothesis
                
                label = torch.tensor(example['label'], device=device) # get true label
                pred = probs(premise, hypothesis)[0] # get model prediction logits

                loss += CELoss(pred, label) / batch_size # add to loss, norm by batch size
                i += 1

                if i % batch_size == 0: # at the end of the batch
                    print(f'Batches completed: {int(i / batch_size)}. Average time per batch: {(datetime.now() - start) / i}. Loss: {loss}.')
                    batch_indices.append(i)
                    losses.append(loss)
                    loss.backward() # backprop
                    optimizer.step() # update params
                    optimizer.zero_grad(set_to_none=True) # reset grad
                    loss = 0 # reset loss

    print('Saving FFN.')
    torch.save(ffn, os.path.join(linears, 'ta' + str(ta_num) + '.bin'))
    print('Done.')

    # summary statistics
    plt.figure(0)
    batch_indices = np.array(batch_indices)
    losses = np.array(losses)
    plt.plot(batch_indices, losses)   
    plt.title(f'Fine-tuning loss. Max loss: batch {np.argmax(batch_indices)}. Min loss: batch {np.argmin(batch_indices)}.')
    plt.xlabel('Batch index')
    plt.ylabel('Loss')
    plt.savefig(os.path.join(linears, 'finetune_loss_' + ta_num), format='png')
