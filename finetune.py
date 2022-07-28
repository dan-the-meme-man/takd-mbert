import os
import torch
from datetime import datetime
from transformers import logging
from transformers import BertConfig, BertTokenizer, BertForSequenceClassification
from datasets import load_dataset
import numpy as np
import matplotlib.pyplot as plt

# stops console spam
logging.set_verbosity_error()

# paths
proj_dir = os.path.join('/home', 'ddegenaro', 'CAMeMBERT')
trained = os.path.join(proj_dir, 'trained')
finetuned = os.path.join(proj_dir, 'finetuned_small_lr')

# hyperparameters
epochs = 3
batch_size = 32
max_length = 128 # maximum input sequence length
lr = 1e-7 # learning rate
device = 'cuda:0'
vocab_size = 119547 # DO NOT MODIFY 
steps_per_ta = int(392702 / batch_size) # DO NOT MODIFY
hundred_batch = batch_size * 100 # for logging

print('Getting tokenizer.')
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case=False)
print('Done.')

print('Getting fine-tuning data.') # same fine-tuning data used by Jiao et al. 2021
train_data = load_dataset('xnli', split='train', language='en').shuffle(seed=69)
print('Done.')

CELoss = torch.nn.CrossEntropyLoss()

for ta_num in reversed(range(6, 12)): # fine-tune all the TAs
    
    print('Getting model.')
    model_path = os.path.join(trained, f'ta_{ta_num}.bin')
    pretrained_model = torch.load(model_path) # model without finetuning layer
    model = BertForSequenceClassification(BertConfig(num_hidden_layers=ta_num, vocab_size=vocab_size, num_labels=3))
    model.load_state_dict(pretrained_model.state_dict(), strict=False) # load weights, randomly initialize fine-tuning layer
    model.train() # ensure training mode
    model.to(device) # put to GPU
    model.zero_grad()

    # freeze embeddings
    for name, param in model.named_parameters():
        if 'embed' in name:
            param.requires_grad = False
    print('Done.')
    
    # optimizer for fine-tuning the underlying model
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=lr/100, weight_decay=0)
    optimizer.zero_grad(set_to_none=True) # ensure grad = 0 at start

    # to make plot of loss
    batch_indices = []
    losses = []

    with torch.set_grad_enabled(True): # calculate gradients during fine-tuning

        for epoch in range(epochs):
            
            loss = 0 # initialize useful loop variables
            i = 0
            start = datetime.now()
            print(f'Fine-tuning TA {ta_num}. Begin epoch {epoch + 1}/{epochs}:', start)

            for example in train_data: # go through all fine-tuning examples (English XNLI)
                
                premise = example['premise']
                hypothesis = example['hypothesis']

                inputs = tokenizer(premise, hypothesis, return_tensors='pt', max_length=max_length, padding='max_length', truncation=True)
                inputs.to(device)

                with torch.set_grad_enabled(True):
                    preds = model(**inputs).logits[0]
                
                label = torch.tensor(example['label'], device=device) # get true label
                
                loss += CELoss(preds, label) / batch_size # add to loss, norm by batch size
                i += 1

                if i % batch_size == 0: # at the end of the batch
                    batch_idx = int(i / batch_size)
                    avg_time = (datetime.now() - start) / (batch_idx + 1)
                    eta = datetime.now() + (steps_per_ta - batch_idx + (epochs - epoch - 1 + epochs * (ta_num - 6)) * steps_per_ta) * avg_time
                    if i % hundred_batch == 0:    
                        print(f'Batches completed: {batch_idx:5}/{steps_per_ta}. Average time per batch: {avg_time}. ETA: {eta}. Loss: {loss}.')
                    batch_indices.append(batch_idx + epochs * steps_per_ta)
                    losses.append(loss)
                    loss.backward() # backprop
                    optimizer.step() # update params
                    optimizer.zero_grad(set_to_none=True) # reset gradients
                    loss = 0 # reset loss

    print(f'Saving finetuned model. Training took: {datetime.now() - start}.')
    model_save_path = os.path.join(finetuned, f'ta_{ta_num}.bin')
    torch.save(model, model_save_path)
    print(f'Saved model to {model_save_path}.')
    
    # summary statistics
    plt.figure()
    batch_indices = np.array(batch_indices)
    losses = (torch.tensor(losses)).detach().numpy()
    plt.scatter(batch_indices, losses, label=str(ta_num), s=2)
    plt.title(f'Fine-tuning loss. Max loss: batch {np.argmax(losses)}. Min loss: batch {np.argmin(losses)}.')
    plt.xlabel('Batch index')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    graph_loc = os.path.join(finetuned, f'finetune_loss_{ta_num}.png')
    plt.savefig(graph_loc, format='png')
    print(f'Saved loss graph to {graph_loc}.')
