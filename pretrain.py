import os
import time
import gc
from collections import OrderedDict
from copy import deepcopy
from datetime import datetime
import pickle
import subprocess as sp
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertForPreTraining, BertConfig, BertTokenizer
from transformers.tokenization_utils_base import BatchEncoding

def get_gpu_memory():
    command = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = sp.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
    memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
    return memory_free_values

# paths
proj_dir = os.path.join('/home', 'ddegenaro', 'CAMeMBERT')
def in_proj_dir(dir):
    return os.path.join(proj_dir, dir)
pretraining_test = in_proj_dir('pretraining_test.txt')
pretraining_txt = in_proj_dir('pretraining.txt')
inits = in_proj_dir('inits')
ckpts = in_proj_dir('ckpts')
trained = in_proj_dir('trained')

# initialize teacher model mBERT
print('Getting mBERT.')
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
# this line will complain that decoder bias was not in the checkpoint
mBERT = BertForPreTraining.from_pretrained("bert-base-multilingual-cased")
print('Done.')

# hyperparameters
batch_size = 256
print('Getting corpus. Warning: requires ~50GB RAM.')
corpus = open(pretraining_txt).readlines() # change to pretraining_test to test on small data
length = len(corpus)
print(f'Done. Loaded {length} lines = {length / batch_size} batches.')
MSELoss = torch.nn.MSELoss() # loss between logits of two models
lr = 1e-4 # learning rate
max_length = 128 # maximum input sequence length - subword tokenization
steps_per_ta = int(400000 / 6) # training steps for each student
dropout = 0.1 # dropout probability
weight_decay = 0.01
epochs = 1
device = 'cuda:0'

class BertData(Dataset):
    def __init__(self, ta_num):    
        start_idx = ta_num * steps_per_ta * batch_size
        end_idx = start_idx + steps_per_ta * batch_size + 1
        self.lines = corpus[start_idx:end_idx]
        self.length = len(self.lines)
        print(f'TA: {ta_num}. Loaded lines {start_idx}-{end_idx} ({self.length} lines = {self.length / batch_size} batches.)')
    def __len__(self):
        return self.length
    def __getitem__(self, idx):
        return tokenizer(self.lines[idx], max_length=max_length, padding='max_length', truncation=True, return_tensors='pt')

teacher = mBERT # first network to copy from

for ta_num in reversed(range(6, 12)): # TA builder loop
    
    batch_indices = []
    losses = []

    data_loader = DataLoader(BertData(ta_num=ta_num), batch_size=batch_size, num_workers=8, pin_memory=True)

    teacher_state_dict = teacher.state_dict()

    # create a BertConfig with a multilingual vocabulary for the next TA
    config_obj = BertConfig(vocab_size=119547, num_hidden_layers=ta_num, hidden_dropout_prob=dropout, attention_probs_dropout_prob=dropout)

    student = BertForPreTraining(config_obj) # initialize next model and state dict
    student_state_dict = OrderedDict()

    torch.cuda.empty_cache()

    teacher.to(device) # use GPU
    student.to(device)

    print('Building student.')
    for key in teacher_state_dict: # copy architecture and weights besides top layer
        if str(ta_num) not in key:
            student_state_dict[key] = deepcopy(teacher_state_dict[key])
  
    # freeze embeddings
    for name, param in student.named_parameters():
        if 'embed' in name:
            param.requires_grad = False
    print('Done.')

    # save init for this TA
    print('Saving student.')
    torch.save(student, os.path.join(inits, 'ta' + str(ta_num) + '.bin'))
    print('Done.')

    # load next state dict into the next model
    student.load_state_dict(student_state_dict)

    student.train() # ensure training mode

    # generate Adam optimizer close to mBERT's
    optimizer = torch.optim.Adam(student.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-6, weight_decay=weight_decay)

    optimizer.zero_grad(set_to_none=True) # just to be sure

    with torch.set_grad_enabled(True):

        for epoch in range(epochs):

            loss = 0 # initialize useful loop variables
            start = datetime.now()

            print(f'Begin epoch {epoch+1}/{epochs}. Current time: {datetime.now()}.')

            for batch_idx, inputs in enumerate(data_loader): # each iter of this loop is one batch
        
                for j in inputs: # unwrap extra []
                    inputs[j] = inputs[j][0]

                inputs.to(device)
        
                # get logits to compare
                teacher_logits = teacher(**inputs).prediction_logits[0]
                student_logits = student(**inputs).prediction_logits[0]

                # calculate the loss between them and update
                loss = MSELoss(teacher_logits, student_logits)
        
                # learning step
                loss.backward()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if batch_idx % 1000 == 0:
                    print(f'Batches completed: {batch_idx}. Average time per batch: {(datetime.now()-start)/(batch_idx+1)}. Loss: {loss}.')
                    batch_indices.append(batch_idx)
                    losses.append(loss)
                    if batch_idx >= steps_per_ta: # done with this TA
                        break
      
        # save checkpoint if doing multiple epochs
        print('Saving checkpoint.')
        torch.save(student, os.path.join(ckpts, 'ta' + str(ta_num) + '_ckpt' + str(epoch) + '.bin'))
        print('Done.')

    # save trained model for this TA
    print('Saving trained student.')
    torch.save(student, os.path.join(trained, 'ta' + str(ta_num) + '.bin'))
    print('Done.')

    # summary statistics
    plt.figure(0)
    batch_indices = np.array(batch_indices)
    losses = np.array(losses)
    plt.plot(batch_indices, losses)   
    plt.title(f'Pretraining loss. Max loss: batch {np.argmax(batch_indices)}. Min loss: batch {np.argmin(batch_indices)}.')
    plt.xlabel('Batch index')
    plt.ylabel('Loss')
    plt.savefig(os.path.join(trained, 'pretraining_loss_' + ta_num), format='png')

    teacher = student # prepare to initialize next network

# end for
