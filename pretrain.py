import os
from collections import OrderedDict
from copy import deepcopy
from datetime import datetime
import subprocess as sp
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertForPreTraining, BertConfig, BertTokenizer, get_scheduler

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
mBERT = BertForPreTraining.from_pretrained('bert-base-multilingual-cased')
print('Done.')

# hyperparameters
batch_size = 256
training_data_file = pretraining_txt # can change to pretraining_test for smaller dataset
MSELoss = torch.nn.MSELoss() # loss between logits of two models
lr = 2e-5 # learning rate
max_length = 128 # maximum input sequence length - subword tokenization
steps_per_ta = 10000 # training steps for each student - be careful not to exceed bounds!
num_warmup_steps = 1000
dropout = 0.1 # dropout probability
weight_decay = 0.01
epochs = 1 # can make training lengthy - change at your own risk!
device = 'cuda:0' # GPU needed!
num_workers = 8 # processes to load data during training - about 2-8 per GPU is reasonable

log_freq = int(steps_per_ta / 1000) if steps_per_ta >= 1000 else 1

print(f'Getting corpus. Requires ~{(os.path.getsize(training_data_file) / 1e9):.2f} GB of RAM.')
corpus = open(training_data_file).readlines() # change to pretraining_test to test on small data
print(f'Done.')

class BertData(Dataset): # needed by data loader
    def __init__(self, ta_num):
        start_idx = (ta_num - 6) * steps_per_ta * batch_size
        end_idx = start_idx + steps_per_ta * batch_size
        self.lines = corpus[start_idx:end_idx]
        self.length = len(self.lines)
        print(f'TA: {ta_num}. Loaded lines {start_idx}-{end_idx} ({self.length} lines = {int(self.length / batch_size)} batches.)')
    def __len__(self):
        return self.length
    def __getitem__(self, idx):
        return tokenizer(self.lines[idx], max_length=max_length, padding='max_length', truncation=True, return_tensors='pt')

teacher = mBERT # first network to copy from

for ta_num in reversed(range(6, 12)): # TA builder loop
    
    batch_indices = []
    losses = []

    data_loader = DataLoader(BertData(ta_num=ta_num), batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    
    torch.cuda.empty_cache()

    teacher_state_dict = teacher.state_dict()
    teacher.to(device)
    teacher.zero_grad()

    print('Building student.')
    # create a BertConfig with a multilingual vocabulary for the next TA, initialize student
    student = BertForPreTraining(BertConfig(vocab_size=119547, num_hidden_layers=ta_num, hidden_dropout_prob=dropout, attention_probs_dropout_prob=dropout))
    
    student_state_dict = OrderedDict()
    for key in teacher_state_dict: # copy architecture and weights besides top layer
        if str(ta_num) not in key:
            student_state_dict[key] = deepcopy(teacher_state_dict[key])
    student.load_state_dict(student_state_dict) # load weights/architecture
    
    for name, param in student.named_parameters(): # freeze embeddings
        if 'embed' in name:
            param.requires_grad = False
    
    student.train() # ensure training mode
    student.to(device)
    student.zero_grad()
    print('Done.')

    # save init for this TA
    print('Saving student.')
    torch.save(student, os.path.join(inits, 'ta_' + str(ta_num) + '.bin'))
    print('Done.')

    # generate Adam optimizer close to mBERT's
    optimizer = torch.optim.Adam(student.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-6, weight_decay=weight_decay)
    optimizer.zero_grad(set_to_none=True) # just to be surei
    lr_scheduler = get_scheduler(name="linear", optimizer=optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=steps_per_ta)
    with torch.set_grad_enabled(True):

        for epoch in range(epochs):

            loss = 0 # initialize loss
            start = datetime.now()

            print(f'Begin epoch {epoch+1}/{epochs}. Current time: {datetime.now()}.')

            for batch_idx, inputs in enumerate(data_loader): # each iter of this loop is one batch
        
                for j in inputs: # unwrap extra []
                    inputs[j] = inputs[j][0]

                inputs.to(device)
        
                # get logits to compare
                teacher_logits = teacher(**inputs).prediction_logits
                student_logits = student(**inputs).prediction_logits

                # calculate the loss between them and update
                loss = MSELoss(teacher_logits, student_logits)
        
                # learning step
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                batch_indices.append(batch_idx)
                losses.append(loss.to('cpu'))

                # logging
                if batch_idx % log_freq == 0:
                    avg_time = (datetime.now() - start) / (batch_idx + 1)
                    eta = datetime.now() + (steps_per_ta - batch_idx + (epochs - epoch - 1 + epochs * (ta_num - 6)) * steps_per_ta) * avg_time
                    print(f'Batches completed: {batch_idx + 1:6}/{steps_per_ta}. Average time per batch: {avg_time}. ETA: {eta}. Loss: {loss:.4}.')
        
        # save checkpoint if doing multiple epochs
        print('Saving checkpoint.')
        torch.save(student, os.path.join(ckpts, 'ta_' + str(ta_num) + '_ckpt_' + str(epoch) + '.bin'))
        print('Done.')

    # save trained model for this TA
    print(f'Saving trained student. Training took: {datetime.now() - start}.')
    torch.save(student, os.path.join(trained, 'ta_' + str(ta_num) + '.bin'))
    print('Done.')

    # summary statistics
    plt.figure(0)
    batch_indices = np.array(batch_indices)
    losses = (torch.tensor(losses)).detach().numpy()
    plt.plot(batch_indices, losses)
    plt.title(f'Pretraining loss. Max loss: batch {np.argmax(losses)}. Min loss: batch {np.argmin(losses)}.')
    plt.xlabel('Batch index')
    plt.ylabel('Loss')
    graph_loc = os.path.join(trained, 'pretraining_loss_' + str(ta_num) + '.png')
    plt.savefig(graph_loc, format='png')
    print(f'Saved loss graph to {graph_loc}.')

    teacher = student # prepare to initialize next network

# end for
