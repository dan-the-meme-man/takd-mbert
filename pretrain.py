import os
from collections import OrderedDict
from copy import deepcopy
from datetime import datetime
import subprocess as sp
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertForPreTraining, BertConfig, BertTokenizer, get_linear_schedule_with_warmup

# paths
proj_dir = os.path.join('/home', 'ddegenaro', 'CAMeMBERT')
def in_proj_dir(dir):
    return os.path.join(proj_dir, dir)
pretraining_test = in_proj_dir('pretraining_test.txt')
pretraining_txt = '/scratch/ddegenaro/pretraining.txt'
ckpts = in_proj_dir('ckpts')
trained = in_proj_dir('trained')

# set up plotting
plt.figure(0)
plt.title(f'Pretraining loss')
plt.xlabel('Batch index')
plt.ylabel('Loss')

# initialize teacher model mBERT
print('Getting mBERT.')
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
# this line will complain that decoder bias was not in the checkpoint
mBERT = BertForPreTraining.from_pretrained('bert-base-multilingual-cased', output_attentions=True, output_hidden_states=True)
print('Done.')

# hyperparameters
lr = 1e-7 # peak learning rate
training_data_file = pretraining_txt # can change to pretraining_test for smaller dataset
steps_per_ta = int(400000 / 6) # training steps for each student - be careful not to exceed bounds!
if training_data_file == pretraining_test:
    steps_per_ta = 30

device = 'cuda:0' # GPU needed!
MSELoss = torch.nn.MSELoss() # loss between logits of two models
batch_size = 256
max_length = 128 # maximum input sequence length - subword tokenization
dropout = 0.1 # dropout probability
weight_decay = 0.01
epochs = 1 # can make training lengthy - change at your own risk!
num_workers = 8 # processes to load data during training - about 2-8 per GPU is reasonable
log_freq = int(steps_per_ta / 500) if steps_per_ta >= 1000 else 1
eps = lr / 100
save_inits = False

print(f'Getting corpus. Requires ~{(os.path.getsize(training_data_file) / 1e9):.2f} GB of RAM.')
corpus = open(training_data_file).readlines() # change to pretraining_test to test on small data
print('Done.')

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
    
    num_warmup_steps = steps_per_ta if ta_num == 11 else int(steps_per_ta / 10)

    batch_indices = [] # for tracking loss over time
    losses = []
    
    # load a range of data from the corpus
    data_loader = DataLoader(BertData(ta_num=ta_num), batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    
    torch.cuda.empty_cache() # ensure GPU memory freed

    teacher.to(device) # move teacher to GPU, set training mode, ensure grads = 0
    teacher.train()
    teacher.zero_grad()

    print('Building student.')
    # create a BertConfig with a multilingual vocabulary for the next TA, initialize student
    student = BertForPreTraining(BertConfig(vocab_size=119547, num_hidden_layers=ta_num, num_attentions_heads=12, hidden_dropout_prob=dropout, attention_probs_dropout_prob=dropout, output_attentions=True, output_hidden_states=True))
    
    teacher_state_dict = teacher.state_dict() # build a state dict for the student
    student_state_dict = OrderedDict()
    assert type(student_state_dict) == type(teacher_state_dict)

    for key in teacher_state_dict: # copy architecture and weights besides top layer
        if str(ta_num) not in key:
            student_state_dict[key] = deepcopy(teacher_state_dict[key])
    
    for key in student_state_dict: # raise an error if something is incorrect
        assert student_state_dict[key].equal(teacher_state_dict[key])
        assert str(ta_num) not in key

    student.load_state_dict(student_state_dict) # load weights/architecture

    for name, param in student.named_parameters(): # freeze embeddings
        if 'embed' in name:
            param.requires_grad = False
    
    student.train() # ensure training mode
    student.to(device)
    student.zero_grad()
    print('Done.')
    
    if save_inits:
        # save init for this TA
        print('Saving student init.')
        torch.save(student, os.path.join(inits, f'ta_{ta_num}.bin'))
        print('Done.')

    # generate Adam optimizer close to mBERT's
    optimizer = torch.optim.Adam(student.parameters(), lr=lr, betas=(0.9, 0.999), eps=eps, weight_decay=weight_decay)
    optimizer.zero_grad(set_to_none=True) # just to be sure
    lr_scheduler = get_linear_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=steps_per_ta) 

    with torch.set_grad_enabled(True):

        for epoch in range(epochs):

            loss = 0 # initialize loss
            start = datetime.now()

            print(f'Begin epoch {epoch+1}/{epochs}. Current time: {datetime.now()}.')

            for batch_idx, inputs in enumerate(data_loader): # each iter of this loop is one batch
        
                for j in inputs: # unwrap extra []
                    inputs[j] = inputs[j][0]

                inputs.to(device)
                
                # get predictions
                teacher_outputs = teacher(**inputs)
                student_outputs = student(**inputs)

                # get hiddens to compare
                teacher_hiddens = teacher_outputs.hidden_states
                student_hiddens = student_outputs.hidden_states
                
                # get attentions to compare
                teacher_attentions = teacher_outputs.attentions
                student_attentions = student_outputs.attentions
                
                # calculate the loss between them
                for i in range(ta_num): # for each layer in the student
                    loss += MSELoss(teacher_hiddens[i+1], student_hiddens[i]) # learn from hidden layer above in teacher
                    for j in range(len(teacher_hiddens[i+1])): # for each attention matrix
                        loss += MSELoss(teacher_attentions[i+1][j], student_attentions[i][j]) / 12 # learn same way
                loss += MSELoss(teacher_hiddens[ta_num+1], student_hiddens[ta_num]) # last hidden

                loss /= ta_num
                
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
                    print(f'Batches completed: {batch_idx:6}/{steps_per_ta}. Average time per batch: {avg_time}. ETA: {eta}. Loss: {loss:.4}.')
 
                loss = 0

        # save checkpoint if doing multiple epochs
        if epochs > 1 and epoch != epochs - 1:
            print('Saving checkpoint.')
            ckpt_path = os.path.join(ckpts, f'ta_{ta_num}_ckpt_{epoch}.bin')
            torch.save(student, ckpt_path)
            print('Saved checkpoint to {ckpt_path}.')

    # save trained model for this TA    
    print(f'Saving trained student model. Training took: {datetime.now() - start}.')
    model_save_path = os.path.join(trained, f'ta_{ta_num}.bin')
    torch.save(student, model_save_path)
    print(f'Saved model to {model_save_path}.')

    # summary statistics
    batch_indices = np.array(batch_indices)
    losses = (torch.tensor(losses)).detach().numpy()
    plt.figure()
    plt.title(f'Pretraining loss. Max loss: batch {np.argmax(losses)}. Min loss: batch: {np.argmin(losses)}.')
    plt.xlabel('Batch index')
    plt.ylabel('Loss')
    plt.scatter(batch_indices, losses, s=2, label=str(ta_num))
    plt.legend(loc='upper right')
    graph_loc = os.path.join(trained, f'pretraining_loss_ta_{ta_num}.png')
    plt.savefig(graph_loc, format='png')
    print(f'Wrote loss curve to {graph_loc}.')
    teacher = student # for next loop
