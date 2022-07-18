import os
import numpy as np
import torch
from transformers import BertTokenizer
from transformers import logging
from datasets import load_dataset
from evaluate import load

# prevent console spam
logging.set_verbosity_error()

max_length = 128
device = 'cuda:2'

langs = ['ar', 'bg', 'de', 'el', 'en', 'es', 'fr', 'hi', 'ru', 'sw', 'th', 'tr', 'ur', 'vi', 'zh']

# paths
proj_dir = os.path.join('/home', 'ddegenaro', 'CAMeMBERT')
trained = os.path.join(proj_dir, 'trained')
linears = os.path.join(proj_dir, 'linears')

print('Getting tokenizer.')
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case=False)
print('Done.')

print('Getting test data.')
test_data = load_dataset('xnli', split='test', language='all_languages').shuffle(seed=69)
xnli_metric = load("xnli")
print('Done.')

class FFN(torch.nn.Module):
    def __init__(self):
        super(FFN, self).__init__()
        self.linear = torch.nn.Sequential(torch.nn.LazyLinear(3))
    def forward(self, x):
        x = torch.reshape(x, (1, max_length*119547))
        return self.linear(x)

def probs_test(sentence1, sentence2):
    inputs = tokenizer(sentence1, sentence2, return_tensors='pt', padding='max_length', max_length=max_length, truncation=True)
    inputs.to(device)

    with torch.no_grad():
        bert_outputs = model(**inputs).prediction_logits
        final_outputs = ffn(bert_outputs)

    return final_outputs

def final_class(sentence1, sentence2):
    return np.argmax(np.array(probs_test(sentence1, sentence2).to('cpu')))

for ta_num in reversed(range(6, 12)):
    
    print('Getting model.')
    model = torch.load(os.path.join(trained, 'ta' + str(ta_num) + '.bin'))
    model.eval()
    model.to(device)

    ffn = torch.load(os.path.join(linears, 'ta' + str(ta_num) + '.bin'))
    ffn.eval()
    ffn.to(device)
    print('Done.')

    preds = dict.fromkeys(langs)    
    for lang in langs:
        preds[lang] = []

    refs = []

    i = 0
    
    print('Begin evaluation.')

    for item in test_data:

        refs.append(item['label'])

        for j in range(len(langs)):
  
            sentence1 = item['premise'][langs[j]]
            sentence2 = item['hypothesis']['translation'][j]

            predicted_class_id = final_class(sentence1, sentence2)

            preds[langs[j]].append(predicted_class_id)

        i += 1

        if i % 1000 == 0:
            print(f'Completed {i} examples.')

    results = dict.fromkeys(langs)
    for lang in langs:
        results[lang] = xnli_metric.compute(predictions=preds[lang], references=refs)

    results_file = os.path.join(proj_dir, 'results_' + str(ta_num))
    print(f'TA {ta_num} evaluated. Results written to {results_file}.')
    with open(results_file, 'w+') as f:
        f.write(f'TA: {ta_num}.')
        f.write(str(results))
