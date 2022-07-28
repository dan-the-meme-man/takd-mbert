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
device = 'cuda:1'

langs = ['ar', 'bg', 'de', 'el', 'en', 'es', 'fr', 'hi', 'ru', 'sw', 'th', 'tr', 'ur', 'vi', 'zh']

# paths
proj_dir = os.path.join('/home', 'ddegenaro', 'CAMeMBERT')
trained = os.path.join(proj_dir, 'trained')
finetuned = os.path.join(proj_dir, 'finetuned')

print('Getting tokenizer.')
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case=False)
print('Done.')

print('Getting test data.')
test_data = load_dataset('xnli', split='test', language='all_languages').shuffle(seed=69)
xnli_metric = load("xnli")
print('Done.')

for ta_num in reversed(range(6, 12)):
    
    print('Getting model.')
    model = torch.load(os.path.join(finetuned, f'ta_{ta_num}.bin'))
    model.eval()
    model.to(device)

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
            
            inputs = tokenizer(sentence1, sentence2, return_tensors='pt', padding='max_length', max_length=max_length, truncation=True)
            inputs.to(device)

            with torch.no_grad():
                outputs = np.argmax(np.array(model(**inputs).logits[0].to('cpu')))

            preds[langs[j]].append(outputs)

        i += 1

        if i % 1000 == 0:
            print(f'Completed {i} examples.')

    results = dict.fromkeys(langs)
    
    ###
    print(i)
    print(refs)
    ###

    for lang in langs:
        results[lang] = xnli_metric.compute(predictions=preds[lang], references=refs)
        
        ###
        print(preds[lang])
        ###

    results_file = os.path.join(proj_dir, 'results_' + str(ta_num))
    print(f'TA {ta_num} evaluated. Results written to {results_file}.')
    with open(results_file, 'w+') as f:
        f.write(f'TA: {ta_num}.')
        f.write(str(results))
