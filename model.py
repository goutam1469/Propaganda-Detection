#Importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch import nn
from transformers import BertTokenizer, BertModel, BertForSequenceClassification, BertTokenizerFast, BertForTokenClassification, Trainer, TrainingArguments
from torch.optim import Adam
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from sklearn.utils.class_weight import compute_class_weight
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from nltk.probability import FreqDist
from transformers import pipeline
from sklearn.utils.class_weight import compute_class_weight
import logging
import pickle

logging.basicConfig(level=logging.INFO)

#Loading the training and test data
train_path = r'\propaganda_detection_v2\propaganda_train.tsv'
# test_path = '/content/propaganda_val.tsv'
train_df = pd.read_csv(train_path, delimiter = '\t', quotechar='|')
# test_df = pd.read_csv(test_path, delimiter = '\t', quotechar='|')


#Approach : Build a naive bayes binary classifier and then do the span detector (using CountVectorizer)

#Initializing the tokenizer
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

#converting labels in to binary
train_df['labels'] = train_df['label'].apply(lambda x:x if x == 'not_propaganda' else 'propaganda')
#test_df['labels'] = test_df['label'].apply(lambda x:x if x== 'not_propaganda' else 'propaganda')

#vectorizing the text
vectorizer = CountVectorizer()
train_transform = vectorizer.fit_transform(train_df.tagged_in_context)
#test_transform = vectorizer.transform(test_df.tagged_in_context)

#naive bayes model
classifier = MultinomialNB()
classifier.fit(train_transform, train_df.labels)

#Saving the first binary classifier
pickle.dump(classifier, open('classifier.pkl','wb'))


#Sub category classification

# Prepare BIO data
def get_bio_labels(text, label, tokenizer):
    if '<BOS>' in text and '<EOS>' in text:
        pre, rest = text.split('<BOS>')
        span, post = rest.split('<EOS>')
        clean_text = pre.strip() + ' ' + span.strip() + ' ' + post.strip()
        tokens = tokenizer.tokenize(clean_text)
        span_tokens = tokenizer.tokenize(span.strip())
        span_start = len(tokenizer.tokenize(pre.strip()))
        span_len = len(span_tokens)
        bio_labels = ['O'] * len(tokens)
        if span_len > 0:
            bio_labels[span_start] = 'B-' + label
            for i in range(1, span_len):
                bio_labels[span_start + i] = 'I-' + label
        return tokens, bio_labels
    else:
        clean_text = text.replace('<BOS>', '').replace('<EOS>', '').strip()
        tokens = tokenizer.tokenize(clean_text)
        bio_labels = ['O'] * len(tokens)
        return tokens, bio_labels



# Filter out 'not_propaganda' samples during BIO label generation
train_bio = []
for _, row in train_df_v2.iterrows():
    if row['label'] == 'not_propaganda':
        continue
    bio_data = get_bio_labels(row['tagged_in_context'], row['label'], tokenizer)
    train_bio.append(bio_data)


# test_bio = []
# for _, row in test_df.iterrows():
#     if row['label'] == 'not_propaganda':
#         continue
#     bio_data = get_bio_labels(row['tagged_in_context'], row['label'], tokenizer)
#     test_bio.append(bio_data)


# Build label2id and id2label
all_labels = set(l for _, labels in train_bio for l in labels)
label_list = sorted(all_labels)
label2id = {l: i for i, l in enumerate(label_list)}
id2label = {i: l for l, i in label2id.items()}

# Encode examples with proper subword alignment
def encode_examples(bio_data, tokenizer, label2id, max_length=128):
    input_ids, attention_masks, label_ids = [], [], []
    for tokens, labels in bio_data:
        enc = tokenizer(tokens, truncation=True, padding='max_length', max_length=max_length, is_split_into_words=True, return_tensors='pt')
        word_ids = enc.word_ids(batch_index=0)
        label_id = []
        previous_word_idx = None
        for word_idx in word_ids:
            if word_idx is None:
                label_id.append(-100)
            elif word_idx != previous_word_idx:
                label_id.append(label2id[labels[word_idx]])
            else:
                # For subword tokens, use I- label if B-, else same
                if labels[word_idx].startswith('B-'):
                    label_id.append(label2id[labels[word_idx].replace('B-', 'I-')])
                else:
                    label_id.append(label2id[labels[word_idx]])
            previous_word_idx = word_idx
        label_id = label_id[:max_length] + [label2id['O']] * (max_length - len(label_id))
        input_ids.append(enc['input_ids'][0])
        attention_masks.append(enc['attention_mask'][0])
        label_ids.append(label_id)
    return {
        'input_ids': torch.stack(input_ids),
        'attention_mask': torch.stack(attention_masks),
        'labels': torch.tensor(label_ids)
    }

train_enc = encode_examples(train_bio, tokenizer, label2id)
# test_enc = encode_examples(test_bio, tokenizer, label2id)

class PropagandaDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings
    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.encodings.items()}
    def __len__(self):
        return self.encodings['input_ids'].shape[0]

train_dataset = PropagandaDataset(train_enc)
# test_dataset = PropagandaDataset(test_enc)


#Initializing the BERT model and training
model = BertForTokenClassification.from_pretrained('bert-base-uncased', num_labels=len(label2id), id2label=id2label, label2id=label2id)

#defining the hyper parameters for training
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    eval_strategy="epoch",
    save_strategy="no",
    logging_dir='./logs',
    logging_steps=10,
    report_to="none"
)

#Initializing the trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
)

trainer.train()
results = trainer.evaluate()
# print("BERT Token Classification Evaluation:", results)





# Predict on test set
predictions, labels, _ = trainer.predict(test_dataset)
pred_labels = np.argmax(predictions, axis=2)

# Convert IDs back to labels
pred_label_names = [[id2label[id] if id != -100 else 'O' for id in sent] for sent in pred_labels]


#Converting BIO encoding back to span
def get_main_span_and_technique(tokens, labels):
    # Find all spans (ignore 'O' and 'not_propaganda')
    spans = []
    current_span = []
    current_label = None
    for token, label in zip(tokens, labels):
        if label.startswith('B-'):
            if current_span:
                spans.append((current_span, current_label))
            current_span = [token]
            current_label = label[2:]
        elif label.startswith('I-') and current_span and label[2:] == current_label:
            current_span.append(token)
        else:
            if current_span:
                spans.append((current_span, current_label))
                current_span = []
                current_label = None
    if current_span:
        spans.append((current_span, current_label))

    # Remove not_propaganda spans
    spans = [s for s in spans if s[1] != 'not_propaganda']

    if not spans:
        # No propaganda span found
        return None, 'not_propaganda'
    else:
        # Pick the longest span (if tie, pick the first)
        main_span, main_label = max(spans, key=lambda x: len(x[0]))
        return ' '.join(main_span), main_label


#Adding the technique prediction for further classification
# test_df['predicted_results'] = test_results

#Creating a pipeline to run through the entire architecture in a convenient manner

# Create the NER pipeline
# ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

# y_true = []
# y_pred = []

# test_df_v2 = test_df[test_df.predicted_results == 'propaganda']

# for idx, row in test_df_v2.iterrows():
#     sentence = row['tagged_in_context'] #.replace('<BOS>', '').replace('<EOS>', '')
#     true_label = row['label']
#     y_true.append(true_label)

#     ner_results = ner_pipeline(sentence)
#     # Get all predicted techniques (entity_group)
#     techniques = [entity['entity_group'] for entity in ner_results if entity['entity_group'] != 'not_propaganda']
#     if techniques:
#         # If multiple, pick the most common (or just the first)
#         pred_label = max(set(techniques), key=techniques.count)
#     else:
#         pred_label = 'not_propaganda'
#     y_pred.append(pred_label)

#     pred_span_tokens = [ent['word'] for ent in ner_results if ent['entity_group'] != 'not_propaganda']
#     pred_span = ' '.join(pred_span_tokens).replace(" ##", "")  # Remove BERT's subword markers

#     print(f"Sentence: {sentence}")
#     print(f"Original span : {true_label}")
#     print(f"Predicted span : {pred_span}")
#     print(f"Predicted: {pred_label} | True: {true_label}")
#     print()



