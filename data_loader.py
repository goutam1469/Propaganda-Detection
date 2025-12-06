import pandas as pd

train_path = r'propaganda_dataset_v2\propaganda_train.tsv'
val_path = r'propaganda_dataset_v2\propaganda_val.tsv'

def load_data():
    train_df = pd.read_csv(train_path, delimiter = '\t', quotechar='|')
    val_df = pd.read_csv(val_path, delimiter = '\t', quotechar='|')
    return train_df, val_df

