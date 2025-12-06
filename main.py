import preprocess
import data_loader
import classifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

#Vector Representation
format = 'tfidf'

#Load Data
train_df, val_df = data_loader.load_data()

#Preprocess Data
train_df['span'] = train_df['tagged_in_context'].apply(preprocess.extract_bos_eos)
val_df['span'] = val_df['tagged_in_context'].apply(preprocess.extract_bos_eos)

#Vectorization
vectorizer = TfidfVectorizer()
vectorized_train_df = vectorizer.fit_transform(train_df.span)
vectorized_val_df = vectorizer.transform(val_df.span)


prop_classifier = classifier.prop_classifier(vectorized_train_df, train_df)

prediction = classifier.predict(vectorized_val_df)
print(prediction)