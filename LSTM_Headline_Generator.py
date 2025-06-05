with open("/content/dataset.txt", encoding="latin-1") as f:
    dataset = f.read().splitlines()

dataset[:10]

import string
import unicodedata

def clean_and_normalize_text(txt):
    # Remove punctuation and convert to lowercase
    txt = "".join(c for c in txt if c not in string.punctuation).lower()
    # Normalize unicode characters and encode to ASCII
    txt = unicodedata.normalize('NFKD', txt).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    return txt

dataset = [clean_and_normalize_text(headline) for headline in dataset]

dataset[:10]

from tensorflow.keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer()

def generate_token_sequences(tokenizer, dataset):
    # Build the tokenizer
    tokenizer.fit_on_texts(dataset)
    total_words = len(tokenizer.word_index) + 1

    # Tokenize the text in the dataset
    dataset_tokens = []
    for text in dataset:
        text_tokens = tokenizer.texts_to_sequences([text])[0]
        # Generate n-grams from the tokenized text
        for i in range(1, len(text_tokens)):
            n_gram = text_tokens[:i+1]
            dataset_tokens.append(n_gram)

    return dataset_tokens, total_words


dataset_tokens, total_words = generate_token_sequences(tokenizer, dataset)

import pickle

with open('tokenizer.pkl', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow.keras.utils as ku

def prepare_sequences_and_labels(dataset_tokens, total_words):
    # Determine the length of the longest sequence in the dataset
    max_sequence_len = max([len(text) for text in dataset_tokens])

    # Apply padding to all sequences to ensure they are of the same length
    dataset_tokens = np.array(pad_sequences(dataset_tokens, maxlen=max_sequence_len, padding='pre'))

    # Generate input features and target labels
    X_train, y_train = dataset_tokens[:, :-1], dataset_tokens[:, -1]

    # One-hot encode the labels
    y_train = ku.to_categorical(y_train, num_classes=total_words)

    return X_train, y_train, max_sequence_len


X_train, y_train, max_sequence_len = prepare_sequences_and_labels(dataset_tokens, total_words)

max_sequence_len

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dropout, Dense, BatchNormalization

def create_model(max_sequence_len, total_words):

    model = Sequential()

    # Add Input Embedding Layer
    model.add(Embedding(input_dim=total_words, output_dim=50))

    # Add Hidden Layer 1 - LSTM Layer
    model.add(LSTM(128, return_sequences=True))
    model.add(Dropout(0.2))

    # Add Hidden Layer 2 - LSTM Layer
    model.add(LSTM(64))
    model.add(Dropout(0.2))

    # Add Batch Normalization
    model.add(BatchNormalization())

    # Add Output Layer
    model.add(Dense(total_words, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

model = create_model(max_sequence_len, total_words)
model.summary()

from tensorflow.keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)

model.fit(X_train, y_train, epochs=100, batch_size=32, callbacks=[early_stopping])

def generate_text_from_prompt(prompt, num_words, model, tokenizer, max_sequence_len):
    generated_text = prompt

    for _ in range(num_words):
        # Preprocess the prompt
        prompt_proc = clean_and_normalize_text(generated_text)
        prompt_proc = tokenizer.texts_to_sequences([prompt_proc])[0]
        prompt_proc = pad_sequences([prompt_proc], maxlen=max_sequence_len-1, padding='pre')

        # Predict the next word
        predict = model.predict(prompt_proc, verbose=0)
        predicted_index = np.argmax(predict, axis=1)[0]

        # Convert predicted index to word
        next_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted_index:
                next_word = word
                break

        # Append the predicted word to the generated text
        generated_text += " " + next_word

    return generated_text.title()

print(generate_text_from_prompt("Cybersecurity", 5, model, tokenizer, max_sequence_len))

print(generate_text_from_prompt("Artificial Intelligence", 3, model, tokenizer, max_sequence_len))

print(generate_text_from_prompt("Future", 7, model, tokenizer, max_sequence_len))

print(generate_text_from_prompt("Blockchain", 8, model, tokenizer, max_sequence_len))

print(generate_text_from_prompt("Automobiles", 6, model, tokenizer, max_sequence_len))

# Define the file paths for saving the model and weights
model_path = 'trained_model.model.h5'
weights_path = 'trained_model.weights.h5'

# Save the trained model architecture
model.save(model_path)

# Save the trained model weights
model.save_weights(weights_path)