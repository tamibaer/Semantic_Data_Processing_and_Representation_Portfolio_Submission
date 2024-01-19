import re
import numpy as np
import difflib as d
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import regex as re
import os

# https://www.geeksforgeeks.org/next-word-prediction-with-deep-learning-in-nlp/

class Autocompletion:
    def __init__(self, data_path = None, model_path='saved_model/my_model'):
        self.model = None
        self.X = None
        self.y = None
        self.max_sequence_len = None
        self.total_words = None 
        self.tokenizer = None
        self.sentence_amount = 100
        self.model_path = model_path

        if not data_path: return

        if os.path.exists(model_path):
            self.model = tf.keras.models.load_model(model_path)
            self.max_sequence_len, _, self.tokenizer, _ = self.do_preprocessing(data_path)
            print("Model loaded ...")
        else:
            # 18min training
            self.X, self.y, self.max_sequence_len, self.total_words, self.tokenizer = self.load_dataset(data_path, self.sentence_amount)

    def load_train_data(self, path):
        f = open(path, "r",  encoding='utf-8')
        data = f.read()
        return self.preprocess_data(data)
    
    def train_model(self):
        if self.X is None or self.y is None or self.max_sequence_len is None: return
        self.model = Sequential([
            Embedding(self.total_words, 10, input_length=self.max_sequence_len-1),
            LSTM(128),
            Dense(self.total_words, activation='softmax'),
        ])
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.model.fit(self.X, self.y, epochs=500, verbose=1)
        self.model.save(self.model_path)
    
    def make_suggestions(self, input):
        corrected_input = self.preprocess_data(input, backspace_replace="")

        context = ' '.join(corrected_input.split(" ")[:-1])
        current_word = corrected_input.split(" ")[-1]

        token_list = self.tokenizer.texts_to_sequences([context])[0]
        token_list = pad_sequences([token_list], maxlen=self.max_sequence_len-1, padding='pre')
        predicted_probs = self.model.predict(token_list, verbose=0)
        top_predictions = np.flip(np.argsort(predicted_probs)[0][-50:])
        predicted_word = [self.tokenizer.index_word[w] for w in top_predictions]

        if len(current_word) > 0:
            wordmatch = []
            sum = 0
            for canidate in predicted_word:
                alignment = self.similar(current_word, canidate)
                wordmatch.append(alignment)
                sum += alignment

            wordmatch = np.array(wordmatch) / sum if sum > 0 else np.array(wordmatch)
            return [predicted_word[w] for w in np.flip(np.argsort(wordmatch)[-5:])]
        else:
            return predicted_word[0:5]
            
        
    def preprocess_data(self, data, backspace_replace=" "):
        t = data.replace("\n", backspace_replace)
        replaced = re.sub('[^[a-zA-Z ßäüöÜÖÄ]]*', '', t)
        replaced = replaced.lower()
        replaced = re.sub(' +', ' ', replaced)
        return replaced

    def similar(self, a, b):
        ma = 0
        for i in range(min(len(a), len(b))):
            if a[i] == b[i]: ma += 1
        return ma / len(a)
    
    def file_to_sentence_list(self, file_path):
        with open(file_path, 'r', encoding='utf8') as file: text = file.read()
        return [sentence.strip() for sentence in re.split(r'(?<=[.!?])\s+', text) if sentence.strip()]
 
    def do_embedding(self, text_data, tokenizer):
        input_sequences = []
        for token_list in tokenizer.texts_to_sequences(text_data):
            for i in range(1, len(token_list)): input_sequences.append(token_list[:i+1])
        max_sequence_len = max([len(seq) for seq in input_sequences])
        input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))
        return input_sequences, max_sequence_len
    
    def create_tokenizer(self, text_data):
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(text_data)
        total_words = len(tokenizer.word_index) + 1
        return tokenizer, total_words

    def load_dataset(self, file_path, sentence_amount=100):
        max_sequence_len, total_words, tokenizer, input_sequences = self.do_preprocessing(file_path, sentence_amount)
        X, y = input_sequences[:, :-1], input_sequences[:, -1]
        y = tf.keras.utils.to_categorical(y, num_classes=total_words)
        return X, y, max_sequence_len, total_words, tokenizer
    
    def do_preprocessing(self, file_path, sentence_amount=100):
        text_data = self.file_to_sentence_list(file_path)[0:sentence_amount]
        tokenizer, total_words = self.create_tokenizer(text_data)
        input_sequences, max_sequence_len = self.do_embedding(text_data, tokenizer)
        return max_sequence_len, total_words, tokenizer, input_sequences
    