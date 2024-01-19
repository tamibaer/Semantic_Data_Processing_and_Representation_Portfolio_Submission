import re
import numpy as np
import difflib as d
import gensim
import regex as re
import os
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

class PredictionModel(nn.Module):
    def __init__(self, embedding_model):
        super().__init__()
        self.lstm = nn.LSTM(input_size=10, hidden_size=10, num_layers=1, batch_first=True)
        self.multihead_attn = nn.MultiheadAttention(10, 5)
        self.dropout = nn.Dropout(0.2)
        self.linear = nn.Linear(150, 100)
        self.linear2 = nn.Linear(100, 50)
        self.linear3 = nn.Linear(50, 10)
        self.relu = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.softmax = nn.Softmax()
        self.embedd = nn.Parameter(torch.tensor(embedding_model.syn1neg.T), requires_grad=False)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = torch.flatten(x, start_dim=1)
        x = self.linear(self.dropout(x))
        x = self.relu(x)
        x = self.linear2(self.dropout(x))
        x = self.relu2(x)
        x = self.linear3(self.dropout(x))
        x = torch.matmul(x, self.embedd)
        return x

class Autocompletion:
    def __init__(self, data_path = None, model_path='saved_model/prediction_model.pt'):
        self.model_path = model_path

        if not data_path: return

        self.dataloader = None
        text = self.load_train_data(data_path)
        self.embedding_model = self.train_embedding_model(text)

        if not os.path.exists(model_path):
            self.dataloader = self.create_training_dataloader(text)
        else:
            self.prediction_model = PredictionModel(self.embedding_model)
            self.prediction_model.load_state_dict(torch.load(model_path))
            self.prediction_model.eval()
            print("Model loaded ...")

    def load_train_data(self, path):
        f = open(path, "r",  encoding='utf-8')
        data = f.read()
        return data
    
    def train_embedding_model(self, text):
        data = []
        for sentence in text.split(". "):
            tmp = []
            sent = self.preprocess_data(sentence)
            for word in sent.split():
                    tmp.append(word)
            data.append(tmp)

        return gensim.models.Word2Vec(data, min_count=1, vector_size=10, window=5, epochs=100)
    
    def create_training_dataloader(self, text):
        x_values = []
        y_values = []
        replaced = self.preprocess_data(text)
        text = ' '.join(replaced.split()[:2000])
        for i in range(1, len(text.split())):
            x = self.encode(' '.join(text.split()[:i]), fill=True)
            y = self.embedding_model.wv.key_to_index[text.split()[i]]
            x_values.append(x)
            y_values.append(y)

        tensor_x = torch.Tensor(x_values).type(torch.float) 
        tensor_y = torch.Tensor(y_values).type(torch.long)

        my_dataset = TensorDataset(tensor_x,tensor_y) 
        my_dataloader = DataLoader(my_dataset, batch_size=512) 
        return my_dataloader

    def encode(self, input, length=15, fill=False):
        result = []
        for w in input.split()[-length:]:
            result.append(self.embedding_model.wv[w])
        if fill:
            while len(result) < length: result.insert(0, np.zeros(self.embedding_model.vector_size))
        return result
    
    def train_model(self):
        if self.dataloader is None: return
        n_epochs = 500
        self.prediction_model = PredictionModel(self.embedding_model)

        optimizer = optim.Adam(self.prediction_model.parameters())
        loss_fn = nn.CrossEntropyLoss() 

        hist = []
        for epoch in range(n_epochs):
            print(f"Epoch: {epoch}")
            self.prediction_model.train()
            for X_batch, y_batch in self.dataloader:
                y_pred = self.prediction_model(X_batch)
                loss = loss_fn(y_pred, y_batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
         
            self.prediction_model.eval()
            print(f"Loss: {loss}")
            hist.append(loss.item())

        torch.save(self.prediction_model.state_dict(), self.model_path)
    
    def make_suggestions(self, input):
        corrected_input = self.preprocess_data(input, backspace_replace="")

        context = ' '.join(corrected_input.split(" ")[:-1])
        current_word = corrected_input.split(" ")[-1]

        prompt = self.preprocess_data(context)
        prompt = self.encode(prompt, fill=True)
        x = torch.tensor([prompt], dtype=torch.float32)

        prediction = self.prediction_model(x)
        similar_words_indices = prediction.detach().numpy()[0].argsort()[-50:][::-1] 
        predicted_word = [self.embedding_model.wv.index_to_key[idx] for idx in similar_words_indices]

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
    