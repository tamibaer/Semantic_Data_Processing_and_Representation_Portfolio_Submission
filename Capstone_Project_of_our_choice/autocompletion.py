import re
import numpy as np
import difflib as d

class Autocompletion:
    def __init__(self, data_path = None):
        self.data = None
        self.model = None
        if data_path:
            self.data = self.load_train_data(data_path)

    def load_train_data(self, path):
        f = open(path, "r",  encoding='utf-8')
        data = f.read()
        return self.preprocess_data(data)
    
    def train_model(self):
        model = {}
        previous_token = None
        for w in self.data.split(" "):
            if previous_token is not None:
                if w in model[previous_token]:
                    model[previous_token][w] += 1
                else:
                    model[previous_token][w] = 1
            if w in model:
                pass
            else:
                model[w] = {}

            previous_token = w
        self.model = model
    
    def make_suggestions(self, input):
        corrected_input = self.preprocess_data(input, backspace_replace="")

        context = ' '.join(corrected_input.split(" ")[:-1])

        markov_input = context.split(" ")[-1]
        sorted_candidates = []
        if markov_input not in self.model: 
            return sorted_candidates
        else:
            canidates = self.model[markov_input]
            sorted_candidates = sorted(canidates.items(), key=lambda x:x[1], reverse=True)

        combined = {}
        if len(corrected_input.split(" ")[-1]) > 0:
            wordmatch = []
            sum = 0
            for canidate in sorted_candidates:
                alignment = self.similar(corrected_input.split(" ")[-1], canidate[0])
                wordmatch.append(alignment)
                sum += alignment

            if sum > 0:
                for i in range(len(wordmatch)): wordmatch[i] /= sum
            for a, b in zip(sorted_candidates, wordmatch): combined[a[0]] = a[1] * b
        else:
            for a in sorted_candidates: combined[a[0]] = a[1]

        result = sorted(combined.items(), key=lambda x:x[1], reverse=True)
        return [word[0] for word in result[:5]]
        
    def preprocess_data(self, data, backspace_replace=" "):
        t = data.replace("\n", backspace_replace)
        replaced = re.sub('[^[a-zA-Z ßäüöÜÖÄ]]*', '', t)
        replaced = replaced.lower()
        replaced = re.sub(' +', ' ', replaced)
        return replaced

    def similar(self, a, b):
        seqmatch = d.SequenceMatcher(None, a, b)
        return seqmatch.ratio()
    
    