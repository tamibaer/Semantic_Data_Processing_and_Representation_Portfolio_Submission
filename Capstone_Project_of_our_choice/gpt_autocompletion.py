from transformers import GPT2LMHeadModel, GPT2Tokenizer

import torch


class Autocompletion:
    def __init__(self, data_path=None, model_path="gpt2-fine-tuned"):
        self.model_path = model_path
        self.finetuned_model = None
        self.finetuned_tokenizer = None

        self.load_fine_tuned_model()

    def load_fine_tuned_model(self):
        model_name = self.model_path
        self.finetuned_model = GPT2LMHeadModel.from_pretrained(model_name)
        self.finetuned_tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    def load_train_data(self, path):
        f = open(path, "r", encoding="utf-8")
        data = f.read()
        return self.preprocess_data(data)

    def train_model(self):
        return

    def make_suggestions(self, input_text, num_samples=5):
        input_ids = self.finetuned_tokenizer.encode(input_text, return_tensors="pt")

        attention_mask = torch.ones(
            input_ids.shape, dtype=input_ids.dtype, device=input_ids.device
        )

        outputs = self.finetuned_model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_length=10,
            do_sample=True,
            num_return_sequences=num_samples,
            pad_token_id=self.finetuned_tokenizer.eos_token_id,
        )

        next_words = [
            self.finetuned_tokenizer.decode(
                output[input_ids.shape[-1] :], skip_special_tokens=True
            ).split()[0]
            for output in outputs
        ]

        next_words = [
            word
            for word in set(filter(lambda x: x not in ["\n"], next_words))
            if not word.isdigit()
        ]

        next_words.sort(key=lambda word: (next_words.count(word), word), reverse=True)
        print(next_words)
        return next_words
