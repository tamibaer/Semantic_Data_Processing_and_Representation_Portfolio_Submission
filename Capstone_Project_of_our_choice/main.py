import tkinter as tk

# from autocompletion import Autocompletion
# from gpt_autocompletion import Autocompletion
# from embedding_autocompletion import Autocompletion
from lstm_autocompletion import Autocompletion


class MyApp:
    def __init__(self, root):
        self.autocompletion = Autocompletion(
            "datasets/shakespeare_preprocessed.txt", model_path="saved_model/shakespeare_model"
        )
        self.autocompletion.train_model()

        self.root = root
        self.root.title("Autocompletion App")

        self.user_text = tk.Text(root, wrap=tk.WORD, height=5, width=40)
        self.user_text.pack(padx=10, pady=10)
        self.user_text.bind("<KeyRelease>", self.on_key_press)
        self.user_text.bind("<Tab>", self.on_tab_press)
        self.user_text.bind("<Return>", self.on_return_press)

        self.suggested_words_select = tk.Listbox(
            root, selectmode=tk.SINGLE, height=5, exportselection=False
        )
        self.suggested_words_select.pack(pady=5)
        self.suggested_words_select.bind("<Tab>", self.on_tab_press)
        self.suggested_words_select.bind("<Return>", self.on_return_press)
        self.suggested_words_select.bind("<Escape>", self.on_escape_pressed)

    def on_key_press(self, _):
        self.suggested_words_select.selection_set(0)
        self.update_suggested_words()

    def on_tab_press(self, _):
        if root.focus_get() == self.user_text:
            self.suggested_words_select.focus_set()
            self.suggested_words_select.selection_set(0)
        else:
            current_index = self.suggested_words_select.curselection()
            if current_index:
                next_index = (current_index[0] + 1) % self.suggested_words_select.size()
                self.suggested_words_select.selection_clear(current_index)
                self.suggested_words_select.selection_set(next_index)
                self.suggested_words_select.activate(next_index)
        return "break"

    def on_return_press(self, event):
        if root.focus_get() == self.user_text:
            return event

        self.insert_selected_word()
        self.user_text.focus_set()
        return "break"

    def on_escape_pressed(self, _):
        self.user_text.focus_set()

    def update_suggested_words(self):
        current_text = self.user_text.get("1.0", tk.END).lower()

        matching_words = self.autocompletion.make_suggestions(current_text)

        self.suggested_words_select.delete(0, tk.END)
        for word in matching_words:
            self.suggested_words_select.insert(tk.END, word)

    def insert_selected_word(self):
        selected_word = self.suggested_words_select.get(tk.ACTIVE)
        if selected_word:
            if self.user_text.get("1.0", "end-1c").endswith(" "):
                self.user_text.insert(tk.END, selected_word + " ")
                self.update_suggested_words()
            else:
                current_text = self.user_text.get("1.0", "end-1c").split()
                if current_text:
                    current_text[-1] = selected_word
                    new_text = " ".join(current_text)
                    self.user_text.delete("1.0", "end-1c")
                    self.user_text.insert("1.0", new_text + " ")


if __name__ == "__main__":
    root = tk.Tk()
    app = MyApp(root)
    root.mainloop()
