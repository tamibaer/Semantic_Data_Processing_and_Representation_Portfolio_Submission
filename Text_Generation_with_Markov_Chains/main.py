import markovify
import codecs

with open("koalitionsvertrag.txt", encoding="utf8") as f:
    text = f.read()

text_model = markovify.Text(text)

o = codecs.open("neuer_koalitionsvertrag.txt", "a", "utf-8")
o.write("Markovified Koalitionsvertag \n\n")
for i in range(59):
    o.write(text_model.make_sentence() + "\n")
o.close()
