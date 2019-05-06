import io
filename = 'test.txt'
with io.open(filename, 'r', encoding='utf8') as f:
    text = f.read()
text = text.replace('\n\n', '\n')

with io.open(filename, 'w', encoding='utf8') as f:
    f.write(text)
