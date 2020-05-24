# %%
import time
from pprint import pprint

# %%
FNAME = 'demo.txt'

def load_txt(filename=FNAME):
    with open(filename, 'rb') as f:
        document = f.readlines()

    return [e.decode('utf-8').replace('\r\n', '') for e in document]

# %%
document = load_txt(FNAME)
document

# %%
words = dict()
contains = dict()

for sent in document:
    for c in sent:
        if c in words:
            words[c] += 1
            contains[c].add(sent)
        else:
            words[c] = 1
            contains[c] = set()
            contains[c].add(sent)

words = sorted(words.items(), key=lambda x: x[1], reverse=True)
pprint(words)
# %%

# %%
