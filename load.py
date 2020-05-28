# %%
# System
import time
from pprint import pprint

# Computing
import jieba
import numpy as np
from sklearn import cluster, manifold
from sklearn.preprocessing import StandardScaler
from gensim.models import Word2Vec

# Plotting
import matplotlib.pyplot as plt

jieba.enable_paddle()

# %%


def regulate(string):
    """Regulate string symbol

    Arguments:
        string {str} -- String to be regulated

    Returns:
        Regulated string
    """
    table = {',': '，', ':': '：'}
    for symbol in table:
        string = string.replace(symbol, table[symbol])
    return string


def end_sentence(string):
    """Put period at the end of the string

    Arguments:
        string {str} -- String to be ended

    Returns:
        Ended string
    """
    if not string.endswith('。'):
        return f'{string}。'
    return string


def split_sentence(string):
    """Split sentences based on [string]

    Arguments:
        string {str} -- Paragraph to be splitted

    Returns:
        Splitted paragraph
    """
    string = regulate(string)
    if string.endswith('。'):
        string = string[:-1]
    return [end_sentence(e) for e in string.split('。')]


def load_txt(filename):
    """Make paragraphs, a nested list.

    Keyword Arguments:
        filename {str} -- File name to be loaded (default: {FNAME})

    Returns:
        Parsed regulated text 
    """
    with open(filename, 'rb') as f:
        document = f.readlines()

    paragraphes = []
    for para in document:
        para = para.decode('utf-8').replace('\r\n', '')
        splits = split_sentence(para)
        if not para:
            continue
        paragraphes.append(splits)

    return paragraphes


# %% Read .txt file
FNAME = 'demo.txt'

DOCUMENT = load_txt(FNAME)
DOCUMENT

# %% Get words
# Word table of the document
ALL_WORDS = dict()
# Words list as they occurred in the document,
# one sub list means a paragraph.
COMMON_TEXTS = []

divider = 'splitsplit'

table = {
    '、': divider,
    '（': divider,
    '）': divider,
    '，': divider,
    '。': divider,
    '；': divider,
    ';': divider,
    '“': divider,
    '”': divider,
    '"': divider
}

for para in DOCUMENT:
    for sent in para:
        for rep in table:
            sent = sent.replace(rep, table[rep])
        words = []
        for split in sent.split(divider):
            seg_list = jieba.cut(split, use_paddle=True)
            seg_list = [e.replace(' ', '') for e in seg_list]
            seg_list = [e for e in seg_list if len(e) > 1]
            for seg in seg_list:
                if seg in ALL_WORDS:
                    ALL_WORDS[seg] += 1
                else:
                    ALL_WORDS[seg] = 1
            words += seg_list

        # Make sure no repeated word in words
        words = [e for e in set(words)]
        COMMON_TEXTS.append(words)

# WORDS is sorted ALL_WORDS
WORDS = sorted(ALL_WORDS.items(), key=lambda x: x[1], reverse=True)

pprint(WORDS)

# %% Prepare Word2Vec Model

MODEL = Word2Vec(COMMON_TEXTS,
                 sg=1,
                 hs=1,
                 size=100,
                 window=5,
                 negative=3,
                 min_count=5,
                 sample=0.001,
                 workers=4)

# %%
MODEL[WORDS[0][0]]

# %%
MODEL.most_similar([WORDS[0][0]])

# %%
MODEL.similarity(WORDS[0][0], WORDS[1][0])

# %% Get Word Vectors,
# into paired words and vectors
words = []
vectors = []
for word in ALL_WORDS:
    try:
        v = MODEL[word]
    except KeyError:
        continue
    words.append(word)
    vectors.append(v)

vectors = np.array(vectors)
vectors.shape

# %% Clustering
spectral = cluster.SpectralClustering(n_clusters=7)
labels = spectral.fit_predict(vectors)

# %% Plot on 2-D space
# TSNE decomposition
tsne = manifold.TSNE(n_components=2)
v = StandardScaler().fit_transform(vectors)
vectors2 = tsne.fit_transform(v)

# Plot
fig, axes = plt.subplots(2, 1, figsize=(8, 16))

# Plot as node
colors = dict()
ax = axes[0]
for j in np.unique(labels):
    s = ax.scatter(vectors2[labels == j, 0], vectors2[labels == j, 1])
    colors[j] = s.get_facecolor().ravel().tolist()

# Plot as text
ax = axes[1]
for j, w in enumerate(words):
    t = ax.text(vectors2[j, 0], vectors2[j, 1], w,
                dict(fontproperties='SimHei', fontsize=20),
                color=colors[labels[j]])

ax.set_xlim((min(vectors2[:, 0]), max(vectors2[:, 0])))
ax.set_ylim((min(vectors2[:, 1]), max(vectors2[:, 1])))

# %%
