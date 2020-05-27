# %%
import time
import jieba
from pprint import pprint

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


# %%
FNAME = 'demo.txt'

DOCUMENT = load_txt(FNAME)
DOCUMENT

# %%
ALL_WORDS = dict()
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

        words = [e for e in set(words)]
        COMMON_TEXTS.append(words)

WORDS = sorted(ALL_WORDS.items(), key=lambda x: x[1], reverse=True)

pprint(WORDS)

# %%
from gensim.models import Word2Vec  #  noqa

MODEL = Word2Vec(COMMON_TEXTS,
                 sg=1,
                 size=100,
                 window=5,
                 min_count=5,
                 negative=3,
                 sample=0.001,
                 hs=1,
                 workers=4)

# %%
MODEL[WORDS[0][0]]
# %%
MODEL.most_similar([WORDS[0][0]])

# %%
MODEL.similarity(WORDS[0][0], WORDS[1][0])

# %%
