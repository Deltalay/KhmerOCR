import regex as re

BASE = r"[\u1780-\u17A2]"

INDEP_V = r"[\u17A3-\u17B5]"

SUB = r"\u17D2[\u1780-\u17A2]"


DEP_V = r"[\u17B6-\u17C5]"

DIAC = r"[\u17C6-\u17D1]"

GRAPHEME_PATTERN = (
    f"(?:{BASE}|{INDEP_V})(?:{SUB})*(?:{DEP_V})?(?:{DIAC})*"
)

GRAPHEME_RE = re.compile(GRAPHEME_PATTERN)


def split_graphemes(text):
    """
    Returns a list of Khmer grapheme clusters as visually meaningful syllables.
    """
    return GRAPHEME_RE.findall(text)


class GraphemeTokenizer:


    def __init__(self, vocab=None):
        # blank = 0 for CTC
        self.blank = 0

        # If no vocab provided, make an empty one
        if vocab is None:
            vocab = {}

        self.vocab = vocab            
        self.id_to_token = {}         

        # load reverse map
        for k, v in self.vocab.items():
            self.id_to_token[v] = k


    def add_token(self, token):
        """
        Add a grapheme token to vocabulary.
        """
        if token not in self.vocab:
            new_id = len(self.vocab) + 1   # IDs start at 1 (0 is blank)
            self.vocab[token] = new_id
            self.id_to_token[new_id] = token

        return self.vocab[token]

    def build_vocab_from_dataset(self, texts):

        for t in texts:
            for g in split_graphemes(t):
                self.add_token(g)



    def encode(self, text):
        """
        Convert Khmer text → list of cluster IDs.
        Unknown clusters get added automatically.
        """
        clusters = split_graphemes(text)
        ids = []
        for g in clusters:
            if g not in self.vocab:
                self.add_token(g)
            ids.append(self.vocab[g])
        return ids

    def decode(self, ids):

        out = []
        for i in ids:
            if i == self.blank:
                continue
            out.append(self.id_to_token.get(i, ""))  # unknowns should not happen
        return "".join(out)

    # ---- misc ----

    def blank_id(self):
        return self.blank

    def get_size(self):
        return len(self.vocab) + 1  # include blank id


# -------------------------
# Simple usage demo
# -------------------------

if __name__ == "__main__":
    tok = GraphemeTokenizer()
    tok.build_vocab_from_dataset()