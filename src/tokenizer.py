import sentencepiece as spm

# # Train a unigram tokenizer on your corpus
# spm.SentencePieceTrainer.train(
#     input='corpus.txt',          # path to your grapheme-level corpus
#     model_prefix='khmer_sp',     # output prefix for model files
#     vocab_size=15000,            # desired vocab size
#     character_coverage=1.0,      # include all characters
#     model_type='unigram'         # unigram model
# )


class Tokenizer:
    def __init__(self, model_file="khmer_sp.model"):
        self.model_file = model_file
        self.sp = spm.SentencePieceProcessor(self.model_file)

    def encode(self, text):
        return self.sp.encode_as_ids(text)  # Return list of IDs (ints)

    def decode(self, ids):
        return self.sp.decode_ids(ids)  # Decode from list of IDs (ints)

    def __len__(self):
        return self.sp.get_piece_size()
    def blank_id(self):
        return self.sp.PieceToId('[PAD]')