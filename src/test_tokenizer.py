class Tokenizer:
    def __init__(self):
        self.khmer_start = 6016
        self.khmer_end = 6143
        # 127 (from 1 to 127 for khmer)
        self.khmer_max = self.khmer_end - self.khmer_start


    def encode(self, text):
        token = []
        for ch in text:
            code = ord(ch)
            if self.khmer_start <= code <= self.khmer_end:
                # from 0 + 1 so it is 1
                token.append(code - self.khmer_start + 1)
                continue
            if ch == " ":
                # +2
                token.append(self.khmer_max + 1 + 1)
                continue

            # ASCII digits +4
            if "0" <= ch <= "9":
                token.append(self.khmer_max + 3 + 1 + (ord(ch) - 48))
                continue

            # UNK = +3 
            token.append(self.khmer_max + 1 + 1 + 1)

        return token

    def blank_id(self):
        return 0

    def get_size(self):
        # blank + khmer + space + UNK + digits(10)
        return 1 + self.khmer_max + 1 + 1 + 10

    def decode(self, tokens):
        text = []
        for t in tokens:
            if t == 0:
                continue

            if 1 <= t <= self.khmer_max + 1 :
                text.append(chr(self.khmer_start + t - 1))
                continue

            if t == self.khmer_max + 2:
                text.append(" ")
                continue

            if self.khmer_max + 4 <= t <= self.khmer_max + 13:
                digit = t - (self.khmer_max + 4)
                text.append(chr(digit + 48))
                continue

            text.append("UNK")

        return "".join(text)


tokenizer = Tokenizer()
print(tokenizer.encode("សាលារ រៀន 01234 ១២៣៤ afdsa"))
print(tokenizer.decode(tokenizer.encode("សាលារ រៀន 01234 ១២៣៤ afdsa")))
