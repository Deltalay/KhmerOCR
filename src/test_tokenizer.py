class Tokenizer:
    def encode(self, text):
        token = []
        # Range from 6016 - 6143
        khmer_max = 6143 - 6015
        for i in text:
            dec = ord(i)
            #  0 - 9
            if dec >= 48 and dec <= 57:
                # +1 for handle 0
                dec = khmer_max + 3 - 48 + dec
                token.append(dec)
                continue
            if dec >= 6016 and dec <= 6143:
                dec = dec - 6015
                token.append(dec)
                continue
            if i == " ":
                dec = khmer_max + 1
                token.append(dec)
                continue
            # khmer_max + 2 = UNK
            token.append(khmer_max + 2)
        return token

    def decode(self, token):
        text = []
        khmer_max = 6143 - 6015
        for i in token:
            if i == khmer_max + 1:
                text.append(" ")
                continue
            # Normal Khmer Text
            if i >= 1 and i <= khmer_max:
                dec = i + 6015
                text.append(chr(dec))
                continue
            # This should be handle number
            if i >= khmer_max + 3 - 48:
                dec = i - khmer_max - 3 + 48
                text.append(chr(dec))
                continue
            text.append("UNK")
        return "".join(text)


tokenizer = Tokenizer()
print(tokenizer.encode("សាលារ រៀន 01234 ១២៣៤"))
print(tokenizer.decode(tokenizer.encode("សាលារ រៀន 01234 ១២៣៤")))
