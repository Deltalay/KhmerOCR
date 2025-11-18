import regex


file_path = "labels.txt"
corpus_file = "corpus.txt"

with open(file_path, "r", encoding="utf-8") as f_in, open(corpus_file, "w", encoding="utf-8") as f_out:
    for line in f_in:
        line = line.strip()
        if not line:
            continue
        try:
            _, kh_text = line.split("\t")
            
            f_out.write(kh_text + "\n")
        except ValueError:
            continue

print(f"Corpus saved to {corpus_file}")
