from tokenizer import Tokenizer

tokenizer = Tokenizer()
print("Text: សម្រាប់ពិនិត្យអក្ខរាវិរុទ្ធភាសាខ្មែរ និងអង់គ្លេស")
print("Encode: " +  str(tokenizer.encode("សម្រាប់ពិនិត្យអក្ខរាវិរុទ្ធភាសាខ្មែរ និងអង់គ្លេស ")))
print("Decode: " + tokenizer.decode(tokenizer.encode("សម្រាប់ពិនិត្យអក្ខរាវិរុទ្ធភាសាខ្មែរ និងអង់គ្លេស ")))
