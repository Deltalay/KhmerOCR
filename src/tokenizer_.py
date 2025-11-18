from tokenizer import Tokenizer

tokenizer = Tokenizer()
print(tokenizer.encode("សម្រាប់ពិនិត្យអក្ខរាវិរុទ្ធភាសាខ្មែរ និងអង់គ្លេស "))
print(tokenizer.decode(tokenizer.encode("សម្រាប់ពិនិត្យអក្ខរាវិរុទ្ធភាសាខ្មែរ និងអង់គ្លេស ")))
