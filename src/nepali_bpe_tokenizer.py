from datasets import load_dataset

# dataset = load_dataset("Sakonii/nepalitext-language-model-dataset")

# dataset
dataset = load_dataset("text", data_files={"train":"lspd.txt", "test": "Nepali.txt"})


dataset = dataset["train"]



def get_training_corpus():
    for i in range(0, len(dataset), 1000):
        yield dataset[i : i + 1000]["text"]

from tokenizers import (
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
    Tokenizer,
)

tokenizer = Tokenizer(models.BPE())

tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

print(tokenizer.pre_tokenizer.pre_tokenize_str("Let's test pre-tokenization!"))

trainer = trainers.BpeTrainer(vocab_size=50000, special_tokens=["<|endoftext|>"])
tokenizer.train_from_iterator(get_training_corpus(), trainer=trainer)

tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)



sentence = "रामले खाना खायो"
encoding = tokenizer.encode(sentence)
start, end = encoding.offsets[4]
sentence[start:end]

print(encoding.tokens)

print(encoding.offsets)

tokenizer.decoder = decoders.ByteLevel()

print(tokenizer.decode(encoding.ids))

print(tokenizer.encode(sentence))

tokenizer.model.save("./models/tokenizer/")
tokenizer.save("NepaliBPETokenizer.json")
