from cs336_basics.bpe import run_train_bpe
from cs336_basics.Tokenizer import Tokenizer

vocab, merges = run_train_bpe("E:\githome\cs336\data\debug_corpus.txt", vocab_size=256+6, special_tokens=["<|endoftext|>"])