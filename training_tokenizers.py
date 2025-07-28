from datasets import load_dataset
from tokenizers import (
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
    Tokenizer,
    Regex
)

from transformers import PreTrainedTokenizerFast

ds = load_dataset("uonlp/CulturaX", "en")

def get_training_corpus():
    for i in range(0, len(ds), 1000):
        yield ds[i : i + 1000]["text"]

### WORDPIECE TOKENIZER ###

def word_piece():
    # loading existing BERT tokenizer
    tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]"))
    tokenizer.normalizer = normalizers.BertNormalizer(lowercase=True)

    # this normalizer removes white spaces and regularizes unknown characters
    pre_tokenizer = pre_tokenizers.Sequence(
        [pre_tokenizers.WhitespaceSplit(), pre_tokenizers.Punctuation()]
    )
    
    # now, we add our special tokens and define our trainer
    special_tokens = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]
    trainer = trainers.WordPieceTrainer(vocab_size=25000, special_tokens=special_tokens)
    tokenizer.train_from_iterator(get_training_corpus(), trainer=trainer)
    encoding = tokenizer.encode("Let's test this tokenizer.")
    cls_token_id = tokenizer.token_to_id("[CLS]")
    sep_token_id = tokenizer.token_to_id("[SEP]")

    tokenizer.post_processor = processors.TemplateProcessing(
        single=f"[CLS]:0 $A:0 [SEP]:0",
        pair=f"[CLS]:0 $A:0 [SEP]:0 $B:1 [SEP]:1",
        special_tokens=[("[CLS]", cls_token_id), ("[SEP]", sep_token_id)],
    )
    print(cls_token_id, sep_token_id)
    tokenizer.decoder = decoders.WordPiece(prefix="##")
    tokenizer.decode(encoding.ids)
    tokenizer.save("wordpiece_english.json")


### BPE TOKENIZER ###

def bpe():
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    trainer = trainers.BpeTrainer(vocab_size=25000, special_tokens=["<|endoftext|>"])
    tokenizer.train_from_iterator(get_training_corpus(), trainer=trainer)
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
    sentence = "Let's test this tokenizer."
    encoding = tokenizer.encode(sentence)
    start, end = encoding.offsets[4]
    tokenizer.decoder = decoders.ByteLevel()
    print(tokenizer.decode(encoding.ids))
    wrapped_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token="<|endoftext|>",
        eos_token="<|endoftext|>",
    )

### UNIGRAM TOKENIZER ###

def unigram():
    tokenizer = Tokenizer(models.Unigram())
    tokenizer.normalizer = normalizers.Sequence(
        [
            normalizers.Replace("``", '"'),
            normalizers.Replace("''", '"'),
            normalizers.NFKD(),
            normalizers.StripAccents(),
            normalizers.Replace(Regex(" {2,}"), " "),
        ]
    )
    tokenizer.pre_tokenizer = pre_tokenizers.Metaspace()
    special_tokens = ["<cls>", "<sep>", "<unk>", "<pad>", "<mask>", "<s>", "</s>"]
    trainer = trainers.UnigramTrainer(
        vocab_size=25000, special_tokens=special_tokens, unk_token="<unk>"
    )
    tokenizer.model = models.Unigram()
    tokenizer.train_from_iterator(get_training_corpus(), trainer=trainer)
    encoding = tokenizer.encode("Let's test this tokenizer.")
    print(encoding.tokens)
    cls_token_id = tokenizer.token_to_id("<cls>")
    sep_token_id = tokenizer.token_to_id("<sep>")
    tokenizer.post_processor = processors.TemplateProcessing(
        single="$A:0 <sep>:0 <cls>:2",
        pair="$A:0 <sep>:0 $B:1 <sep>:1 <cls>:2",
        special_tokens=[("<sep>", sep_token_id), ("<cls>", cls_token_id)],
    )
    encoding = tokenizer.encode("Let's test this tokenizer...", "on a pair of sentences!")
    print(encoding.tokens)
    print(encoding.type_ids)
    tokenizer.decoder = decoders.Metaspace()
    wrapped_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
        cls_token="<cls>",
        sep_token="<sep>",
        mask_token="<mask>",
        padding_side="left",
    )

def multiling():
    pass

if __name__ == "__main__":
    unigram()