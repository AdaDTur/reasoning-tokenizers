import sys
import psutil
import numpy as np
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

def get_training_corpus_memory_threshold(ds, memory_limit_gb=0.5):
    memory_limit_bytes = memory_limit_gb * 1024 * 1024 * 1024
    batch_size = 1000
    total_batches = 0
    
    for i in range(0, len(ds['train']), batch_size):
        process = psutil.Process()
        current_memory = process.memory_info().rss
        
        if current_memory > memory_limit_bytes:
            print(f"Memory threshold reached: {current_memory / (1024**3):.2f} GB")
            print(f"Total batches processed: {total_batches}")
            break
            
        batch = ds['train'][i : i + batch_size]["text"]
        total_batches += 1
        yield batch

def get_training_corpus_char_threshold(ds, char_limit=500_000_000):
    batch_size = 1000
    total_chars = 0
    total_batches = 0
    
    for i in range(0, len(ds['train']), batch_size):
        batch = ds['train'][i : i + batch_size]["text"]
        
        batch_chars = sum(len(text) for text in batch)
        
        if total_chars + batch_chars > char_limit:
            print(f"Character threshold reached: {total_chars:,} characters")
            print(f"Total batches processed: {total_batches}")
            break
            
        total_chars += batch_chars
        total_batches += 1
        yield batch

def create_bin_files(name, ds, lang, tokenizer, train_split=0.9):
    all_tokens = []
    
    for batch in get_training_corpus_char_threshold(ds):
        for text in batch:
            tokens = tokenizer.encode(text)
            if hasattr(tokens, 'ids'):
                all_tokens.extend(tokens.ids)
            else:
                all_tokens.extend(tokens)
    
    all_tokens = np.array(all_tokens, dtype=np.uint16)
    
    split_idx = int(len(all_tokens) * train_split)
    train_tokens = all_tokens[:split_idx]
    val_tokens = all_tokens[split_idx:]
    
    train_tokens.tofile(f'{name}_{lang}_train.bin')
    val_tokens.tofile(f'{name}_{lang}_val.bin')
    
    print(f"Training tokens: {len(train_tokens):,}")
    print(f"Validation tokens: {len(val_tokens):,}")

### WORDPIECE TOKENIZER ###

def wordpiece(ds, lang, use_memory_threshold=True):
    tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]"))
    tokenizer.normalizer = normalizers.BertNormalizer(lowercase=True)
    
    special_tokens = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]
    trainer = trainers.WordPieceTrainer(vocab_size=25000, special_tokens=special_tokens)
    
    corpus_func = get_training_corpus_memory_threshold if use_memory_threshold else get_training_corpus_char_threshold
    tokenizer.train_from_iterator(corpus_func(ds), trainer=trainer)
    
    cls_token_id = tokenizer.token_to_id("[CLS]")
    sep_token_id = tokenizer.token_to_id("[SEP]")

    tokenizer.post_processor = processors.TemplateProcessing(
        single=f"[CLS]:0 $A:0 [SEP]:0",
        pair=f"[CLS]:0 $A:0 [SEP]:0 $B:1 [SEP]:1",
        special_tokens=[("[CLS]", cls_token_id), ("[SEP]", sep_token_id)],
    )
    tokenizer.decoder = decoders.WordPiece(prefix="##")
    
    create_bin_files('wordpiece', ds, lang, tokenizer)

### BPE TOKENIZER ###

def bpe(ds, lang, use_memory_threshold=True):
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    trainer = trainers.BpeTrainer(vocab_size=25000, special_tokens=["<|endoftext|>"])
    
    corpus_func = get_training_corpus_memory_threshold if use_memory_threshold else get_training_corpus_char_threshold
    tokenizer.train_from_iterator(corpus_func(ds), trainer=trainer)
    
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
    tokenizer.decoder = decoders.ByteLevel()

    wrapped_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token="<|endoftext|>",
        eos_token="<|endoftext|>",
    )
    
    create_bin_files('bpe', ds, lang, wrapped_tokenizer)

### UNIGRAM TOKENIZER ###

def unigram(ds, lang, use_memory_threshold=True):
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
    
    corpus_func = get_training_corpus_memory_threshold if use_memory_threshold else get_training_corpus_char_threshold
    tokenizer.train_from_iterator(corpus_func(ds), trainer=trainer)
    
    cls_token_id = tokenizer.token_to_id("<cls>")
    sep_token_id = tokenizer.token_to_id("<sep>")
    tokenizer.post_processor = processors.TemplateProcessing(
        single="$A:0 <sep>:0 <cls>:2",
        pair="$A:0 <sep>:0 $B:1 <sep>:1 <cls>:2",
        special_tokens=[("<sep>", sep_token_id), ("<cls>", cls_token_id)],
    )

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
    
    create_bin_files('unigram', ds, lang, wrapped_tokenizer)

if __name__ == "__main__":
    langs = ['en', 'tr', 'es', 'fr', 'fi']
    ds = load_dataset('parquet', data_files="culturax/en/en_part_00000.parquet")
    bpe(ds, 'en')
    wordpiece(ds, 'en')
    unigram(ds, 'en')
