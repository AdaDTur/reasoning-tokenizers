import psutil
import numpy as np
from datasets import load_dataset
import os
from transformers import PreTrainedTokenizerFast
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

def batch_iter_texts(ds_split, batch_size=1000, text_builder=None, text_col="text"):
    n = len(ds_split)
    for i in range(0, n, batch_size):
        batch = ds_split[i:i+batch_size]
        if text_builder is not None:
            yield [text_builder(batch[j]) for j in range(len(batch[text_col if text_col in batch.column_names else batch.column_names[0]]))]
        else:
            yield batch[text_col]

def culturaX_with_char_limit(ds, char_limit=10_000_000_000, batch_size=1000):
    total_chars = 0
    for batch in batch_iter_texts(ds['train'], batch_size=batch_size, text_builder=None, text_col="text"):
        batch_chars = sum(len(x) for x in batch)
        if total_chars + batch_chars > char_limit:
            remaining = char_limit - total_chars
            out = []
            for x in batch:
                if remaining <= 0:
                    break
                if len(x) <= remaining:
                    out.append(x)
                    remaining -= len(x)
                else:
                    out.append(x[:remaining])
                    remaining = 0
            if out:
                yield out
            break
        else:
            total_chars += batch_chars
            yield batch

def gsm8k_text_builder(row):
    raise NotImplementedError

def mbpp_text_builder(row):
    raise NotImplementedError

def build_rowwise_builder(columns, kind):
    if kind == "gsm8k":
        has_text = "text" in columns
        has_q = "question" in columns
        has_a = "answer" in columns
        def fn(batch, idx):
            if has_text:
                return batch["text"][idx]
            q = batch["question"][idx] if has_q else ""
            a = batch["answer"][idx] if has_a else ""
            return f"Q:\n{q}\n\nA:\n{a}"
        return fn
    elif kind == "mbpp":
        has_text = "text" in columns
        has_prompt = "prompt" in columns
        has_code = "code" in columns or "solution" in columns
        code_field = "code" if "code" in columns else ("solution" if "solution" in columns else None)
        def fn(batch, idx):
            if has_text:
                return batch["text"][idx]
            prompt = batch["prompt"][idx] if has_prompt else ""
            code = batch[code_field][idx] if code_field else ""
            return f"# Prompt\n{prompt}\n\n# Solution\n{code}"
        return fn
    else:
        raise ValueError("unknown kind")

def batch_iter_texts_flexible(ds_split, kind, batch_size=1000):
    n = len(ds_split)
    cols = ds_split.column_names
    for i in range(0, n, batch_size):
        batch = ds_split[i:i+batch_size]
        builder = build_rowwise_builder(cols, kind)
        m = len(batch[cols[0]]) if len(cols) else 0
        yield [builder(batch, j) for j in range(m)]

def combined_corpus_iter(culturax_ds, gsm8k_ds, mbpp_ds, char_limit, batch_size=1000, max_sentence_length=1000):
    for batch in culturaX_with_char_limit(culturax_ds, char_limit=char_limit, batch_size=batch_size):
        # Filter out sentences that are too long
        filtered_batch = [text for text in batch if len(text) <= max_sentence_length]
        if filtered_batch:
            yield filtered_batch
    if gsm8k_ds is not None and 'train' in gsm8k_ds:
        for batch in batch_iter_texts_flexible(gsm8k_ds['train'], kind="gsm8k", batch_size=batch_size):
            # Filter out sentences that are too long
            filtered_batch = [text for text in batch if len(text) <= max_sentence_length]
            if filtered_batch:
                yield filtered_batch
    if mbpp_ds is not None and 'train' in mbpp_ds:
        for batch in batch_iter_texts_flexible(mbpp_ds['train'], kind="mbpp", batch_size=batch_size):
            # Filter out sentences that are too long
            filtered_batch = [text for text in batch if len(text) <= max_sentence_length]
            if filtered_batch:
                yield filtered_batch

def create_bin_files(name, lang, tokenizer, iterator, train_split=0.9):
    all_tokens = []
    for batch in iterator:
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
    os.makedirs(f'.gitignore/tokenizers/{name}/{lang}/', exist_ok=True)
    train_tokens.tofile(f'.gitignore/tokenizers/{name}/{lang}/{name}_{lang}_train.bin')
    val_tokens.tofile(f'.gitignore/tokenizers/{name}/{lang}/{name}_{lang}_val.bin')
    print(f"Training tokens: {len(train_tokens):,}")
    print(f"Validation tokens: {len(val_tokens):,}")

def bpe(corpus_iterator, lang):
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    trainer = trainers.BpeTrainer(vocab_size=25000, special_tokens=["<|endoftext|>"])
    tokenizer.train_from_iterator(corpus_iterator, trainer=trainer)
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
    tokenizer.decoder = decoders.ByteLevel()
    wrapped = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token="<|endoftext|>",
        eos_token="<|endoftext|>",
    )
    wrapped.save_pretrained(f".gitignore/tokenizers/bpe/{lang}")
    return wrapped

def wordpiece(corpus_iterator, lang):
    print("Starting WordPiece training")
    tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]"))
    tokenizer.normalizer = normalizers.BertNormalizer(lowercase=True)
    special_tokens = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]
    trainer = trainers.WordPieceTrainer(vocab_size=25000, special_tokens=special_tokens)
    tokenizer.train_from_iterator(corpus_iterator, trainer=trainer)
    print("Finished WordPiece training")
    cls_id = tokenizer.token_to_id("[CLS]")
    sep_id = tokenizer.token_to_id("[SEP]")
    tokenizer.post_processor = processors.TemplateProcessing(
        single=f"[CLS]:0 $A:0 [SEP]:0",
        pair=f"[CLS]:0 $A:0 [SEP]:0 $B:1 [SEP]:1",
        special_tokens=[("[CLS]", cls_id), ("[SEP]", sep_id)],
    )
    tokenizer.decoder = decoders.WordPiece(prefix="##")
    wrapped = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer
    )
    wrapped.save_pretrained(f".gitignore/tokenizers/wordpiece/wordpiece_{lang}")
    return wrapped

def unigram(corpus_iterator, lang, max_sentence_length=10000):
    print("Starting Unigram training")
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
    
    filtered_corpus = []
    for batch in corpus_iterator:
        for text in batch:
            if len(text) <= max_sentence_length:
                filtered_corpus.append(text)
    
    print(f"Training on {len(filtered_corpus)} sentences (filtered from original corpus)")
    tokenizer.train_from_iterator(filtered_corpus, trainer=trainer)
    print("Finished Unigram training")
    cls_id = tokenizer.token_to_id("<cls>")
    sep_id = tokenizer.token_to_id("<sep>")
    tokenizer.post_processor = processors.TemplateProcessing(
        single="$A:0 <sep>:0 <cls>:2",
        pair="$A:0 <sep>:0 $B:1 <sep>:1 <cls>:2",
        special_tokens=[("<sep>", sep_id), ("<cls>", cls_id)],
    )
    tokenizer.decoder = decoders.Metaspace()
    
    # Save the tokenizer
    wrapped = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="<unk>",
        pad_token="<pad>",
        cls_token="<cls>",
        sep_token="<sep>",
        mask_token="<mask>",
        bos_token="<s>",
        eos_token="</s>",
    )
    os.makedirs(f".gitignore/tokenizers/unigram/{lang}", exist_ok=True)
    wrapped.save_pretrained(f".gitignore/tokenizers/unigram/{lang}")
    return wrapped

def char_level(corpus_iterator, lang):
    specials = ["<unk>"]
    charset = set()
    for batch in corpus_iterator:
        for text in batch:
            charset.update(list(text))
    vocab = {tok: i for i, tok in enumerate(specials + sorted(charset))}
    model = models.WordLevel(vocab=vocab, unk_token="<unk>")
    tokenizer = Tokenizer(model)
    tokenizer.pre_tokenizer = pre_tokenizers.Split(Regex(r""), behavior="isolated")
    wrapped = PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="<unk>")
    os.makedirs(f".gitignore/tokenizers/char/char_{lang}", exist_ok=True)
    wrapped.save_pretrained(f".gitignore/tokenizers/char/char_{lang}")
    return wrapped

if __name__ == "__main__":
    langs = ['tr']
    for lang in langs:
        culturax_fp = f".gitignore/culturax/{lang}/{lang}_part_00000.parquet"
        gsm8k_train_fp = ".gitignore/clean_datasets/gsm8k_train.parquet"
        mbpp_train_fp = ".gitignore/clean_datasets/mbpp_train.parquet"

        culturax_ds = load_dataset('parquet', data_files=culturax_fp)
        gsm8k_ds = load_dataset('parquet', data_files={"train": gsm8k_train_fp})
        mbpp_ds = load_dataset('parquet', data_files={"train": mbpp_train_fp})

        corpus_iter = combined_corpus_iter(
            culturax_ds,
            gsm8k_ds,
            mbpp_ds,
            char_limit=10_000_000_000,
            batch_size=1000,
            max_sentence_length=1000
        )

        tok = unigram(corpus_iter, lang, max_sentence_length=1000)

        corpus_iter_for_bins = combined_corpus_iter(
            culturax_ds, gsm8k_ds, mbpp_ds, char_limit=10_000_000_000, batch_size=1000, max_sentence_length=1000
        )
        create_bin_files('unigram', lang, tok, corpus_iter_for_bins, train_split=0.9)