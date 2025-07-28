from transformers import BertTokenizerFast
from transformers import GPT2TokenizerFast

wrapped_tokenizer = BertTokenizerFast(tokenizer_object='wordpiece_english.json')
wrapped_tokenizer = GPT2TokenizerFast(tokenizer_object='')