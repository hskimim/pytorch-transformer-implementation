# https://eagle705.github.io/articles/2019-05/SentencePiece

import sentencepiece as spm
templates = '--input={} --model_prefix={} --vocab_size={} --control_symbols=[CLS],[SEP] --user_defined_symbols=[MASK] --pad_id=0 --bos_id=1 --eos_id=2 --unk_id=3 --input_sentence_size=3000000 --train_extremely_large_corpus=true'
vocab_size = 10000
prefix = 'm'
input_file = './wikitext-2-raw/prep_wiki.train.raw'

cmd = templates.format(input_file, prefix, vocab_size)

spm.SentencePieceTrainer.Train(cmd)
