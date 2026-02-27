import sentencepiece as spm

spm.SentencePieceTrainer.train(
    input="corpus.txt",
    model_prefix="sp",
    vocab_size=800,
    model_type="bpe",
    character_coverage=1.0,
    bos_id=0,
    eos_id=1,
    unk_id=2,
    pad_id=3
)