"""
Shared setup for the CFPB evaluation rebuild.

Every artefact produced here is cached to _cache/ so that a failure in a later
stage costs seconds to retry instead of re-deriving the whole pipeline.

Design decision (fair comparison): all 10 architectures train on the SAME
stratified TRAIN_BUDGET subset and are scored on the FULL test split. The
budget is set by what BERT can afford; applying it to every model is what makes
the master table an apples-to-apples comparison.

Cache trust: every file under _cache/ is written by this script on this machine.
The joblib / allow_pickle loads below deserialise only those self-produced
artefacts, never third-party data. Delete _cache/ to force a clean rebuild.
"""
import json
import os
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(PROJECT, "_cache")
PARQUET = os.path.join(PROJECT, "df_model_preprocessed.parquet")
NB_PATH = os.path.join(PROJECT, "CFPB_Complaint_Classification_v2.ipynb")

os.makedirs(CACHE, exist_ok=True)

TRAIN_BUDGET = 200_000
MAX_LEN = 256
VOCAB_SIZE = 30_000
EMBED_DIM = 100
TFIDF_FEATURES = 25_000
SEED = 42
NUM_CLASSES = 9


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def cpath(name):
    return os.path.join(CACHE, name)


def cached(name, build, loader=np.load, saver=None):
    """Return the cached artefact, building and persisting it on first call."""
    path = cpath(name)
    if os.path.exists(path):
        log(f"  cache hit  {name}")
        return loader(path)
    log(f"  building   {name} ...")
    t0 = time.time()
    obj = build()
    (saver or np.save)(path, obj)
    log(f"  built      {name} in {time.time() - t0:.1f}s")
    return obj


def save_json(name, obj):
    with open(cpath(name), "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)


def load_json(name):
    path = cpath(name)
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_splits():
    """Stratified 70/15/15 split, label encoding, and the shared train budget subset.

    Mirrors notebook cells 27-28 exactly (same seed, same stratification) so the
    partitions here are identical to the ones documented in Section 3.5.
    """
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder

    meta = load_json("splits_meta.json")
    if meta and all(os.path.exists(cpath(f)) for f in meta["files"]):
        log("  cache hit  splits")
        d = {k: np.load(cpath(f), allow_pickle=True) for k, f in zip(meta["keys"], meta["files"])}
        d["classes"] = np.array(meta["classes"])
        return d

    log("  building   splits (loading parquet) ...")
    t0 = time.time()
    df = pd.read_parquet(PARQUET, columns=["product_9", "narrative_classical", "narrative_contextual"])
    log(f"    parquet loaded: {len(df):,} rows in {time.time() - t0:.1f}s")

    train_df, temp_df = train_test_split(
        df, test_size=0.30, random_state=SEED, stratify=df["product_9"]
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, random_state=SEED, stratify=temp_df["product_9"]
    )
    del temp_df, df

    encoder = LabelEncoder().fit(train_df["product_9"])
    y_train_full = encoder.transform(train_df["product_9"])
    y_val = encoder.transform(val_df["product_9"])
    y_test = encoder.transform(test_df["product_9"])

    # Shared budget subset, stratified so the class prior is preserved
    sub_idx, _ = train_test_split(
        np.arange(len(train_df)),
        train_size=TRAIN_BUDGET,
        stratify=y_train_full,
        random_state=SEED,
    )
    sub_idx.sort()

    out = {
        "y_train_full": y_train_full,
        "y_train": y_train_full[sub_idx],
        "y_val": y_val,
        "y_test": y_test,
        "sub_idx": sub_idx,
        "train_classical": train_df["narrative_classical"].values[sub_idx],
        "train_contextual": train_df["narrative_contextual"].values[sub_idx],
        "train_classical_full": train_df["narrative_classical"].values,
        "val_classical": val_df["narrative_classical"].values,
        "val_contextual": val_df["narrative_contextual"].values,
        "test_classical": test_df["narrative_classical"].values,
        "test_contextual": test_df["narrative_contextual"].values,
    }

    files, keys = [], []
    for k, v in out.items():
        fn = f"split_{k}.npy"
        np.save(cpath(fn), v)
        files.append(fn)
        keys.append(k)
    save_json("splits_meta.json", {"keys": keys, "files": files, "classes": list(encoder.classes_)})
    out["classes"] = np.array(encoder.classes_)
    log(f"  built      splits in {time.time() - t0:.1f}s")
    return out


class TokenCorpus:
    """Streaming tokeniser - avoids materialising ~90M token strings at once."""

    def __init__(self, texts):
        self.texts = texts

    def __iter__(self):
        for t in self.texts:
            yield t.split()


def build_tfidf(S):
    """TF-IDF fitted on the FULL training split (matches notebook Section 3.6).

    Only the classifier training budget is capped at TRAIN_BUDGET; the
    representation itself sees all 1.41M training narratives. Still zero
    leakage - validation and test are transform-only.
    """
    import joblib
    from sklearn.feature_extraction.text import TfidfVectorizer

    if all(os.path.exists(cpath(f)) for f in ["tfidf_train.npz", "tfidf_val.npz", "tfidf_test.npz"]):
        log("  cache hit  tfidf")
        return (
            sp.load_npz(cpath("tfidf_train.npz")),
            sp.load_npz(cpath("tfidf_val.npz")),
            sp.load_npz(cpath("tfidf_test.npz")),
            joblib.load(cpath("tfidf_vectorizer.joblib")),
        )

    log("  building   tfidf ...")
    t0 = time.time()
    vec = TfidfVectorizer(
        max_features=TFIDF_FEATURES, ngram_range=(1, 2), sublinear_tf=True, min_df=5, max_df=0.85
    )
    vec.fit(S["train_classical_full"])
    Xtr = vec.transform(S["train_classical"])
    Xva = vec.transform(S["val_classical"])
    Xte = vec.transform(S["test_classical"])
    sp.save_npz(cpath("tfidf_train.npz"), Xtr)
    sp.save_npz(cpath("tfidf_val.npz"), Xva)
    sp.save_npz(cpath("tfidf_test.npz"), Xte)
    joblib.dump(vec, cpath("tfidf_vectorizer.joblib"))
    log(f"  built      tfidf {Xtr.shape} in {time.time() - t0:.1f}s")
    return Xtr, Xva, Xte, vec


def build_sequences(S):
    """Word2Vec CBOW embeddings + padded integer sequences for the recurrent models.

    Sequence lengths are stored alongside the ids so the recurrent models can
    mask padding via pack_padded_sequence instead of reading a hidden state
    that has been walked through up to 256 PAD steps.
    """
    from collections import Counter

    from gensim.models import Word2Vec

    names = ["seq_train.npy", "seq_val.npy", "seq_test.npy",
             "len_train.npy", "len_val.npy", "len_test.npy", "embedding_matrix.npy"]
    if all(os.path.exists(cpath(f)) for f in names):
        log("  cache hit  sequences")
        return tuple(np.load(cpath(f)) for f in names)

    log("  building   word2vec + sequences (fit on full training split) ...")
    t0 = time.time()
    corpus = TokenCorpus(S["train_classical_full"])
    w2v = Word2Vec(
        sentences=corpus, vector_size=EMBED_DIM, window=5, min_count=5,
        workers=4, epochs=5, seed=SEED,
    )
    log(f"    word2vec vocab: {len(w2v.wv):,} (from {len(S['train_classical_full']):,} docs)")

    counts = Counter()
    for toks in corpus:
        counts.update(toks)
    # Only admit words Word2Vec actually learned (min_count=5). Taking the top
    # 30k by raw frequency instead would hand ~10k rare slots a random vector
    # that no pretrained embedding ever fills.
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for w, c in counts.most_common(VOCAB_SIZE - 2):
        if c < 5:
            break
        vocab[w] = len(vocab)

    def encode(texts):
        ids = np.zeros((len(texts), MAX_LEN), dtype=np.int32)
        lens = np.ones(len(texts), dtype=np.int32)
        unk = vocab["<UNK>"]
        for i, t in enumerate(texts):
            toks = t.split()[:MAX_LEN]
            if toks:
                ids[i, : len(toks)] = [vocab.get(tok, unk) for tok in toks]
                lens[i] = len(toks)
        return ids, lens

    seq_tr, len_tr = encode(S["train_classical"])
    seq_va, len_va = encode(S["val_classical"])
    seq_te, len_te = encode(S["test_classical"])

    rng = np.random.default_rng(SEED)
    emb = rng.normal(scale=0.1, size=(len(vocab), EMBED_DIM)).astype(np.float32)
    emb[0] = 0.0
    matched = 0
    for w, i in vocab.items():
        if w in w2v.wv:
            emb[i] = w2v.wv[w]
            matched += 1
    log(f"    embedding coverage: {matched:,}/{len(vocab):,}")

    for fn, arr in zip(names, [seq_tr, seq_va, seq_te, len_tr, len_va, len_te, emb]):
        np.save(cpath(fn), arr)
    save_json("vocab_meta.json", {"vocab_size": len(vocab), "w2v_vocab": len(w2v.wv), "matched": matched})
    log(f"  built      sequences in {time.time() - t0:.1f}s")
    return seq_tr, seq_va, seq_te, len_tr, len_va, len_te, emb


def class_weights(y):
    from sklearn.utils.class_weight import compute_class_weight

    return compute_class_weight("balanced", classes=np.unique(y), y=y)


def metrics(y_true, y_pred):
    from sklearn.metrics import accuracy_score, f1_score

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def setup(need_seq=True, need_tfidf=True):
    log("=== Stage: setup ===")
    S = load_splits()
    log(f"  train budget {len(S['y_train']):,} | val {len(S['y_val']):,} | test {len(S['y_test']):,}")
    out = {"S": S}
    if need_tfidf:
        out["tfidf"] = build_tfidf(S)
    if need_seq:
        out["seq"] = build_sequences(S)
    return out


if __name__ == "__main__":
    d = setup()
    S = d["S"]
    log("Setup complete.")
    log(f"  classes: {list(S['classes'])}")
    if "tfidf" in d:
        log(f"  tfidf train: {d['tfidf'][0].shape}  test: {d['tfidf'][2].shape}")
    if "seq" in d:
        log(f"  seq train: {d['seq'][0].shape}  test: {d['seq'][2].shape}")
