# Text indexing and retrieval over a document corpus

**An inverted index, TF-IDF weighting and cosine ranking, built from scratch on 363
European Parliament debates in French — then three approximate search structures compared
on the same queries.**

No search library does the work here. The index, the weighting and the ranking are written
out, which is the point: the mechanics of a search engine are easier to reason about once
you have had to build them.

---

## The corpus

Europarl, French section — the transcribed debates of the European Parliament.

| | |
|---|---|
| Documents | **363** |
| Distinct terms, NLTK tokenisation | **112,515** |
| Distinct terms, regular-expression tokenisation | 81,638 |

The gap between the two tokenisers is itself a result. NLTK's `word_tokenize` keeps
apostrophes and punctuation-bound forms that `\b\w+\b` silently merges or drops — French
elisions such as `l'ordre` survive as one token instead of becoming `l` and `ordre`. Thirty
thousand extra terms is not noise; it is the vocabulary a naive regular expression throws
away.

---

## 1. The inverted index

For every term, the set of documents containing it:

```python
index[word].add(document_path)
```

That single structure is what makes search possible at all. Without it, answering
"which documents contain *commission*?" means reading all 363 files; with it, it is one
dictionary lookup.

Checked on three words of very different frequency:

| Term | Documents |
|---|---|
| `européenne` | 363 — every document |
| `indique` | 293 |
| `toto` | 2 |

`toto` appearing twice is the useful check: a term that should be absent is present in two
documents, which sends you to look at them rather than assuming the index is wrong.

A multi-word query intersects the posting lists — `"liberté humaine"` returns 311
documents, `"sensibilisation minorités"` returns 133.

---

## 2. Concordance — the term in its context

An index says *where*. It does not say *whether it is the sense you meant*. So each hit is
displayed with thirty characters either side, which is the classic keyword-in-context view:

```python
def afficher_contextes(chaine, terme, taille_contexte=30):
    for match in re.finditer(re.escape(terme), chaine):
        gauche = max(match.start() - taille_contexte, 0)
        yield chaine[gauche:match.end() + taille_contexte]
```

---

## 3. TF-IDF, computed rather than imported

Boolean presence ranks nothing: 363 documents contain `européenne`, and they cannot all be
equally relevant.

- **TF** — how often the term appears in a document, normalised by that document's length,
  so a long debate does not outrank a short one by volume alone.
- **IDF** — the inverse of how many documents contain the term, so a word present
  everywhere carries almost no information.

The verification step is the part worth keeping: **the term frequencies of every document
sum to 1.0000**, printed document by document. A weighting scheme that silently fails to
normalise produces a ranking that looks plausible and is wrong, and nothing in the output
would show it.

---

## 4. Ranking by cosine similarity

Query and document both become vectors in the term space; the score is the cosine of the
angle between them. Cosine rather than Euclidean distance on purpose — it compares
*direction*, so document length does not dominate the result.

Top of the ranking for `"liberté humaine"`:

```
0.017527  ep-01-11-29-fr.txt
0.006318  ep-00-03-30-fr.txt
0.004969  ep-02-09-23-fr.txt
0.004378  ep-01-09-12-fr.txt
0.003909  ep-00-04-10-fr.txt
```

---

## 5. Three search structures, measured

Scoring a query against all 363 documents is exhaustive search. It is exact and it does not
scale. Three standard alternatives, timed on the same query:

| Structure | Build time | Search time |
|---|---|---|
| **KD-Tree** | 5.45 s | 0.034 s |
| **K-Means** (10 clusters) | 10.11 s | **0.014 s** |
| **LSH** *(nearest neighbours, ball tree)* | 5.45 s | 0.065 s |

K-Means searches fastest because a query is compared against ten centroids before touching
any document; it pays for that in build time, which is twice the others.

### What I would do differently

The notebook concludes that KD-Tree is the most precise because its similarity scores are
highest. **That reasoning is wrong, and it is worth stating.** The three methods return
different document sets, so comparing their raw scores measures which documents each
happened to find, not how many of the right ones it found.

The correct comparison is against exhaustive search: run every query both ways, and measure
what fraction of the true top-k each approximate method recovers. That is recall@k, and it
is the only figure that separates "fast" from "fast and still right". It is the first thing
I would add.

---

## Running it

```bash
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt')"
jupyter notebook index_texte.ipynb
```

The corpus is not included — it is the French section of the
[Europarl parallel corpus](https://www.statmt.org/europarl/). Place it under
`data/europarl/fr/` and the notebook paths resolve.

The notebook produces `index.json`, `tf.json` and `idf.json`, which the later cells reload
rather than recompute.

---

## A note on this repository

The notebook was committed as a `.zip` because with its full outputs embedded it weighed
**119 MB** — above GitHub's 100 MB file limit. Truncating each output to its first few
thousand characters brings it to **60 KB** with every figure still visible, and the notebook
now renders directly on GitHub. The absolute paths from my own machine were replaced with
relative ones at the same time.

---

## Stack

Python · NLTK · scikit-learn · SciPy · NumPy

## Author

Mouhammad Thahir OUSMANE
