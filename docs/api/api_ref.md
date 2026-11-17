# Akin API Reference

## MinHash

Akin provides `KMinHash` and `OPMinHash` for creating fixed-length signatures
from text. Construct a MinHash object with its configuration, then call
`transform` with one string or an iterable of strings. The result is a list of
signature tuples, one per input text.

Both classes create overlapping shingles. With `n_gram_type="char"`, shingles
contain characters; with `n_gram_type="term"`, they contain whitespace-separated
terms. Duplicate shingles within a text are collapsed.

### Common parameters

`n_gram` `int, optional, default: 9`  
Number of characters or terms in each shingle. Must be at least 2. A text
without a complete shingle raises `ValueError`.

`n_gram_type` `str, optional, default: "char"`  
Use `"char"` for character shingles or `"term"` for whitespace-separated
term shingles.

`permutations` `int, optional, default: 100`  
Number of values in each signature. Must be a positive integer.

`hash_bits` `int, optional, default: 64`  
Hash width. Supported values are 32, 64, and 128 bits.

`seed` `int, optional, default: 1`  
Seed used to generate hashes. Must be in the range `0` to `2**32 - 1`.
Use the same seed and MinHash settings to generate reproducible, compatible
signatures with the same MinHash algorithm.

### `OPMinHash`

```python
akin.OPMinHash(n_gram=9, n_gram_type="char", permutations=100,
               hash_bits=64, seed=1)
```

Uses one-permutation hashing (OPH): each unique shingle is hashed once and
assigned to one of `permutations` bins. The minimum hash in each bin is used
in the signature. Empty bins are filled through deterministic densification.

#### Methods

`.transform(text_corpus)`  
Generate signatures for a string or iterable of strings. Returns a list of
tuples, each containing `permutations` unsigned integer hash values.

#### Attributes

`n_gram`, `n_gram_type`, `permutations`, `hash_bits`, and `seed` contain the
values used to configure the MinHash object.

### `KMinHash`

```python
akin.KMinHash(n_gram=9, n_gram_type="char", permutations=100,
              hash_bits=64, seed=1)
```

Uses an independent seeded hash function for each signature position. Each
position contains the minimum hash value for that function across the text's
unique shingles.

#### Methods

`.transform(text_corpus)`  
Generate signatures for a string or iterable of strings. Returns a list of
tuples, each containing `permutations` unsigned integer hash values.

#### Attributes

`n_gram`, `n_gram_type`, `permutations`, `hash_bits`, and `seed` contain the
values used to configure the MinHash object. `hash_seeds` contains the
generated seed for each signature position.

## LSH

`LSH` indexes labelled MinHash signatures by splitting each signature into
bands and hashing each band into a bucket. Documents that share at least one
bucket are returned as candidates.

```python
akin.LSH(permutations, jaccard_similarity=None, probability=0.95,
         n_bands=None, n_rows=None, seed=1)
```

### Parameters

`permutations` `int`  
Length of each MinHash signature indexed by this object.

`jaccard_similarity` `float, optional, default: None`  
Target similarity used to select banding parameters automatically. This is
not a query-time similarity threshold.

`probability` `float, optional, default: 0.95`  
Target probability of retrieving a candidate at `jaccard_similarity`.
Used only when selecting banding parameters automatically.

`n_bands` `int, optional, default: None`  
Explicit number of bands. Must be provided with `n_rows`.

`n_rows` `int, optional, default: None`  
Number of signature values per band. Must be provided with `n_bands`, and
`n_bands * n_rows` must equal `permutations`.

`seed` `int, optional, default: 1`  
Seed used to hash bands into buckets.

Provide both `n_bands` and `n_rows`, or provide `jaccard_similarity` for
automatic band selection.

### Methods

`.update(minhash_signatures, labels)`  
Add signatures and their corresponding labels to the index. The number of
labels must match the number of signatures. Labels must be unique and
hashable, and signatures must have the configured length and compatible
MinHash algorithm, seed, and settings.

`.query(label, include_similarity=False)`  
Return candidate labels for an indexed document, excluding the queried
label. Candidate ordering is not guaranteed. If `include_similarity=True`,
return `(estimated_similarity, label)` tuples. The estimate is the fraction
of matching signature positions; no similarity threshold is applied.

`.remove(labels)`  
Remove the indexed documents with the supplied iterable of labels.

### Attributes

`n_bands`  
Number of bands in the index.

`n_rows`  
Number of signature values in each band.

`permutations`  
Expected length of each indexed MinHash signature.

`seed`  
Seed used to hash bands.

`keys`  
Mapping from indexed labels to their stored signatures.
