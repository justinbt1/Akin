# Akin
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Build Status](https://app.travis-ci.com/justinbt1/Akin.svg?branch=main)](https://app.travis-ci.com/justinbt1/Akin)
![PyPI - Downloads](https://img.shields.io/pypi/dm/akin)

<br>
Python library for detecting near duplicate texts in a corpus at scale using Locality Sensitive Hashing.  

<br>
<br>
This algorithm identifies similar texts in 
a corpus by efficiently estimating their Jaccard similarity with sub-linear time complexity. This can be used to detect 
near duplicate texts at scale or locate different versions of a document.  

### Installation
Install from PyPI using pip:
```
python3 -m pip install akin
```
Or with uv using:
```
uv add akin
```

### API Documentation
See the [API documentation here](https://justinbt1.github.io/Akin/api/api_ref/) for API 
and usage guide.

### Quick Start Example
``` python
from akin import OPMinHash, LSH

content = [
    'Jupiter is primarily composed of hydrogen with a quarter of its mass being helium',
    'Jupiter moving out of the inner Solar System would have allowed the formation of inner '
    'planets.',
    'A helium atom has about four times as much mass as a hydrogen atom, so the composition '
    'changes when described as the proportion of mass contributed by different atoms.',
    'Jupiter is primarily composed of hydrogen and a quarter of its mass being helium',
    'A helium atom has about four times as much mass as a hydrogen atom and the composition '
    'changes when described as a proportion of mass contributed by different atoms.',
    'Theoretical models indicate that if Jupiter had much more mass than it does at present, it '
    'would shrink.',
    'This process causes Jupiter to shrink by about 2 cm each year.',
    'Jupiter is mostly composed of hydrogen with a quarter of its mass being helium',
    'The Great Red Spot is large enough to accommodate Earth within its boundaries.'
]

labels = [i for i in range(0, len(content))]

# Generate MinHash signatures.
minhash = OPMinHash(n_gram=9, permutations=100, hash_bits=64, seed=3)
signatures = minhash.transform(content)

# Create LSH model.
lsh = LSH(permutations=100, jaccard_similarity=0.5, probability=0.95)
lsh.update(signatures, labels)

# Query to find near duplicates for text 1.
print(lsh.query(0))
>>> [3, 7]

# Generate minhash signature and add new texts to LSH model.
new_text = [
    'Jupiter is primarily composed of hydrogen with a quarter of its mass being helium',
    'Jupiter moving out of the inner Solar System would have allowed the formation of '
    'inner planets.'
]

new_labels = ['doc1', 'doc2']

new_signatures = minhash.transform(new_text)
lsh.update(new_signatures, new_labels)
print(lsh.query(0))
>>> ['doc_1', 3, 7]

# Remove text and label from model.
lsh.remove([7])
print(lsh.query(0))
>>> ['doc1', 3]
