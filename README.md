# Akin
![Python Version](https://img.shields.io/badge/Python-3.6%20%7C%203.12-blue.svg)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Build Status](https://app.travis-ci.com/justinbt1/Akin.svg?branch=main)](https://app.travis-ci.com/justinbt1/Akin)
[![Downloads](https://static.pepy.tech/badge/akin)](https://pepy.tech/project/akin)
<br>
Python library for detecting near duplicate texts in a corpus at scale using Locality Sensitive Hashing, 
adapted from the algorithm described in chapter three of 
[Mining Massive Datasets](http://infolab.stanford.edu/~ullman/mmds/ch3.pdf). This algorithm identifies similar texts in 
a corpus efficiently by estimating their Jaccard similarity with sub-linear time complexity. This can be used to detect 
near duplicate texts at scale or locate different versions of a document.  

### Warning!
Note this library is in a beta 0. version and may be subject to frequent breaking changes!  
Please pin your versions to avoid these changes from breaking your code.

### Installation
Install from PyPI using pip:
```python3 pip install akin```  
Note for the ARM Apple M1 architecture some dependencies may need to be installed separately using conda.

### API Documentation
See the [full documentation here](https://github.com/justinbt1/Akin/blob/dev/docs/api_documentation.md) for API 
and usage guide.

### Quick Start Example
``` python
from akin import MultiHash, LSH

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


# Generate MinHash signatures.
minhash = MultiHash(n_gram=9, permutations=100, hash_bits=64, seed=3)
signatures  minhash.transform(content)

# Create LSH model.
lsh = LSH(no_of_bands=50)
lsh.update(signatures)

# Query to find near duplicates for text 1.
print(lsh.query(1, min_jaccard=0.5))
>>> [8, 4]

# Generate minhash signature and add new texts to LSH model.
new_text = [
    'Jupiter is primarily composed of hydrogen with a quarter of its mass being helium',
    'Jupiter moving out of the inner Solar System would have allowed the formation of '
    'inner planets.'
]

new_labels = ['doc1', 'doc2']

new_minhash = MinHash(new_text, n_gram=9, permutations=100, hash_bits=64, seed=3)

lsh.update(new_minhash, new_labels)

# Check contents of documents.
print(lsh.contains())
>>> [1, 2, 3, 4, 5, 6, 7, 8, 9, 'doc1', 'doc2']

# Remove text and label from model.
lsh.remove(5)
print(lsh.contains())
>>> [1, 2, 3, 4, 6, 7, 8, 9, 'doc1', 'doc2']

# Return adjacency list for all similar texts.
adjacency_list = lsh.adjacency_list(min_jaccard=0.55)
print(adjacency_list)
>>> {
        1: ['doc1', 4], 2: ['doc2'], 3: [], 4: [1, 'doc1'], 6: [], 
        7: [], 8: [], 9: [], 'doc1': [1, 4], 'doc2': [2]
    }
```
