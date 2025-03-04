# Getting Started
## Installation
Install from PyPI using pip: 
```shell
python3 -m pip install akin
```
## Quick Start Example
```python
from akin import UniMinHash, LSH

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

labels = [i for i in range(1, len(content))]

# Generate MinHash signatures.
minhash = UniMinHash(n_gram=9, permutations=100, hash_bits=64, seed=3)
signatures  minhash.transform(content)

# Create LSH model.
lsh = LSH(permutations=minhash.permutations)
lsh.update(signatures, labels)

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

# Remove text and label from model.
lsh.remove(5)

# Return adjacency list for all similar texts.
adjacency_list = lsh.adjacency_list(min_jaccard=0.55)
print(adjacency_list)
>>> {
        1: ['doc1', 4], 2: ['doc2'], 3: [], 4: [1, 'doc1'], 6: [], 
        7: [], 8: [], 9: [], 'doc1': [8, 1, 4], 'doc2': [2]
    }
```