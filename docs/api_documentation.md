# Akin API Guide

### MinHash
Creates a MinHash object that contains matrix of Minhash Signatures for each text.


#### MinHash Parameters
``` python
MinHash(
    text, 
    method='multi_hash', 
    n_gram=9, 
    n_gram_type='char', 
    permutations=100, 
    hash_bits=64, 
    seed=None
)
```  
**text**  
`{list or ndarray}`  
Iterable containing strings of text for each text in a corpus.<br><br>
**method**  
`str, optional, default: 'multi_hash'`  
Method for random sampling via hashing, must be 'multi_hash' or 'k_smallest_values'.<br>
If multi_hash selected texts are hashed once per permutation and the minimum hash value selected each time to 
construct a signature.<br>  
If k_smallest_values selected each text is hashed once and k smallest values selected for k permutations. 
This method is less computationally intensive than multi_hash but also less stable.<br><br>
**n_gram**  
`int, optional, default: 9`  
Size of each overlapping text shingle to break text into prior to hashing. Shingle size should be carefully selected 
dependent on average text length as too low a shingle size will yield false similarities, whereas too high a shingle 
size will fail to return similar documents.  

For character shingles a size of 5 is recommended for shorter texts such as emails, the default size of 9 is 
recommended for longer texts or documents.<br><br>
**n_gram_type**  
`str, optional, default: 'char'`  
Type of n gram to use for shingles, must be 'char' to split text into character shingles or 'term' to split text into 
overlapping sequences of words.<br><br>
**permutations**  
`int, optional, default: 100`  
Number of randomly sampled hash values to use for generating each texts minhash signature. Intuitively the larger the 
number of permutations, the more accurate the estimated Jaccard similarity between the texts but longer the algorithm 
will take to run.<br><br>
**hash_bits**  
`int, optional, default: 64`  
Hash value size to be used to generate minhash signatures from shingles, must be 32, 64 or 128 bit. Hash value size 
should be chosen based on text length and a trade off between performance and accuracy. Lower hash values risk false 
hash collisions leading to false similarities between documents for larger corpora of texts.<br><br>
**seed**  
`int, optional, default: None`  
Seed from which to generate random hash function, necessary for reproducibility or to allow updating of the LSH model 
with new minhash values later.<br><br>

#### MinHash Properties
**n_gram:** `int`  
``` python 
.n_gram
```  
Returns size of each overlapping text shingle used to create minhash signatures.<br><br>
**n_gram_type:** `int`  
``` python 
.n_gram_type
```  
Returns type of n-gram used for text shingling.<br><br>
**permutations:** `int`  
``` python 
.permutations
```  
Returns number of permutations used to create signatures.<br><br>
**hash_bits:** `int`  
``` python 
.hash_bits
```  
Returns hash value size used to create signatures.<br><br>
**method:** `str`  
``` python 
.method
```  
Returns hashing method used in minhash function.<br><br>
**seed:** `int`  
``` python 
.seed
```  
Returns seed value used to generate random hashes in minhash function.<br><br>
**signatures:** `numpy.array`  
``` python 
.signatures
```  
Returns matrix of text signatures generated by minhash function.<br>
n = text row, m = selected permutations.<br>

### LSH
Creates an LSH model of text similarity that can be used to return similar texts based on estimated Jaccard similarity.

#### LSH Parameters
``` python
LSH(minhash=None, labels=None, no_of_bands=None)
```  
**minhash**  
`optional, default: None`  
Minhash object containing minhash signatures returned by MinHash class.<br><br>
**labels**  
`{list or ndarray}, optional, default: None`  
List, array or Pandas series containing unique labels for each text in minhash object signature. This should be provided in the same order as texts passed to the MinHash class. Example labels include filepaths and database ids.<br><br>
**no_of_bands**  
`optional, default: permutations // 2`  
Number of bands to break minhash signature into before hashing into buckets. A smaller number of bands will result in a stricter algorithm, requiring larger possibly leading to false negatives missing some similar texts, whereas a higher number may lead to false similarities. <br><br>

#### LSH Methods
<b>update</b><br>
Updates model from a MinHash object containing signatures generated from new texts and their corresponding labels.<br>
``` python
.update(minhash, new_labels)
```  
<b>minhash:</b> MinHash object containing signatures of new texts, parameters must match any previous MinHash objects.<br>
<b>new_labels:</b> List, array or Pandas series containing text labels.<br><br>
<b>query</b><br>
Takes a label and returns the labels of any similar texts.<br>
``` python
.query(label, min_jaccard=None, sensitivity=1)
```
<b>label:</b> Label of text to return list of similar texts for.<br>
<b>min_jaccard:</b> Jaccard similarity threshold texts have to exceed to be returned as similar.<br>
<b>sensitivity:</b> Number of buckets texts must share to be returned as similar.<br><br>
<b>remove</b><br>
Remove file label and minhash signature from model.<br>
``` python
.remove(label)
```  
<b>label:</b> Label of text to remove from LSH model.<br><br>
<b>contains</b><br>
Returns list of labels contained in the model.<br>
``` python
.contains()
```  
<b>adjacency_list</b><br>
Returns an adjacency list that can be used to create a text similarity graph.<br>
``` python
.adjacency_list(min_jaccard=None, sensitivity=1)
```  
<b>min_jaccard:</b> Jaccard similarity threshold texts have to exceed to be returned as similar.<br>
<b>sensitivity:</b> Number of buckets texts must share to be returned as similar.<br><br>

#### LSH Properties
**no_of_bands:** `int`  
``` python
.no_of_bands
```  
Number of bands used in LSH model.<br><br>
**permutations:** `int`  
``` python
.permutations
```  
Number of permutations used to create minhash signatures used in LSH model.