import pytest
from akin import MinHash
import numpy as np

seed = 3
content = [
    'Jupiter is primarily composed of hydrogen with a quarter of its mass being helium',
    'Jupiter moving out of the inner Solar System would have allowed the formation of inner planets.',
    'A helium atom has about four times as much mass as a hydrogen atom, so the composition changes '
    'when described as the proportion of mass contributed by different atoms.',
    'Jupiter is primarily composed of hydrogen and a quarter of its mass being helium',
    'A helium atom has about four times as much mass as a hydrogen atom and the composition changes '
    'when described as a proportion of mass contributed by different atoms.',
    'Theoretical models indicate that if Jupiter had much more mass than it does at present, it would shrink.',
    'This process causes Jupiter to shrink by about 2 cm each year.',
    'Jupiter is mostly composed of hydrogen with a quarter of its mass being helium',
    'The Great Red Spot is large enough to accommodate Earth within its boundaries.'
]


def test_minhash_defaults():
    minhash = MinHash(content)
    assert type(minhash.signatures) is np.ndarray
    assert minhash.signatures.shape == (9, 100)
    assert minhash.n_gram == 9
    assert minhash.n_gram_type == 'char'
    assert minhash.permutations == 100
    assert minhash.hash_bits == 64
    assert minhash.method == 'multi_hash'
    assert minhash._hash_seeds.shape[0] == 100


def multi_hash_tests(first_hash, second_hash, hash_size):
    minhash = MinHash(
        content, hash_bits=hash_size, seed=seed
    )
    assert minhash.seed == 3
    assert minhash.method == 'multi_hash'
    assert type(minhash.signatures) is np.ndarray
    assert minhash.signatures.shape == (9, 100)
    signature = minhash.signatures
    assert signature[0][0] == first_hash
    assert signature[-1][-1] == second_hash


def test_multi_minhash_64():
    multi_hash_tests(
        -9050934246571064385,
        -9214867028879031438,
        64
    )


def test_multi_minhash_32():
    multi_hash_tests(
        -2146652248,
        -2083408229,
        32
    )


def test_multi_minhash_128():
    multi_hash_tests(
        6975552809044285838442055830789296621,
        1468352371533149607987200777331494350,
        128
    )


def k_smallest_hash_tests(first_hash, second_hash, hash_size):
    minhash = MinHash(
        content,
        permutations=53,
        hash_bits=hash_size,
        method='k_smallest_values',
        seed=seed
    )
    assert minhash._hash_seeds == 83957611
    assert minhash.method == 'k_smallest_values'
    assert type(minhash.signatures) is np.ndarray
    assert minhash.signatures.shape == (9, 53)
    signature = minhash.signatures
    assert signature[0][0] == first_hash
    assert signature[-1][-1] == second_hash
    with pytest.raises(ValueError):
        MinHash(
            content,
            permutations=200,
            hash_bits=hash_size,
            method='k_smallest_values',
            seed=seed
        )


def test_k_minhash_64():
    k_smallest_hash_tests(
        -9050934246571064385,
        5299643506028682639,
        64
    )


def test_k_minhash_32():
    k_smallest_hash_tests(
        -2146652248,
        1112636791,
        32
    )


def test_k_minhash_128():
    k_smallest_hash_tests(
        6975552809044285838442055830789296621,
        257973159872861001802369490457024221505,
        128
    )


def test_terms_minhash():
    minhash = MinHash(content, n_gram_type='term', seed=seed)
    assert minhash.n_gram_type == 'term'
    assert type(minhash.signatures) is np.ndarray
    signature = minhash.signatures
    assert signature.shape == (9, 100)
    assert signature[0][0] == -8115786556000760185
    assert np.array(signature[0][0]).dtype == 'int64'
    assert signature[-1][-1] == -579511180950999701


def test_string_input_minhash():
    minhash = MinHash(content[0])
    assert type(minhash.signatures) is np.ndarray
    assert minhash.signatures.shape == (1, 100)


def test_minhash_errors():
    with pytest.raises(ValueError):
        MinHash(content, n_gram_type='words')
    with pytest.raises(ValueError):
        MinHash(content, hash_bits=65)
    with pytest.raises(ValueError):
        MinHash(content, method='universal')
    with pytest.raises(ValueError):
        MinHash(content, n_gram=63)
