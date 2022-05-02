import pytest
from akin import minhash
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
    multi_hash = minhash.MultiHash(content)
    assert type(multi_hash.signatures) is np.ndarray
    assert multi_hash.signatures.shape == (9, 100)
    assert multi_hash.n_gram == 9
    assert multi_hash.n_gram_type == 'char'
    assert multi_hash.permutations == 100
    assert multi_hash.hash_bits == 64
    assert multi_hash.hash_seeds.shape[0] == 100


def multi_hash_tests(first_hash, second_hash, hash_size):
    multi_hash = minhash.MultiHash(
        content, hash_bits=hash_size, seed=seed
    )
    assert multi_hash.seed == 3
    assert type(multi_hash.signatures) is np.ndarray
    assert multi_hash.signatures.shape == (9, 100)
    signature = multi_hash.signatures
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
    bottom_k_hash = minhash.BottomK(
        content,
        permutations=53,
        hash_bits=hash_size,
        seed=seed
    )

    assert bottom_k_hash.seed == seed
    assert type(bottom_k_hash.signatures) is np.ndarray
    assert bottom_k_hash.signatures.shape == (9, 53)

    signature = bottom_k_hash.signatures
    assert signature[0][0] == first_hash
    assert signature[-1][-1] == second_hash

    with pytest.raises(ValueError):
        minhash.BottomK(
            content,
            permutations=200,
            hash_bits=hash_size,
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
    multi_hash = minhash.MultiHash(content, n_gram_type='term', seed=seed)

    assert multi_hash.n_gram_type == 'term'
    assert type(multi_hash.signatures) is np.ndarray

    signature = multi_hash.signatures
    assert signature.shape == (9, 100)
    assert signature[0][0] == -8115786556000760185
    assert np.array(signature[0][0]).dtype == 'int64'
    assert signature[-1][-1] == -579511180950999701


def test_string_input_minhash():
    multi_hash = minhash.MultiHash(content[0])
    assert type(multi_hash.signatures) is np.ndarray
    assert multi_hash.signatures.shape == (1, 100)


def test_minhash_errors():
    with pytest.raises(ValueError):
        minhash.MinHash(content, n_gram_type='words')
    with pytest.raises(ValueError):
        minhash.MinHash(content, hash_bits=65)
    # with pytest.raises(ValueError):
    #     minhash.BottomK(content, n_gram=630)
