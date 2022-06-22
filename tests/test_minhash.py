import pytest
from akin import minhash

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
    multi_hash = minhash.MultiHash()
    assert multi_hash.n_gram == 9
    assert multi_hash.n_gram_type == 'char'
    assert multi_hash.permutations == 100
    assert multi_hash.hash_bits == 64
    assert multi_hash.hash_seeds.shape[0] == 100


def multi_hash_tests(first_hash, second_hash, hash_size):
    multi_hash = minhash.MultiHash(hash_bits=hash_size, seed=seed)
    assert multi_hash.seed == 3
    signatures = multi_hash.transform(content)
    assert type(signatures) is list
    assert type(signatures[0]) is tuple
    assert len(signatures) == 9
    assert len(signatures[0]) == 100
    assert signatures[0][0] == first_hash
    assert signatures[-1][-1] == second_hash


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
        permutations=53,
        hash_bits=hash_size,
        seed=seed
    )

    assert bottom_k_hash.seed == seed

    signatures = bottom_k_hash.transform(content)

    assert type(signatures) is list
    assert type(signatures[0]) is tuple
    assert len(signatures) == 9
    assert len(signatures[0]) == 53

    assert signatures[0][0] == first_hash
    assert signatures[-1][-1] == second_hash


def test_k_minhash_64():
    k_smallest_hash_tests(
        -8690990394074104221,
        3450127865886110699,
        64
    )


def test_k_minhash_32():
    k_smallest_hash_tests(
        -1850189223,
        1059578105,
        32
    )


def test_k_minhash_128():
    k_smallest_hash_tests(
        4622532303668074294069371384102143164,
        279003082584568259907909477798660768049,
        128
    )


def test_k_minhash_errors():
    with pytest.raises(ValueError):
        bottom_k_hash = minhash.BottomK(permutations=100)
        bottom_k_hash.transform(content)


def test_terms_minhash():
    multi_hash = minhash.MultiHash(n_gram_type='term', seed=seed)
    signatures = multi_hash.transform(content)

    assert multi_hash.n_gram_type == 'term'
    assert type(signatures) is list
    assert type(signatures[0]) is tuple
    assert len(signatures) == 9
    assert len(signatures[0]) == 100
    assert type(signatures[0][0]) is int
    assert signatures[0][0] == -8115786556000760185
    assert signatures[-1][-1] == -579511180950999701


def test_string_input_minhash():
    multi_hash = minhash.MultiHash()
    signatures = multi_hash.transform(content[0])
    assert type(signatures) is list
    assert type(signatures[0]) is tuple
    assert len(signatures) == 1
    assert len(signatures[0]) == 100


def test_minhash_errors():
    with pytest.raises(ValueError):
        minhash.MinHash(n_gram_type='words')

    with pytest.raises(ValueError):
        minhash.MinHash(hash_bits=65)

    multi_hash = minhash.MultiHash(n_gram=62)
    multi_hash.transform(content)

    with pytest.raises(ValueError):
        multi_hash = minhash.MultiHash(n_gram=63)
        multi_hash.transform(content)
