import pytest
import mmh3
from akin import MultiHash, LSH, DictionaryArray

seed = 3
labels = [1, 2, 3, 4, 5, 6, 7, 8, 9]
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

minhash = MultiHash(seed=3, permutations=20)
signatures = minhash.transform(content)


def test_lsh_init_defaults():
    lsh = LSH(4)
    assert lsh.no_of_bands == 4
    assert lsh.seed == 1
    assert lsh._buckets._hash_arrays == DictionaryArray(4)._hash_arrays
    assert lsh.permutations is None


def test_lsh_init_params():
    lsh = LSH(4, permutations=100, seed=2)
    assert lsh.no_of_bands == 4
    assert lsh.seed == 2
    assert lsh._buckets._hash_arrays == DictionaryArray(4)._hash_arrays
    assert lsh.permutations is 100


def test_lsh_lsh():
    lsh = LSH(no_of_bands=5, permutations=9)
    signature = (45, 48, 21, 13, 29, 87, 43, 32, 12)
    bands = lsh._lsh(signature)
    bands = [band for band in bands]

    expected_bands = [
        mmh3.hash64(str((45, 48)), 1)[0],
        mmh3.hash64(str((21, 13)), 1)[0],
        mmh3.hash64(str((29, 87)), 1)[0],
        mmh3.hash64(str((43, 32)), 1)[0],
        mmh3.hash64(str((12,)), 1)[0]
    ]

    assert bands == expected_bands

    lsh = LSH(no_of_bands=5, permutations=10, seed=2)
    signature = (45, 48, 21, 13, 29, 87, 43, 32, 12, 10)
    bands = lsh._lsh(signature)
    bands = [band for band in bands]

    expected_bands = [
        mmh3.hash64(str((45, 48)), 2)[0],
        mmh3.hash64(str((21, 13)), 2)[0],
        mmh3.hash64(str((29, 87)), 2)[0],
        mmh3.hash64(str((43, 32)), 2)[0],
        mmh3.hash64(str((12, 10)), 2)[0]
    ]

    assert bands == expected_bands


def test_candidate_duplicates():
    lsh = LSH(4)

    signature = (13435, 54564, 54623, 41224, 21813)

    candidates = {
        (13435, 54564, 54623, 41224, 21813): 1,
        (13435, 54564, 54621, 41224, 21813): 2,
        (65435, 45435, 54545, 45876, 22312): 1,
        (65435, 65435, 65435, 65435, 15435): 3,
        (13435, 54564, 65435, 65435, 65435): 1,
    }

    matches = lsh._candidate_duplicates(signature, candidates, 1)
    assert matches == [
        (13435, 54564, 54623, 41224, 21813),
        (13435, 54564, 54621, 41224, 21813),
        (65435, 45435, 54545, 45876, 22312),
        (65435, 65435, 65435, 65435, 15435),
        (13435, 54564, 65435, 65435, 65435)
    ]

    matches = lsh._candidate_duplicates(signature, candidates, 2)
    assert matches == [
        (13435, 54564, 54621, 41224, 21813),
        (65435, 65435, 65435, 65435, 15435)
    ]

    matches = lsh._candidate_duplicates(signature, candidates, 3)
    assert matches == [(65435, 65435, 65435, 65435, 15435)]

    matches = lsh._candidate_duplicates(signature, candidates, 1, 0.33)
    assert matches == [
        (13435, 54564, 54623, 41224, 21813),
        (13435, 54564, 54621, 41224, 21813),
        (13435, 54564, 65435, 65435, 65435)
    ]

    matches = lsh._candidate_duplicates(signature, candidates, 1, 0.66)
    assert matches == [
        (13435, 54564, 54623, 41224, 21813),
        (13435, 54564, 54621, 41224, 21813)
    ]

    matches = lsh._candidate_duplicates(signature, candidates, 1, 1.0)
    assert matches == [(13435, 54564, 54623, 41224, 21813)]


def test_update():
    lsh = LSH(no_of_bands=10, permutations=20)
    lsh.update(signatures)

    lsh = LSH(no_of_bands=10)
    lsh.update(signatures)

    assert lsh.permutations == 20
    assert type(lsh._buckets._hash_arrays) is list
    assert len(lsh._buckets._hash_arrays) == 10
    assert type(lsh._buckets._hash_arrays[0]) is dict


def test_remove():
    lsh = LSH(no_of_bands=10, permutations=20)
    lsh.update(signatures)
    lsh.remove([signatures[0]])
    lsh.remove(signatures[1:])
    print(lsh._buckets._hash_arrays)
    assert False is True



def test_query():
    pass


def test_minhashes():
    pass


def test_adjacency_list():
    pass
