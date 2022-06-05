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
    'The Great Red Spot is large enough to accommodate Earth within its boundaries.',
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

    for hash_array in lsh._buckets._hash_arrays:
        assert signatures[0] not in hash_array.values()

    lsh.remove(signatures[1:-1])

    with pytest.raises(KeyError):
        lsh.remove([signatures[-1]])


def test_query():
    lsh = LSH(no_of_bands=10, permutations=20)

    with pytest.raises(ValueError):
        lsh.query(signatures[-2], sensitivity=21)

    with pytest.raises(KeyError):
        lsh.query(signatures[0])

    lsh.update(signatures)
    query_result = lsh.query(signatures[-1], min_jaccard=1.0)
    assert query_result == []
    query_result = lsh.query(signatures[-1], min_jaccard=0.1)
    assert query_result == []

    lsh.update(signatures)
    query_result = lsh.query(signatures[0], min_jaccard=1.0)
    assert query_result == []
    query_result = lsh.query(signatures[0], min_jaccard=0.6)
    assert query_result == [(
        -9050934246571064385,
        -9214906801195985462,
        -8886235969777720404,
        -9169161817144622757,
        -9129879279378013046,
        -8911093137960266343,
        -9118940880135677683,
        -9159818792685820463,
        -9184556596179116285,
        -8842993503590404334,
        -8988083533769668685,
        -9060753285625582072,
        -9167546554048977930,
        -8397275405286038812,
        -9102429460330209546,
        -9195348980737651916,
        -9052201375734666520,
        -8856160147624896787,
        -9178165847508784427,
        -8487002183883993654
    )]
    query_result = lsh.query(signatures[0], min_jaccard=0.45)
    assert query_result == [(
        -9050934246571064385,
        -9214906801195985462,
        -8886235969777720404,
        -9169161817144622757,
        -9129879279378013046,
        -8911093137960266343,
        -9118940880135677683,
        -9159818792685820463,
        -9184556596179116285,
        -8842993503590404334,
        -8988083533769668685,
        -9060753285625582072,
        -9167546554048977930,
        -8397275405286038812,
        -9102429460330209546,
        -9195348980737651916,
        -9052201375734666520,
        -8856160147624896787,
        -9178165847508784427,
        -8487002183883993654
        ),
        (
        -8798265333996157434,
        -8962219986627421974,
        -9213179238729627175,
        -9169161817144622757,
        -9129879279378013046,
        -8911093137960266343,
        -9118940880135677683,
        -7287678765549662618,
        -9184556596179116285,
        -8842993503590404334,
        -8988083533769668685,
        -9060753285625582072,
        -9173365861169047783,
        -8550435620119424225,
        -9054449385245629433,
        -8812194805157508727,
        -9052201375734666520,
        -8856160147624896787,
        -9185691847327566933,
        -8487002183883993654
    )]


def test_minhashes():
    lsh = LSH(no_of_bands=10, permutations=20)
    lsh.update(signatures)

    retrieved_hashes = lsh.get_minhashes()
    assert set(signatures) == retrieved_hashes


def test_adjacency_list():
    pass
