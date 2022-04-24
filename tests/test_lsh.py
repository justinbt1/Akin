import pytest
from akin import MinHash, LSH
from collections import defaultdict

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

minhash = MinHash(content, seed=3)


def test_initialize_from_empty_lsh():
    lsh = LSH()
    assert lsh.no_of_bands is None
    assert lsh._buckets == defaultdict(list)
    assert lsh._i_bucket == defaultdict(list)
    assert lsh.permutations is None
    lsh.update(minhash, labels)
    assert list(lsh._i_bucket) == labels
    assert lsh.permutations == 100
    assert lsh.no_of_bands == 50


def test_initialize_lsh_with_params():
    lsh = LSH(minhash, labels, no_of_bands=20)
    assert lsh.no_of_bands == 20
    assert lsh.permutations == 100
    assert list(lsh._i_bucket) == labels


def test_lsh_query():
    lsh = LSH(minhash, labels)
    with pytest.raises(KeyError):
        lsh.query(10)
    with pytest.raises(KeyError):
        lsh.query(0)
    with pytest.raises(ValueError):
        lsh.query(2, sensitivity=100)
    result = lsh.query(1)
    assert result == [8, 4]
    result = lsh.query(1, sensitivity=29)
    assert result == [4]
    result = lsh.query(1, min_jaccard=0.55)
    assert result == [4]


def test_update_lsh():
    lsh = LSH(minhash, labels)
    with pytest.raises(ValueError):
        lsh.update(minhash, labels)
    new_content = [
        'Jupiter is primarily composed of hydrogen with a quarter of its mass being helium',
        'Jupiter moving out of the inner Solar System would have allowed the formation of inner planets.'
    ]
    new_labels = [11, 12]
    incorrect_minhash = MinHash(new_content, permutations=10)
    with pytest.raises(ValueError):
        lsh.update(incorrect_minhash, new_labels)
    correct_minhash = MinHash(new_content)
    lsh.update(correct_minhash, new_labels)
    assert lsh.permutations == 100
    assert list(lsh._i_bucket) == labels + [11, 12]


def test_lsh_contains():
    lsh = LSH(minhash, labels)
    assert lsh.contains() == labels


def test_lsh_remove():
    lsh = LSH(minhash, labels)
    lsh.remove(5)
    assert list(lsh._i_bucket) == [1, 2, 3, 4, 6, 7, 8, 9]
    with pytest.raises(KeyError):
        lsh.remove(11)


def test_lsh_adjacency_list():
    lsh = LSH(minhash, labels)
    with pytest.raises(ValueError):
        lsh.adjacency_list(sensitivity=1000)
    sensitivity_list = lsh.adjacency_list(sensitivity=2)
    assert sensitivity_list == {
        1: [8, 4], 2: [], 3: [5], 4: [1, 8], 5: [3], 6: [], 7: [], 8: [1, 4], 9: []
    }
    jaccard_list = lsh.adjacency_list(min_jaccard=0.6)
    assert jaccard_list == {
        1: [], 2: [], 3: [5], 4: [], 5: [3], 6: [], 7: [], 8: [], 9: []
    }
    default_list = lsh.adjacency_list()
    assert default_list == {
        1: [8, 4], 2: [], 3: [5], 4: [1, 8], 5: [3], 6: [], 7: [], 8: [1, 4], 9: []
    }


def test_lsh_errors():
    with pytest.raises(ValueError):
        LSH(content)
    with pytest.raises(ValueError):
        LSH(labels=labels)
    with pytest.raises(ValueError):
        LSH(minhash, labels, no_of_bands=49)
