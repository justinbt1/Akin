import pytest
from akin import _data_structures


def test_dictionary_array():
    dictionary_array = _data_structures.DictionaryArray(10)
    assert dictionary_array.n_arrays == 10
    assert len(dictionary_array._hash_arrays) == 10

    with pytest.raises(ValueError):
        _data_structures.DictionaryArray(0)

    assert type(dictionary_array._hash_arrays) == list
    assert type(dictionary_array._hash_arrays[0]) == dict

    dictionary_array.update(3, 'test_key', 'test_value')
    assert dictionary_array._hash_arrays[3] == {'test_key': ['test_value']}
