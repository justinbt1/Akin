import pytest
from akin import DictionaryArray


def test_array_1():
    return [
        {},
        {},
        {},
        {
            'test_key': {'test_value', 'test_value_two'},
            'test_key_two': {'test_value_one'}
        },
        {111: {100}}
    ]


def test_dictionary_array():
    dictionary_array = DictionaryArray(10)
    assert dictionary_array.n_arrays == 10
    assert len(dictionary_array._hash_arrays) == 10

    with pytest.raises(ValueError):
        DictionaryArray(0)

    assert type(dictionary_array._hash_arrays) == list
    assert type(dictionary_array._hash_arrays[0]) == dict
    assert type(dictionary_array._hash_arrays[-1]) == dict


def test_update():
    dictionary_array = DictionaryArray(5)

    dictionary_array.update(3, 'test_key', 'test_value')
    assert dictionary_array._hash_arrays[3] == {'test_key': {'test_value'}}

    dictionary_array.update(3, 'test_key', 'test_value')
    assert dictionary_array._hash_arrays[3] == {'test_key': {'test_value'}}

    dictionary_array.update(3, 'test_key', 'test_value_two')
    assert dictionary_array._hash_arrays[3] == {'test_key': {'test_value', 'test_value_two'}}

    dictionary_array.update(3, 'test_key_two', 'test_value_one')
    assert dictionary_array._hash_arrays[3] == {
        'test_key': {'test_value', 'test_value_two'},
        'test_key_two': {'test_value_one'}
    }

    dictionary_array.update(4, 111, (100, 200))
    assert dictionary_array._hash_arrays[4] == {111: {(100, 200)}}

    test_array = test_array_1()
    test_array[4][111] = {(100, 200)}
    assert dictionary_array._hash_arrays == test_array

    dictionary_array.update(0, 'empty_key')
    assert dictionary_array.get(0, 'empty_key') == set()


def test_get_key():
    dictionary_array = DictionaryArray(5)

    dictionary_array._hash_arrays = test_array_1()

    assert dictionary_array.get(3, 'test_key') == {'test_value', 'test_value_two'}
    assert dictionary_array.get(3, 'test_key_two') == {'test_value_one'}
    assert dictionary_array.get(4, 111) == {100}


def test_remove_key():
    dictionary_array = DictionaryArray(5)

    dictionary_array._hash_arrays = test_array_1()

    dictionary_array.remove_key(4, 111)
    assert dictionary_array._hash_arrays[4] == {}

    dictionary_array.remove_key(3, 'test_key')
    assert dictionary_array._hash_arrays[3] == {'test_key_two': {'test_value_one'}}

    assert dictionary_array._hash_arrays == [
        {},
        {},
        {},
        {'test_key_two': {'test_value_one'}},
        {}
    ]


def test_remove_value():
    dictionary_array = DictionaryArray(5)

    dictionary_array._hash_arrays = [
        {'test_key_four': {'remove'}},
        {
            'test_key_three': {'test_value', 'test_value_two'},
            'test_key_four': {'remove'}
        },
        {'test_key_five': {'remove'}},
        {
            'test_key': {'test_value', 'remove'},
            'test_key_two': {'test_value_one'}
        },
        {111: {100}}
    ]

    dictionary_array.remove_value(0, 'test_key_four', 'remove')
    assert dictionary_array._hash_arrays[0] == {}

    dictionary_array.remove_value(1, 'test_key_four', 'remove')
    assert dictionary_array._hash_arrays[1] == {'test_key_three': {'test_value', 'test_value_two'}}

    dictionary_array.remove_value(2, 'test_key_five', 'remove')
    assert dictionary_array._hash_arrays[2] == {}

    dictionary_array.remove_value(3, 'test_key', 'remove')
    assert dictionary_array._hash_arrays[3] == {
        'test_key': {'test_value'},
        'test_key_two': {'test_value_one'}
    }


def test_get_values():
    dictionary_array = DictionaryArray(5)

    dictionary_array._hash_arrays = test_array_1()

    values = dictionary_array.values()
    assert values == {
        'test_value',
        'test_value_two',
        'test_value_one',
        100
    }
