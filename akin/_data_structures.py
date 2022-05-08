
class DictionaryArray:
    def __init__(self, n_arrays):
        if n_arrays < 1:
            raise ValueError('Number of arrays must be an integer of 1 or greater')

        self._hash_arrays = [{}] * n_arrays

    def update(self, array, key, value=None):
        pass

    def remove_key(self, array, key):
        del self._hash_arrays[array][key]

    def remove_value(self, array, key, value):
        pass


class BiDirectionalDict:
    def __init__(self):
        pass


class BiDirectionalArray(DictionaryArray):
    def __init__(self, n_arrays):
        super(BiDirectionalArray, self).__init__(n_arrays)
        self._hash_arrays = [BiDirectionalDict()] * n_arrays
