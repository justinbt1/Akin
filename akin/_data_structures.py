

class DictionaryArray:
    def __init__(self, n_arrays):
        if n_arrays < 1:
            raise ValueError('Number of arrays must be an integer of 1 or greater')

        self.n_arrays = n_arrays
        self._hash_arrays = [{}] * n_arrays

    def update(self, array, key, value=None):
        if value:
            if self._hash_arrays[array].get(key):
                self._hash_arrays[array][key].append(value)
            else:
                self._hash_arrays[array][key] = [value]
        else:
            if not self._hash_arrays[array].get(key):
                self._hash_arrays[array][key] = []

    def remove_key(self, array, key):
        del self._hash_arrays[array][key]

    def remove_value(self, keys, value):
        for i, key in enumerate(keys):
            self._hash_arrays[i][key] = [v for v in self._hash_arrays[i][key] if v != value]
            if not self._hash_arrays[i][key]:
                del self._hash_arrays[i][key]
