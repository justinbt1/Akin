from collections.abc import MutableMapping


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
            if not self._hash_arrays[array].get(key):
                self._hash_arrays[array][key] = []

    def remove_key(self, array, key):
        del self._hash_arrays[array][key]

    def remove_value(self, keys, value):
        for i, key in enumerate(keys):
            self._hash_arrays[i][key].remove(value)
            if not self._hash_arrays[i][key]:
                del self._hash_arrays[i][key]


class BiDirectionalDict(MutableMapping):
    def __init__(self, *args, **kwargs):
        super.__init__(*args, **kwargs)
        self.store = dict()
        self.inverse_store = dict()

    def __getitem__(self, item):
        return self.store.get(item)

    def __setitem__(self, key, value=None):
        if value is None:
            value = []
        self.store[key] = value

        for i in value:
            if self.inverse_store[value]:
                self.inverse_store[value].append(key)
            else:
                self.inverse_store[value] = [key]

    def __delitem__(self, value):
        keys = self.inverse_store[value]
        del self.inverse_store[value]

        for key in keys:
            self.store[key].remove(value)
            if not self.store[key]:
                del self.store[key]

    def __len__(self):
        pass

    def __iter__(self):
        pass

    def __repr__(self):
        pass
