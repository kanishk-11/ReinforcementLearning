class AttributeDict(dict):
    """A `dict` that supports getting, setting, deleting keys via attributes."""

    def __getattr__(self, key):
        return self[key]

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        del self[key]