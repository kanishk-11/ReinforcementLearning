import collections
import os
import csv


class CsvWriter:
    """A logging object writing to a CSV file.

    Each `write()` takes a `OrderedDict`, creating one column in the CSV file for
    each dictionary key on the first call. Successive calls to `write()` must
    contain the same dictionary keys.
    """

    def __init__(self, fname: str):
        if fname is not None and fname != '':
            dirname = os.path.dirname(fname)
            if not os.path.exists(dirname):
                os.makedirs(dirname)

        self._fname = fname
        self._header_written = False
        self._fieldnames = None

    def write(self, values: collections.OrderedDict) -> None:
        if self._fname is None or self._fname == '':
            return

        if self._fieldnames is None:
            self._fieldnames = values.keys()
        with open(self._fname, 'a', encoding='utf8') as file_:
            writer = csv.DictWriter(file_, fieldnames=self._fieldnames)
            if not self._header_written:
                writer.writeheader()
                self._header_written = True
            writer.writerow(values)

    def close(self) -> None:
        """Closes the `CsvWriter`."""