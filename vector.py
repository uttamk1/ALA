"""A simple Vector class for Applied Linear Algebra Lab Assignment 1.

Example:
    vector = Vector([1, 2, 3])
    vector.mean()          # 2.0
    vector.demean().entries  # [-1.0, 0.0, 1.0]
    vector.std()           # sqrt(2 / 3)

Only Python's standard library is required.
"""

from math import fsum, sqrt


class Vector:
    """Store real-valued entries and calculate basic vector statistics."""

    def __init__(self, entries):
        """Copy an iterable of numbers so the input list is not shared."""
        self.entries = list(entries)

    def __len__(self):
        return len(self.entries)

    def __iter__(self):
        return iter(self.entries)

    def __getitem__(self, index):
        return self.entries[index]

    def __repr__(self):
        return f"Vector({self.entries!r})"

    def mean(self):
        """Return the sum of the entries divided by their count.

        Raise ValueError for an empty vector, whose mean is undefined.
        """
        if len(self) == 0:
            raise ValueError("Mean is undefined for an empty vector.")
        return fsum(self.entries) / len(self)

    def demean(self):
        """Return a new Vector with the mean subtracted from every entry.

        The original vector is unchanged. An empty vector raises ValueError.
        """
        average = self.mean()
        return Vector(entry - average for entry in self.entries)

    def std(self):
        """Return the population standard deviation using demean().

        Divide the sum of squared deviations by n, the number of entries.
        An empty vector raises ValueError.
        """
        centered = self.demean()
        variance = fsum(entry ** 2 for entry in centered) / len(self)
        return sqrt(variance)
