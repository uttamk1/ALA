import math
import unittest

from vector import Vector


class TestVector(unittest.TestCase):
    def test_mean(self):
        
        vector = Vector([1, -2, 3, 2])
        self.assertAlmostEqual(vector.mean(), 1.0)

    def test_mean_of_constant_vector(self):
        vector = Vector([5, 5, 5])
        self.assertAlmostEqual(vector.mean(), 5.0)

    def test_mean_after_adding_constant(self):
        # mean(x + c) = mean(x) + c.
        vector = Vector([1, 2, 3])
        shifted = Vector([6, 7, 8])
        self.assertAlmostEqual(shifted.mean(), vector.mean() + 5)

    def test_mean_of_sum(self):
        # mean(x + y) = mean(x) + mean(y).
        x = Vector([1, 2, 3])
        y = Vector([3, 4, 5])
        added = Vector([4, 6, 8])
        self.assertAlmostEqual(added.mean(), x.mean() + y.mean())

    def test_mean_after_scaling(self):
        # mean(a*x) = a*mean(x).
        vector = Vector([1, 2, 3])
        scaled = Vector([-2, -4, -6])
        self.assertAlmostEqual(scaled.mean(), -2 * vector.mean())

    def test_mean_is_between_smallest_and_largest_entries(self):
        vector = Vector([-3, 2, 7])
        self.assertGreaterEqual(vector.mean(), min(vector.entries))
        self.assertLessEqual(vector.mean(), max(vector.entries))

    def test_demean(self):
        
        vector = Vector([1, -2, 3, 2])
        self.assertEqual(vector.demean().entries, [0.0, -3.0, 2.0, 1.0])

    def test_demeaned_vector_has_zero_mean_and_sum(self):
        # Zero sum means orthogonality to the all-ones vector.
        vector = Vector([1, 2, 4])
        centered = vector.demean()
        self.assertAlmostEqual(centered.mean(), 0.0)
        self.assertAlmostEqual(sum(centered.entries), 0.0)

    def test_constant_vector_demeans_to_zero(self):
        vector = Vector([5, 5, 5])
        self.assertEqual(vector.demean().entries, [0.0, 0.0, 0.0])

    def test_demean_after_adding_constant(self):
        vector = Vector([1, 2, 3])
        shifted = Vector([6, 7, 8])
        self.assertEqual(shifted.demean().entries, vector.demean().entries)

    def test_demeaning_twice_has_no_further_effect(self):
        centered = Vector([1, 2, 3]).demean()
        self.assertEqual(centered.demean().entries, centered.entries)

    def test_demean_preserves_original(self):
        vector = Vector([1, 2, 3])
        centered = vector.demean()
        self.assertIsInstance(centered, Vector)
        self.assertIsNot(centered, vector)
        self.assertEqual(vector.entries, [1, 2, 3])

    def test_std(self):
        # variance = (0 + 9 + 4 + 1) / 4 = 3.5.
        vector = Vector([1, -2, 3, 2])
        self.assertAlmostEqual(vector.std(), math.sqrt(3.5))

    def test_std_uses_population_denominator(self):
        # Mean = 2; squared deviations = [1, 1]; population std = 1.
        
        vector = Vector([1, 3])
        self.assertAlmostEqual(vector.std(), 1.0)

    def test_std_of_constant_vector(self):
        vector = Vector([5, 5, 5])
        self.assertAlmostEqual(vector.std(), 0.0)

    def test_nonconstant_vector_has_positive_std(self):
        # Together with the constant-vector test, checks the zero-std property.
        vector = Vector([-3, 2, 7])
        self.assertGreater(vector.std(), 0.0)

    def test_std_after_adding_constant(self):
        # Adding a constant changes the mean, but preserves the spread.
        vector = Vector([1, 3])
        shifted = Vector([6, 8])
        self.assertAlmostEqual(shifted.std(), vector.std())

    def test_std_after_scaling(self):
        # std(a*x) = abs(a)*std(x).
        vector = Vector([1, 3])
        positive_scale = Vector([2, 6])
        negative_scale = Vector([-2, -6])
        zero_scale = Vector([0, 0])
        self.assertAlmostEqual(positive_scale.std(), 2 * vector.std())
        self.assertAlmostEqual(negative_scale.std(), 2 * vector.std())
        self.assertAlmostEqual(zero_scale.std(), 0.0)

    def test_demeaning_preserves_std(self):
        vector = Vector([1, 2, 4])
        self.assertAlmostEqual(vector.demean().std(), vector.std())

    def test_std_equals_demeaned_norm_divided_by_sqrt_length(self):
        # std(x) = ||demean(x)|| / sqrt(n).
        vector = Vector([1, 2, 4])
        centered = vector.demean()
        norm = math.sqrt(sum(entry ** 2 for entry in centered.entries))
        self.assertAlmostEqual(vector.std(), norm / math.sqrt(len(vector)))

    def test_mean_square_equals_variance_plus_squared_mean(self):
        #  rms(x)^2 = std(x)^2 + mean(x)^2.
        vector = Vector([1, 2, 4])
        mean_square = (1 ** 2 + 2 ** 2 + 4 ** 2) / 3
        self.assertAlmostEqual(mean_square, vector.std() ** 2 + vector.mean() ** 2)


if __name__ == "__main__":
    unittest.main()
