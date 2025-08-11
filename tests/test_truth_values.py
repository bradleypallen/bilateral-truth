"""Tests for truth value implementations."""

import unittest
from bilateral_truth.truth_values import TruthValueComponent, GeneralizedTruthValue


class TestTruthValueComponent(unittest.TestCase):
    """Test the TruthValueComponent enum."""

    def test_truth_value_components(self):
        """Test that all three components exist with correct values."""
        self.assertEqual(TruthValueComponent.TRUE.value, "t")
        self.assertEqual(TruthValueComponent.UNDEFINED.value, "e")
        self.assertEqual(TruthValueComponent.FALSE.value, "f")


class TestGeneralizedTruthValue(unittest.TestCase):
    """Test the GeneralizedTruthValue class."""

    def test_initialization(self):
        """Test creating generalized truth values."""
        gtv = GeneralizedTruthValue(TruthValueComponent.TRUE, TruthValueComponent.FALSE)
        self.assertEqual(gtv.u, TruthValueComponent.TRUE)
        self.assertEqual(gtv.v, TruthValueComponent.FALSE)

    def test_components_property(self):
        """Test the components property returns correct tuple."""
        gtv = GeneralizedTruthValue(
            TruthValueComponent.UNDEFINED, TruthValueComponent.TRUE
        )
        self.assertEqual(
            gtv.components, (TruthValueComponent.UNDEFINED, TruthValueComponent.TRUE)
        )

    def test_string_representation(self):
        """Test string representation."""
        gtv = GeneralizedTruthValue(TruthValueComponent.TRUE, TruthValueComponent.FALSE)
        self.assertEqual(str(gtv), "<t,f>")
        self.assertEqual(repr(gtv), "<t,f>")

    def test_equality(self):
        """Test equality comparison."""
        gtv1 = GeneralizedTruthValue(
            TruthValueComponent.TRUE, TruthValueComponent.FALSE
        )
        gtv2 = GeneralizedTruthValue(
            TruthValueComponent.TRUE, TruthValueComponent.FALSE
        )
        gtv3 = GeneralizedTruthValue(
            TruthValueComponent.FALSE, TruthValueComponent.TRUE
        )

        self.assertEqual(gtv1, gtv2)
        self.assertNotEqual(gtv1, gtv3)
        self.assertNotEqual(gtv1, "not a truth value")

    def test_hashing(self):
        """Test that truth values can be used as dictionary keys."""
        gtv1 = GeneralizedTruthValue(
            TruthValueComponent.TRUE, TruthValueComponent.FALSE
        )
        gtv2 = GeneralizedTruthValue(
            TruthValueComponent.TRUE, TruthValueComponent.FALSE
        )
        gtv3 = GeneralizedTruthValue(
            TruthValueComponent.FALSE, TruthValueComponent.TRUE
        )

        # Equal objects should have equal hashes
        self.assertEqual(hash(gtv1), hash(gtv2))

        # Test use in dictionary
        truth_dict = {gtv1: "classical true", gtv3: "classical false"}
        self.assertEqual(truth_dict[gtv2], "classical true")  # gtv2 equals gtv1

    def test_class_methods(self):
        """Test the class method constructors."""
        true_val = GeneralizedTruthValue.true()
        false_val = GeneralizedTruthValue.false()
        undefined_val = GeneralizedTruthValue.undefined()

        # Classical true: <t,f>
        self.assertEqual(true_val.u, TruthValueComponent.TRUE)
        self.assertEqual(true_val.v, TruthValueComponent.FALSE)

        # Classical false: <f,t>
        self.assertEqual(false_val.u, TruthValueComponent.FALSE)
        self.assertEqual(false_val.v, TruthValueComponent.TRUE)

        # Undefined/indeterminate: <e,e>
        self.assertEqual(undefined_val.u, TruthValueComponent.UNDEFINED)
        self.assertEqual(undefined_val.v, TruthValueComponent.UNDEFINED)


if __name__ == "__main__":
    unittest.main()
