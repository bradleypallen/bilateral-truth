"""
Truth value representations for the ζ_c function implementation.

Implements 3-valued logic components and generalized truth values
as described in the bilateral factuality evaluation framework.
"""

from enum import Enum
from typing import Tuple


class TruthValueComponent(Enum):
    """
    Three-valued logic components: true (t), undefined (e), false (f)
    """

    TRUE = "t"
    UNDEFINED = "e"
    FALSE = "f"


class GeneralizedTruthValue:
    """
    Generalized truth value <u,v> where:
    - u represents verifiability component
    - v represents refutability component
    Both u and v are elements of {t, e, f}
    """

    def __init__(
        self, verifiability: TruthValueComponent, refutability: TruthValueComponent
    ):
        """
        Initialize a generalized truth value.

        Args:
            verifiability: The verifiability component (u)
            refutability: The refutability component (v)
        """
        self.u = verifiability
        self.v = refutability

    def __repr__(self) -> str:
        return f"<{self.u.value},{self.v.value}>"

    def __eq__(self, other) -> bool:
        if not isinstance(other, GeneralizedTruthValue):
            return False
        return self.u == other.u and self.v == other.v

    def __hash__(self) -> int:
        return hash((self.u, self.v))

    @property
    def components(self) -> Tuple[TruthValueComponent, TruthValueComponent]:
        """Return the (u, v) components as a tuple."""
        return (self.u, self.v)

    @classmethod
    def true(cls) -> "GeneralizedTruthValue":
        """Create a generalized truth value representing classical truth."""
        return cls(TruthValueComponent.TRUE, TruthValueComponent.FALSE)

    @classmethod
    def false(cls) -> "GeneralizedTruthValue":
        """Create a generalized truth value representing classical falsehood."""
        return cls(TruthValueComponent.FALSE, TruthValueComponent.TRUE)

    @classmethod
    def undefined(cls) -> "GeneralizedTruthValue":
        """Create a generalized truth value representing undefined."""
        return cls(TruthValueComponent.UNDEFINED, TruthValueComponent.UNDEFINED)
