"""
Position model for circular tic-tac-toe board.
"""
from dataclasses import dataclass
from typing import Optional
from .enums import Player


@dataclass
class Position:
    """
    Represents a position on the circular tic-tac-toe board.
    
    Attributes:
        id: Unique identifier for the position (0-31)
        ring: Ring number (0=center, 1=inner, 2=middle, 3=outer)
        angle: Angular position within the ring (0.0 to 360.0 degrees)
        occupant: Player occupying this position, or None if empty
    """
    id: int
    ring: int
    angle: float
    occupant: Optional[Player] = None
    
    def __post_init__(self):
        """Validate position parameters."""
        if not (0 <= self.id <= 31):
            raise ValueError(f"Position id must be between 0 and 31, got {self.id}")
        if not (0 <= self.ring <= 3):
            raise ValueError(f"Ring must be between 0 and 3, got {self.ring}")
        if not (0.0 <= self.angle < 360.0):
            raise ValueError(f"Angle must be between 0.0 and 360.0, got {self.angle}")
    
    def is_empty(self) -> bool:
        """Check if the position is empty."""
        return self.occupant is None
    
    def is_occupied_by(self, player: Player) -> bool:
        """Check if the position is occupied by a specific player."""
        return self.occupant == player