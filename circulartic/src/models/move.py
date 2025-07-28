"""
Move model for circular tic-tac-toe game.
"""
from dataclasses import dataclass
from typing import Optional
import time
from .enums import Player


@dataclass
class Move:
    """
    Represents a move in the circular tic-tac-toe game.
    
    Attributes:
        position: Position ID (0-31) where the move is made
        player: Player making the move
        timestamp: Time when the move was made
        evaluation_score: AI's evaluation of this move (optional)
    """
    position: int
    player: Player
    timestamp: float = None
    evaluation_score: Optional[float] = None
    
    def __post_init__(self):
        """Set timestamp if not provided and validate parameters."""
        if self.timestamp is None:
            self.timestamp = time.time()
        
        if not (0 <= self.position <= 31):
            raise ValueError(f"Position must be between 0 and 31, got {self.position}")
        
        if not isinstance(self.player, Player):
            raise ValueError(f"Player must be a Player enum, got {type(self.player)}")
    
    def __str__(self) -> str:
        """String representation of the move."""
        score_str = f" (score: {self.evaluation_score:.2f})" if self.evaluation_score is not None else ""
        return f"{self.player.value} -> position {self.position}{score_str}"