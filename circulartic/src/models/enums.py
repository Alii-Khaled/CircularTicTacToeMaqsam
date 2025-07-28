"""
Core enums for the circular tic-tac-toe game.
"""
from enum import Enum


class Player(Enum):
    """Represents the two players in the game."""
    X = 'X'
    O = 'O'


class GameState(Enum):
    """Represents the current state of the game."""
    IN_PROGRESS = 'in_progress'
    X_WINS = 'x_wins'
    O_WINS = 'o_wins'
    DRAW = 'draw'


class WinPatternType(Enum):
    """Types of winning patterns in circular tic-tac-toe."""
    RADIAL = "radial"
    CIRCULAR = "circular"
    SPIRAL = "spiral"