"""
Win pattern models for circular tic-tac-toe game.
"""
from dataclasses import dataclass
from typing import List, Optional
from .enums import Player, WinPatternType


@dataclass
class WinPattern:
    """
    Represents a winning pattern in circular tic-tac-toe.
    
    Attributes:
        type: Type of winning pattern (radial, circular, spiral, half-ring)
        positions: List of position IDs that form this pattern
        priority: Strategic importance of this pattern (higher = more important)
        description: Human-readable description of the pattern
    """
    type: WinPatternType
    positions: List[int]
    priority: int
    description: str = ""
    
    def __post_init__(self):
        """Validate pattern parameters."""
        if len(self.positions) != 4:
            raise ValueError(f"Win pattern must have exactly 4 positions, got {len(self.positions)}")
        
        for pos_id in self.positions:
            if not (0 <= pos_id <= 31):
                raise ValueError(f"Position ID must be between 0 and 31, got {pos_id}")
        
        if self.priority < 0:
            raise ValueError(f"Priority must be non-negative, got {self.priority}")
    
    def contains_position(self, position_id: int) -> bool:
        """Check if this pattern contains a specific position."""
        return position_id in self.positions
    
    def get_missing_positions(self, occupied_positions: List[int]) -> List[int]:
        """Get positions in this pattern that are not yet occupied."""
        return [pos for pos in self.positions if pos not in occupied_positions]
    
    def count_occupied_by_player(self, board_positions: List, player: Player) -> int:
        """Count how many positions in this pattern are occupied by a specific player."""
        count = 0
        for pos_id in self.positions:
            if 0 <= pos_id < len(board_positions):
                position = board_positions[pos_id]
                if position.is_occupied_by(player):
                    count += 1
        return count
    
    def is_blocked_by_opponent(self, board_positions: List, player: Player) -> bool:
        """Check if this pattern is blocked by the opponent."""
        opponent = Player.O if player == Player.X else Player.X
        for pos_id in self.positions:
            if 0 <= pos_id < len(board_positions):
                position = board_positions[pos_id]
                if position.is_occupied_by(opponent):
                    return True
        return False
    
    def __str__(self) -> str:
        """String representation of the pattern."""
        return f"{self.type.value.title()} pattern: {self.positions} (priority: {self.priority})"


@dataclass
class Threat:
    """
    Represents a threat (potential winning opportunity) on the board.
    
    Attributes:
        pattern: The winning pattern that creates this threat
        player: Player who has the threat
        completing_moves: Positions that would complete this threat
        severity: How immediate this threat is (4 = immediate win, 3 = one move away, etc.)
        blocked: Whether this threat is blocked by opponent
    """
    pattern: WinPattern
    player: Player
    completing_moves: List[int]
    severity: int
    blocked: bool = False
    
    def __post_init__(self):
        """Validate threat parameters."""
        if not (1 <= self.severity <= 4):
            raise ValueError(f"Severity must be between 1 and 4, got {self.severity}")
        
        for pos_id in self.completing_moves:
            if not (0 <= pos_id <= 31):
                raise ValueError(f"Completing move position must be between 0 and 31, got {pos_id}")
    
    def is_immediate_win(self) -> bool:
        """Check if this threat represents an immediate winning opportunity."""
        return self.severity == 4 and len(self.completing_moves) == 1 and not self.blocked
    
    def is_forcing(self) -> bool:
        """Check if this threat forces the opponent to respond."""
        return self.severity >= 3 and not self.blocked
    
    def __str__(self) -> str:
        """String representation of the threat."""
        status = "BLOCKED" if self.blocked else "ACTIVE"
        return f"{self.player.value} threat (severity {self.severity}): {self.completing_moves} - {status}"


@dataclass
class WinResult:
    """
    Represents the result of a winning condition check.
    
    Attributes:
        winner: Player who won
        winning_pattern: The pattern that created the win
        winning_positions: Specific positions that form the winning line
    """
    winner: Player
    winning_pattern: WinPattern
    winning_positions: List[int]
    
    def __post_init__(self):
        """Validate win result parameters."""
        if len(self.winning_positions) != 4:
            raise ValueError(f"Winning positions must have exactly 4 positions, got {len(self.winning_positions)}")
        
        for pos_id in self.winning_positions:
            if not (0 <= pos_id <= 31):
                raise ValueError(f"Winning position must be between 0 and 31, got {pos_id}")
        
        # Verify winning positions match the pattern
        if set(self.winning_positions) != set(self.winning_pattern.positions):
            raise ValueError("Winning positions must match the winning pattern positions")
    
    def __str__(self) -> str:
        """String representation of the win result."""
        return f"{self.winner.value} wins with {self.winning_pattern.type.value} pattern: {self.winning_positions}"