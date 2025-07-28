"""
Circular board representation for tic-tac-toe game.
"""
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple
import math
from ..models.position import Position
from ..models.enums import Player, GameState
from ..models.move import Move


@dataclass
class Ring:
    """
    Represents a ring on the circular board.
    
    Attributes:
        ring_id: Ring identifier (0=center, 1=inner, 2=middle, 3=outer)
        positions: List of positions in this ring
        size: Number of positions in this ring
    """
    ring_id: int
    positions: List[Position]
    size: int
    
    def __post_init__(self):
        """Validate ring parameters."""
        if not (0 <= self.ring_id <= 3):
            raise ValueError(f"Ring ID must be between 0 and 3, got {self.ring_id}")
        if len(self.positions) != self.size:
            raise ValueError(f"Ring size mismatch: expected {self.size}, got {len(self.positions)}")
    
    def get_empty_positions(self) -> List[Position]:
        """Get all empty positions in this ring."""
        return [pos for pos in self.positions if pos.is_empty()]
    
    def get_occupied_positions(self, player: Optional[Player] = None) -> List[Position]:
        """Get occupied positions in this ring, optionally filtered by player."""
        if player is None:
            return [pos for pos in self.positions if not pos.is_empty()]
        return [pos for pos in self.positions if pos.is_occupied_by(player)]
    
    def get_position_by_angle(self, angle: float) -> Optional[Position]:
        """Find position closest to the given angle."""
        if not self.positions:
            return None
        
        closest_pos = min(self.positions, key=lambda p: abs(p.angle - angle))
        return closest_pos


@dataclass
class ValidationResult:
    """Result of board validation."""
    is_valid: bool
    error_message: Optional[str] = None


@dataclass
class CorruptionReport:
    """Report of board corruption issues."""
    is_corrupted: bool
    issues: List[str]


class CircularBoard:
    """
    Represents the circular tic-tac-toe board with 32 positions arranged in 4 concentric rings.
    
    Ring structure:
    - Ring 0 (center): 8 positions (0-7)
    - Ring 1 (inner): 8 positions (8-15)
    - Ring 2 (middle): 8 positions (16-23)
    - Ring 3 (outer): 8 positions (24-31)
    """
    
    # Ring sizes for the 32-position board
    RING_SIZES = [8, 8, 8, 8]
    TOTAL_POSITIONS = 32
    
    def __init__(self):
        """Initialize the circular board."""
        self.positions: List[Position] = []
        self.rings: List[Ring] = []
        self.current_player: Player = Player.X
        self.move_history: List[Move] = []
        self._initialize_board()
    
    def _initialize_board(self):
        """Initialize the board with all positions and rings."""
        position_id = 0
        
        for ring_id, ring_size in enumerate(self.RING_SIZES):
            ring_positions = []
            
            # Calculate angular spacing for positions in this ring
            angle_step = 360.0 / ring_size if ring_size > 1 else 0.0
            
            for i in range(ring_size):
                angle = i * angle_step
                position = Position(
                    id=position_id,
                    ring=ring_id,
                    angle=angle,
                    occupant=None
                )
                ring_positions.append(position)
                self.positions.append(position)
                position_id += 1
            
            ring = Ring(ring_id=ring_id, positions=ring_positions, size=ring_size)
            self.rings.append(ring)
    
    def get_position(self, position_id: int) -> Optional[Position]:
        """Get position by ID."""
        if 0 <= position_id < len(self.positions):
            return self.positions[position_id]
        return None
    
    def get_ring(self, ring_id: int) -> Optional[Ring]:
        """Get ring by ID."""
        if 0 <= ring_id < len(self.rings):
            return self.rings[ring_id]
        return None
    
    def make_move(self, move: Move) -> bool:
        """
        Make a move on the board.
        
        Args:
            move: Move to make
            
        Returns:
            True if move was successful, False otherwise
        """
        position = self.get_position(move.position)
        if position is None or not position.is_empty():
            return False
        
        position.occupant = move.player
        self.move_history.append(move)
        self._switch_player()
        return True
    
    def undo_move(self) -> bool:
        """
        Undo the last move.
        
        Returns:
            True if undo was successful, False if no moves to undo
        """
        if not self.move_history:
            return False
        
        last_move = self.move_history.pop()
        position = self.get_position(last_move.position)
        if position:
            position.occupant = None
            self._switch_player()
            return True
        return False
    
    def _switch_player(self):
        """Switch the current player."""
        self.current_player = Player.O if self.current_player == Player.X else Player.X
    
    def get_empty_positions(self) -> List[Position]:
        """Get all empty positions on the board."""
        return [pos for pos in self.positions if pos.is_empty()]
    
    def get_occupied_positions(self, player: Optional[Player] = None) -> List[Position]:
        """Get occupied positions, optionally filtered by player."""
        if player is None:
            return [pos for pos in self.positions if not pos.is_empty()]
        return [pos for pos in self.positions if pos.is_occupied_by(player)]
    
    def is_full(self) -> bool:
        """Check if the board is full."""
        return len(self.get_empty_positions()) == 0
    
    def reset(self):
        """Reset the board to initial state."""
        for position in self.positions:
            position.occupant = None
        self.move_history.clear()
        self.current_player = Player.X
    
    def copy(self) -> 'CircularBoard':
        """Create a deep copy of the board."""
        new_board = CircularBoard()
        
        # Copy all moves to recreate the board state
        for move in self.move_history:
            new_board.make_move(move)
        
        return new_board
    
    def get_board_state_hash(self) -> str:
        """Get a hash string representing the current board state."""
        state_chars = []
        for position in self.positions:
            if position.occupant is None:
                state_chars.append('_')
            else:
                state_chars.append(position.occupant.value)
        return ''.join(state_chars)
    
    def validate_move(self, move: Move) -> ValidationResult:
        """
        Validate if a move is legal.
        
        Args:
            move: Move to validate
            
        Returns:
            ValidationResult indicating if move is valid
        """
        # Check if position exists
        if not (0 <= move.position < self.TOTAL_POSITIONS):
            return ValidationResult(False, f"Invalid position: {move.position}")
        
        # Check if position is empty
        position = self.get_position(move.position)
        if position is None:
            return ValidationResult(False, f"Position {move.position} does not exist")
        
        if not position.is_empty():
            return ValidationResult(False, f"Position {move.position} is already occupied")
        
        # Check if it's the correct player's turn
        if move.player != self.current_player:
            return ValidationResult(False, f"It's {self.current_player.value}'s turn, not {move.player.value}'s")
        
        return ValidationResult(True)
    
    def validate_board_state(self) -> bool:
        """
        Validate the current board state for consistency.
        
        Returns:
            True if board state is valid, False otherwise
        """
        try:
            # Check total number of positions
            if len(self.positions) != self.TOTAL_POSITIONS:
                return False
            
            # Check ring structure
            if len(self.rings) != len(self.RING_SIZES):
                return False
            
            # Validate each ring
            for i, ring in enumerate(self.rings):
                if ring.ring_id != i:
                    return False
                if ring.size != self.RING_SIZES[i]:
                    return False
                if len(ring.positions) != ring.size:
                    return False
            
            # Check position IDs are sequential and unique
            position_ids = [pos.id for pos in self.positions]
            expected_ids = list(range(self.TOTAL_POSITIONS))
            if sorted(position_ids) != expected_ids:
                return False
            
            # Validate move history consistency
            test_board = CircularBoard()
            for move in self.move_history:
                if not test_board.validate_move(move).is_valid:
                    return False
                test_board.make_move(move)
            
            # Check if reconstructed board matches current state
            if test_board.get_board_state_hash() != self.get_board_state_hash():
                return False
            
            return True
            
        except Exception:
            return False
    
    def detect_corruption(self) -> CorruptionReport:
        """
        Detect and report any corruption in the board state.
        
        Returns:
            CorruptionReport with details of any issues found
        """
        issues = []
        
        try:
            # Check position count
            if len(self.positions) != self.TOTAL_POSITIONS:
                issues.append(f"Expected {self.TOTAL_POSITIONS} positions, found {len(self.positions)}")
            
            # Check ring count and structure
            if len(self.rings) != len(self.RING_SIZES):
                issues.append(f"Expected {len(self.RING_SIZES)} rings, found {len(self.rings)}")
            
            # Validate ring structure
            for i, ring in enumerate(self.rings):
                if i < len(self.RING_SIZES):
                    expected_size = self.RING_SIZES[i]
                    if ring.size != expected_size:
                        issues.append(f"Ring {i} has size {ring.size}, expected {expected_size}")
                    if len(ring.positions) != ring.size:
                        issues.append(f"Ring {i} position count mismatch: {len(ring.positions)} vs {ring.size}")
            
            # Check for duplicate position IDs
            position_ids = [pos.id for pos in self.positions]
            if len(set(position_ids)) != len(position_ids):
                issues.append("Duplicate position IDs found")
            
            # Check position ID range
            for pos in self.positions:
                if not (0 <= pos.id < self.TOTAL_POSITIONS):
                    issues.append(f"Position ID {pos.id} out of range")
            
            # Check ring assignments
            for pos in self.positions:
                if not (0 <= pos.ring <= 3):
                    issues.append(f"Position {pos.id} has invalid ring {pos.ring}")
            
            # Check angle ranges
            for pos in self.positions:
                if not (0.0 <= pos.angle < 360.0):
                    issues.append(f"Position {pos.id} has invalid angle {pos.angle}")
            
            # Validate move history
            for i, move in enumerate(self.move_history):
                if not (0 <= move.position < self.TOTAL_POSITIONS):
                    issues.append(f"Move {i} has invalid position {move.position}")
                if not isinstance(move.player, Player):
                    issues.append(f"Move {i} has invalid player type")
            
        except Exception as e:
            issues.append(f"Exception during corruption detection: {str(e)}")
        
        return CorruptionReport(is_corrupted=len(issues) > 0, issues=issues)
    
    def __str__(self) -> str:
        """String representation of the board."""
        lines = [f"Circular Board - Current Player: {self.current_player.value}"]
        lines.append(f"Moves played: {len(self.move_history)}")
        lines.append("")
        
        for ring in self.rings:
            ring_name = ["Center", "Inner", "Middle", "Outer"][ring.ring_id]
            lines.append(f"{ring_name} Ring (Ring {ring.ring_id}):")
            
            positions_str = []
            for pos in ring.positions:
                occupant = pos.occupant.value if pos.occupant else '_'
                positions_str.append(f"{pos.id}:{occupant}")
            
            lines.append("  " + " ".join(positions_str))
            lines.append("")
        
        return "\n".join(lines)