"""
Win detection system for circular tic-tac-toe.
"""
from typing import List, Optional, Dict, Set
import math
from ...models.enums import Player, WinPatternType
from ...models.win_pattern import WinPattern, Threat, WinResult
from ...game.board import CircularBoard


class WinDetector:
    """
    Detects winning patterns and threats in circular tic-tac-toe.
    
    Handles three types of winning patterns:
    - Radial: 4 positions in a straight line from center outward
    - Circular: 4 consecutive positions in the same ring
    - Spiral: 4 positions following spiral patterns
    """
    
    def __init__(self):
        """Initialize the win detector with all possible winning patterns."""
        self._winning_patterns: List[WinPattern] = []
        self._patterns_by_type: Dict[WinPatternType, List[WinPattern]] = {
            WinPatternType.RADIAL: [],
            WinPatternType.CIRCULAR: [],
            WinPatternType.SPIRAL: []
        }
        self._patterns_by_position: Dict[int, List[WinPattern]] = {}
        self._generate_all_patterns()
    
    def _generate_all_patterns(self):
        """Generate all possible winning patterns for the 32-position board."""
        self._generate_radial_patterns()
        self._generate_circular_patterns()
        self._generate_spiral_patterns()
        
        # Build lookup tables
        self._build_position_lookup()
    
    def _generate_radial_patterns(self):
        """Generate radial winning patterns (4 in a row from center outward)."""
        # Ring structure: [8, 8, 8, 8] positions
        # Ring 0: positions 0-7
        # Ring 1: positions 8-15
        # Ring 2: positions 16-23
        # Ring 3: positions 24-31
        
        ring_starts = [0, 8, 16, 24]
        ring_sizes = [8, 8, 8, 8]
        
        # For each position in the center ring
        for center_pos in range(8):  # positions 0-7
            center_angle = center_pos * 45.0  # 0°, 45°, 90°, 135°, 180°, 225°, 270°, 315°
            
            # Find aligned positions in outer rings
            radial_line = [center_pos]
            
            # Ring 1 (inner): find closest position to center_angle
            ring1_angle_step = 360.0 / 8
            ring1_index = round(center_angle / ring1_angle_step) % 8
            radial_line.append(ring_starts[1] + ring1_index)
            
            # Ring 2 (middle): find closest position to center_angle
            ring2_angle_step = 360.0 / 8
            ring2_index = round(center_angle / ring2_angle_step) % 8
            radial_line.append(ring_starts[2] + ring2_index)
            
            # Ring 3 (outer): find closest position to center_angle
            ring3_angle_step = 360.0 / 8
            ring3_index = round(center_angle / ring3_angle_step) % 8
            radial_line.append(ring_starts[3] + ring3_index)
            
            pattern = WinPattern(
                type=WinPatternType.RADIAL,
                positions=radial_line,
                priority=100,  # High priority for radial patterns
                description=f"Radial line from center position {center_pos}"
            )
            self._winning_patterns.append(pattern)
            self._patterns_by_type[WinPatternType.RADIAL].append(pattern)
    
    def _generate_circular_patterns(self):
        """Generate circular winning patterns (4 consecutive positions in same ring)."""
        ring_starts = [0, 8, 16, 24]
        ring_sizes = [8, 8, 8, 8]
        
        # All rings have 8 positions, so all can have circular patterns
        for ring_id in range(4):  # rings 0, 1, 2, 3
            ring_start = ring_starts[ring_id]
            ring_size = ring_sizes[ring_id]
            
            # Generate all possible 4-consecutive patterns in this ring
            for start_pos in range(ring_size):
                positions = []
                for i in range(4):
                    pos_index = (start_pos + i) % ring_size
                    positions.append(ring_start + pos_index)
                
                pattern = WinPattern(
                    type=WinPatternType.CIRCULAR,
                    positions=positions,
                    priority=80,  # Medium-high priority
                    description=f"Circular arc in ring {ring_id} starting at position {start_pos}"
                )
                self._winning_patterns.append(pattern)
                self._patterns_by_type[WinPatternType.CIRCULAR].append(pattern)
    
    def _generate_spiral_patterns(self):
        """Generate spiral winning patterns (4 positions following spiral paths)."""
        ring_starts = [0, 8, 16, 24]
        ring_sizes = [8, 8, 8, 8]
        
        # Generate clockwise and counterclockwise spirals
        for direction in [1, -1]:  # 1 = clockwise, -1 = counterclockwise
            # Start from each position in ring 0 (center)
            for center_pos in range(8):
                center_angle = center_pos * 45.0
                
                # Create spiral patterns moving outward
                for angle_increment in [45, 90, 135]:  # Different spiral rates (multiples of 45°)
                    positions = [center_pos]
                    current_angle = center_angle
                    
                    # Add positions from rings 1, 2, 3
                    for ring_id in range(1, 4):
                        current_angle += direction * angle_increment
                        current_angle = current_angle % 360.0
                        
                        ring_start = ring_starts[ring_id]
                        ring_size = ring_sizes[ring_id]
                        angle_step = 360.0 / ring_size  # 45° for all rings
                        
                        # Find closest position in this ring
                        ring_index = round(current_angle / angle_step) % ring_size
                        positions.append(ring_start + ring_index)
                    
                    # Only add if we have 4 unique positions
                    if len(set(positions)) == 4:
                        direction_name = "clockwise" if direction == 1 else "counterclockwise"
                        pattern = WinPattern(
                            type=WinPatternType.SPIRAL,
                            positions=positions,
                            priority=60,  # Medium priority
                            description=f"Spiral {direction_name} from center {center_pos}, increment {angle_increment}°"
                        )
                        self._winning_patterns.append(pattern)
                        self._patterns_by_type[WinPatternType.SPIRAL].append(pattern)
    

    
    def _build_position_lookup(self):
        """Build lookup table for patterns by position."""
        for i in range(32):
            self._patterns_by_position[i] = []
        
        for pattern in self._winning_patterns:
            for pos_id in pattern.positions:
                self._patterns_by_position[pos_id].append(pattern)
    
    def check_win(self, board: CircularBoard) -> Optional[WinResult]:
        """
        Check if there's a winning condition on the board.
        
        Args:
            board: Current board state
            
        Returns:
            WinResult if there's a winner, None otherwise
        """
        for pattern in self._winning_patterns:
            # Check if all positions in pattern are occupied by the same player
            occupied_by = None
            all_occupied = True
            
            for pos_id in pattern.positions:
                position = board.get_position(pos_id)
                if position is None or position.is_empty():
                    all_occupied = False
                    break
                
                if occupied_by is None:
                    occupied_by = position.occupant
                elif position.occupant != occupied_by:
                    all_occupied = False
                    break
            
            if all_occupied and occupied_by is not None:
                return WinResult(
                    winner=occupied_by,
                    winning_pattern=pattern,
                    winning_positions=pattern.positions.copy()
                )
        
        return None
    
    def find_threats(self, board: CircularBoard, player: Player) -> List[Threat]:
        """
        Find all threats for a specific player.
        
        Args:
            board: Current board state
            player: Player to find threats for
            
        Returns:
            List of threats sorted by severity (highest first)
        """
        threats = []
        opponent = Player.O if player == Player.X else Player.X
        
        for pattern in self._winning_patterns:
            player_count = pattern.count_occupied_by_player(board.positions, player)
            opponent_count = pattern.count_occupied_by_player(board.positions, opponent)
            
            # Skip if opponent has any pieces in this pattern (blocked)
            if opponent_count > 0:
                continue
            
            # Only consider patterns where player has at least 1 piece
            if player_count == 0:
                continue
            
            # Find empty positions in this pattern
            empty_positions = []
            for pos_id in pattern.positions:
                position = board.get_position(pos_id)
                if position and position.is_empty():
                    empty_positions.append(pos_id)
            
            if empty_positions:
                # Calculate severity based on how close to winning
                severity = player_count  # 1-3 (4 would be a win)
                
                threat = Threat(
                    pattern=pattern,
                    player=player,
                    completing_moves=empty_positions,
                    severity=severity,
                    blocked=False
                )
                threats.append(threat)
        
        # Sort by severity (highest first), then by pattern priority
        threats.sort(key=lambda t: (t.severity, t.pattern.priority), reverse=True)
        return threats
    
    def find_immediate_wins(self, board: CircularBoard, player: Player) -> List[int]:
        """
        Find positions that would immediately win the game for the player.
        
        Args:
            board: Current board state
            player: Player to find winning moves for
            
        Returns:
            List of position IDs that would result in immediate wins
        """
        winning_moves = []
        opponent = Player.O if player == Player.X else Player.X
        
        for pattern in self._winning_patterns:
            player_count = pattern.count_occupied_by_player(board.positions, player)
            opponent_count = pattern.count_occupied_by_player(board.positions, opponent)
            
            # Check if player has 3 pieces and opponent has 0 in this pattern
            if player_count == 3 and opponent_count == 0:
                # Find the empty position
                for pos_id in pattern.positions:
                    position = board.get_position(pos_id)
                    if position and position.is_empty():
                        winning_moves.append(pos_id)
                        break
        
        return list(set(winning_moves))  # Remove duplicates
    
    def find_blocking_moves(self, board: CircularBoard, player: Player) -> List[int]:
        """
        Find positions that would block opponent's immediate wins.
        
        Args:
            board: Current board state
            player: Player who needs to block
            
        Returns:
            List of position IDs that would block opponent wins
        """
        opponent = Player.O if player == Player.X else Player.X
        return self.find_immediate_wins(board, opponent)
    
    def get_patterns_involving_position(self, position_id: int) -> List[WinPattern]:
        """
        Get all winning patterns that involve a specific position.
        
        Args:
            position_id: Position ID to check
            
        Returns:
            List of patterns involving this position
        """
        return self._patterns_by_position.get(position_id, [])
    
    def get_all_winning_patterns(self) -> List[WinPattern]:
        """Get all possible winning patterns."""
        return self._winning_patterns.copy()
    
    def get_patterns_by_type(self, pattern_type: WinPatternType) -> List[WinPattern]:
        """Get all patterns of a specific type."""
        return self._patterns_by_type.get(pattern_type, []).copy()
    
    def count_patterns_by_type(self) -> Dict[WinPatternType, int]:
        """Get count of patterns by type."""
        return {
            pattern_type: len(patterns)
            for pattern_type, patterns in self._patterns_by_type.items()
        }
    
    def evaluate_position_strategic_value(self, position_id: int) -> int:
        """
        Evaluate the strategic value of a position based on patterns it participates in.
        
        Args:
            position_id: Position to evaluate
            
        Returns:
            Strategic value score
        """
        patterns = self.get_patterns_involving_position(position_id)
        return sum(pattern.priority for pattern in patterns)