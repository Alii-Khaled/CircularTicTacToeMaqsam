"""
Position evaluation system for circular tic-tac-toe AI.

This module implements strategic heuristics to evaluate board positions,
considering center control, pattern recognition, ring dominance, and threat assessment.
"""
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from ...models.enums import Player, WinPatternType
from ...models.position import Position
from ...game.board import CircularBoard
from .win_detector import WinDetector


@dataclass
class EvaluationMetrics:
    """
    Detailed metrics for position evaluation.
    
    Attributes:
        center_control: Control of center positions (30% weight)
        pattern_potential: Potential for creating winning patterns (40% weight)
        ring_dominance: Control within each ring (20% weight)
        threat_level: Immediate threats faced (10% weight)
        total_score: Final weighted evaluation score
    """
    center_control: float
    pattern_potential: float
    ring_dominance: float
    threat_level: float
    total_score: float


class PositionEvaluator:
    """
    Evaluates board positions using strategic heuristics tailored for circular tic-tac-toe.
    
    The evaluation considers:
    - Center control (30%): Prioritizes center and inner ring positions
    - Pattern recognition (40%): Evaluates potential for winning patterns
    - Ring dominance (20%): Measures control within each ring
    - Threat assessment (10%): Evaluates immediate threats and opportunities
    """
    
    # Evaluation weights
    CENTER_CONTROL_WEIGHT = 0.20
    PATTERN_POTENTIAL_WEIGHT = 0.20
    RING_DOMINANCE_WEIGHT = 0.10
    THREAT_ASSESSMENT_WEIGHT = 0.50  # Increased to make threats more important
    
    # Position values by ring
    RING_VALUES = {
        0: 50,  # Center ring - highest value
        1: 30,  # Inner ring
        2: 20,  # Middle ring
        3: 10   # Outer ring
    }
    
    # Pattern type priorities
    PATTERN_PRIORITIES = {
        WinPatternType.RADIAL: 100,
        WinPatternType.CIRCULAR: 80,
        WinPatternType.SPIRAL: 60
    }
    
    def __init__(self):
        """Initialize the position evaluator with win detector."""
        self.win_detector = WinDetector()
    
    def evaluate_position(self, board: CircularBoard, player: Player) -> float:
        """
        Evaluate the current board position for a specific player.
        
        Args:
            board: Current board state
            player: Player to evaluate position for
            
        Returns:
            Evaluation score (positive = good for player, negative = bad)
        """
        metrics = self.get_detailed_evaluation(board, player)
        return metrics.total_score
    
    def get_detailed_evaluation(self, board: CircularBoard, player: Player) -> EvaluationMetrics:
        """
        Get detailed evaluation metrics for a board position.
        
        Args:
            board: Current board state
            player: Player to evaluate position for
            
        Returns:
            EvaluationMetrics with detailed breakdown
        """
        center_control = self._evaluate_center_control(board, player)
        pattern_potential = self._evaluate_pattern_potential(board, player)
        ring_dominance = self._evaluate_ring_dominance(board, player)
        threat_level = self._evaluate_threat_assessment(board, player)
        
        # Calculate weighted total score
        total_score = (
            center_control * self.CENTER_CONTROL_WEIGHT +
            pattern_potential * self.PATTERN_POTENTIAL_WEIGHT +
            ring_dominance * self.RING_DOMINANCE_WEIGHT +
            threat_level * self.THREAT_ASSESSMENT_WEIGHT
        )
        
        return EvaluationMetrics(
            center_control=center_control,
            pattern_potential=pattern_potential,
            ring_dominance=ring_dominance,
            threat_level=threat_level,
            total_score=total_score
        )
    
    def _evaluate_center_control(self, board: CircularBoard, player: Player) -> float:
        """
        Evaluate center control (30% weight).
        
        Prioritizes control of center positions and inner rings.
        
        Args:
            board: Current board state
            player: Player to evaluate for
            
        Returns:
            Center control score
        """
        opponent = Player.O if player == Player.X else Player.X
        score = 0.0
        
        # Evaluate each position based on its ring value
        for position in board.positions:
            ring_value = self.RING_VALUES[position.ring]
            
            if position.is_occupied_by(player):
                score += ring_value
            elif position.is_occupied_by(opponent):
                score -= ring_value
            # Empty positions contribute nothing to current control
        
        return score
    
    def _evaluate_pattern_potential(self, board: CircularBoard, player: Player) -> float:
        """
        Evaluate pattern recognition potential (40% weight).
        
        Considers potential for creating winning patterns and blocking opponent patterns.
        
        Args:
            board: Current board state
            player: Player to evaluate for
            
        Returns:
            Pattern potential score
        """
        opponent = Player.O if player == Player.X else Player.X
        score = 0.0
        
        all_patterns = self.win_detector.get_all_winning_patterns()
        
        for pattern in all_patterns:
            player_count = pattern.count_occupied_by_player(board.positions, player)
            opponent_count = pattern.count_occupied_by_player(board.positions, opponent)
            
            # Pattern is blocked if opponent has any pieces in it
            if opponent_count > 0 and player_count > 0:
                continue  # Mixed pattern, no value
            
            pattern_priority = self.PATTERN_PRIORITIES.get(pattern.type, 50)
            
            if player_count > 0 and opponent_count == 0:
                # Player has pieces in this pattern, opponent doesn't
                if player_count == 1:
                    score += 25 * (pattern_priority / 100)  # Potential pattern
                elif player_count == 2:
                    score += 100 * (pattern_priority / 100)  # Two in a row
                elif player_count == 3:
                    score += 500 * (pattern_priority / 100)  # Three in a row (major threat)
            
            elif opponent_count > 0 and player_count == 0:
                # Opponent has pieces in this pattern, player doesn't
                if opponent_count == 1:
                    score -= 25 * (pattern_priority / 100)  # Opponent potential
                elif opponent_count == 2:
                    score -= 100 * (pattern_priority / 100)  # Opponent two in a row
                elif opponent_count == 3:
                    score -= 500 * (pattern_priority / 100)  # Opponent three in a row (major threat)
        
        return score
    
    def _evaluate_ring_dominance(self, board: CircularBoard, player: Player) -> float:
        """
        Evaluate ring dominance (20% weight).
        
        Measures control within each ring and balanced presence across rings.
        
        Args:
            board: Current board state
            player: Player to evaluate for
            
        Returns:
            Ring dominance score
        """
        opponent = Player.O if player == Player.X else Player.X
        score = 0.0
        
        ring_control_scores = []
        
        for ring in board.rings:
            player_positions = len(ring.get_occupied_positions(player))
            opponent_positions = len(ring.get_occupied_positions(opponent))
            total_positions = ring.size
            
            # Calculate control percentage for this ring
            if total_positions > 0:
                player_control = player_positions / total_positions
                opponent_control = opponent_positions / total_positions
                
                ring_control = player_control - opponent_control
                ring_control_scores.append(ring_control)
                
                # Bonus for controlling majority of positions in a ring
                if player_control > 0.5:
                    score += 75 * player_control
                elif opponent_control > 0.5:
                    score -= 75 * opponent_control
                
                # Add base score for any control in ring
                score += ring_control * 25  # Base ring control score
        
        # Bonus for balanced presence across rings
        if len(ring_control_scores) > 0:
            # Reward having positive control in multiple rings
            positive_rings = sum(1 for control in ring_control_scores if control > 0)
            if positive_rings >= 2:
                score += 25
        
        return score
    
    def _evaluate_threat_assessment(self, board: CircularBoard, player: Player) -> float:
        """
        Evaluate threat assessment (10% weight).
        
        Considers immediate threats and opportunities.
        
        Args:
            board: Current board state
            player: Player to evaluate for
            
        Returns:
            Threat assessment score
        """
        opponent = Player.O if player == Player.X else Player.X
        score = 0.0
        
        # Check if game is already won
        win_result = self.win_detector.check_win(board)
        if win_result:
            if win_result.winner == player:
                score += 10000  # Massive bonus for winning
            else:
                score -= 10000  # Massive penalty for losing
            return score
        
        # Find immediate winning opportunities
        player_wins = self.win_detector.find_immediate_wins(board, player)
        opponent_wins = self.win_detector.find_immediate_wins(board, opponent)
        
        # Immediate wins are extremely valuable
        score += len(player_wins) * 1000
        score -= len(opponent_wins) * 5000  # Much higher penalty for allowing opponent wins
        
        # Find threats for both players
        player_threats = self.win_detector.find_threats(board, player)
        opponent_threats = self.win_detector.find_threats(board, opponent)
        
        # Evaluate player threats
        for threat in player_threats:
            if threat.severity == 3:  # One move away from winning
                score += 200
            elif threat.severity == 2:  # Two moves away
                score += 50
            elif threat.severity == 1:  # Three moves away
                score += 10
        
        # Evaluate opponent threats (negative for player)
        for threat in opponent_threats:
            if threat.severity == 3:  # Opponent one move away from winning
                score -= 200
            elif threat.severity == 2:  # Opponent two moves away
                score -= 50
            elif threat.severity == 1:  # Opponent three moves away
                score -= 10
        
        # Bonus for creating multiple threats simultaneously
        high_severity_player_threats = [t for t in player_threats if t.severity >= 2]
        if len(high_severity_player_threats) >= 2:
            score += 100  # Multiple threat bonus
        
        return score
    
    def evaluate_move(self, board: CircularBoard, move_position: int, player: Player) -> float:
        """
        Evaluate the value of making a move at a specific position.
        
        Args:
            board: Current board state
            move_position: Position ID to evaluate
            player: Player making the move
            
        Returns:
            Move evaluation score
        """
        # Check if position is valid first
        if not (0 <= move_position <= 31):
            return float('-inf')
        
        # Check if position is empty
        position = board.get_position(move_position)
        if position is None or not position.is_empty():
            return float('-inf')
        
        # Create a copy of the board and make the move
        test_board = board.copy()
        from ...models.move import Move
        
        move = Move(position=move_position, player=player)
        if not test_board.make_move(move):
            return float('-inf')  # Invalid move
        
        # Evaluate the resulting position
        return self.evaluate_position(test_board, player)
    
    def get_best_moves(self, board: CircularBoard, player: Player, count: int = 5) -> List[Tuple[int, float]]:
        """
        Get the best available moves for a player.
        
        Args:
            board: Current board state
            player: Player to find moves for
            count: Number of best moves to return
            
        Returns:
            List of (position_id, score) tuples sorted by score (best first)
        """
        empty_positions = [pos.id for pos in board.get_empty_positions()]
        move_scores = []
        
        for position_id in empty_positions:
            score = self.evaluate_move(board, position_id, player)
            move_scores.append((position_id, score))
        
        # Sort by score (highest first)
        move_scores.sort(key=lambda x: x[1], reverse=True)
        
        return move_scores[:count]
    
    def compare_positions(self, board1: CircularBoard, board2: CircularBoard, player: Player) -> float:
        """
        Compare two board positions for a player.
        
        Args:
            board1: First board position
            board2: Second board position
            player: Player to evaluate for
            
        Returns:
            Positive if board1 is better, negative if board2 is better, 0 if equal
        """
        score1 = self.evaluate_position(board1, player)
        score2 = self.evaluate_position(board2, player)
        return score1 - score2