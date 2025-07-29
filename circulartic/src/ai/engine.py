"""
AI Engine controller for circular tic-tac-toe.

This module implements the main AI controller that integrates minimax algorithm
with position evaluation, enforces time limits, implements fallback strategies,
and provides performance monitoring and logging.
"""
import time
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

from ..models.enums import Player
from ..models.move import Move
from ..game.board import CircularBoard
from .search.minimax import SearchAlgorithm, SearchResult
from .evaluation.position_evaluator import PositionEvaluator
from .evaluation.win_detector import WinDetector


class AIDecisionError(Exception):
    """Exception raised when AI fails to make a decision."""
    
    def __init__(self, message: str, board: CircularBoard, search_depth: int):
        super().__init__(message)
        self.board = board
        self.search_depth = search_depth


class FallbackStrategy(Enum):
    """Available fallback strategies when AI encounters errors."""
    REDUCE_DEPTH = "reduce_depth"
    SIMPLE_EVALUATION = "simple_evaluation"
    RANDOM_MOVE = "random_move"


@dataclass
class AIPerformanceMetrics:
    """
    Performance metrics for AI decision making.
    
    Attributes:
        move_time: Time taken to select the move (seconds)
        search_depth: Depth reached in the search
        nodes_evaluated: Number of nodes evaluated
        transposition_hits: Number of transposition table hits
        cutoffs: Number of alpha-beta cutoffs
        fallback_used: Whether fallback strategy was used
        fallback_strategy: Which fallback strategy was used (if any)
        evaluation_score: Final evaluation score of the selected move
    """
    move_time: float
    search_depth: int
    nodes_evaluated: int
    transposition_hits: int
    cutoffs: int
    fallback_used: bool
    fallback_strategy: Optional[FallbackStrategy]
    evaluation_score: float


@dataclass
class AIDecision:
    """
    Complete AI decision with move and performance data.
    
    Attributes:
        move: The selected move
        metrics: Performance metrics for this decision
        confidence: Confidence level in the decision (0.0 to 1.0)
        reasoning: Human-readable explanation of the decision
    """
    move: Move
    metrics: AIPerformanceMetrics
    confidence: float
    reasoning: str


class AIEngine:
    """
    Main AI controller that orchestrates the decision-making process.
    
    Features:
    - Integrates minimax algorithm with position evaluation
    - Enforces time limits (3 seconds maximum)
    - Implements fallback strategies for error handling
    - Provides performance monitoring and logging
    - Supports different difficulty levels
    """
    
    # Time limits
    DEFAULT_TIME_LIMIT = 5.0  # 3 seconds maximum per move
    FALLBACK_TIME_LIMIT = 3.0  # Reduced time for fallback strategies
    
    # Search depth limits
    MAX_SEARCH_DEPTH = 12
    MIN_SEARCH_DEPTH = 3
    FALLBACK_SEARCH_DEPTH = 5
    
    # Performance thresholds
    MIN_CONFIDENCE_THRESHOLD = 0.3
    GOOD_MOVE_TIME_THRESHOLD = 1.0
    
    def __init__(self, time_limit: float = DEFAULT_TIME_LIMIT, 
                 max_depth: int = MAX_SEARCH_DEPTH,
                 enable_logging: bool = True):
        """
        Initialize the AI engine.
        
        Args:
            time_limit: Maximum time per move in seconds
            max_depth: Maximum search depth
            enable_logging: Whether to enable performance logging
        """
        self.time_limit = time_limit
        self.max_depth = max_depth
        self.enable_logging = enable_logging
        
        # Initialize components
        self.search_algorithm = SearchAlgorithm()
        self.position_evaluator = PositionEvaluator()
        self.win_detector = WinDetector()
        
        # Performance tracking
        self.decision_history: List[AIDecision] = []
        self.total_decisions = 0
        self.total_time = 0.0
        self.fallback_count = 0
        
        # Configure logging
        if self.enable_logging:
            self._setup_logging()
    
    def _setup_logging(self):
        """Set up logging for AI performance monitoring."""
        self.logger = logging.getLogger('circular_ttt_ai')
        self.logger.setLevel(logging.INFO)
        
        # Create console handler if none exists
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def select_move(self, board: CircularBoard, player: Player) -> AIDecision:
        """
        Select the best move for the given player and board state.
        
        Args:
            board: Current board state
            player: Player to select move for
            
        Returns:
            AIDecision with the selected move and performance metrics
            
        Raises:
            AIDecisionError: If all fallback strategies fail
        """
        start_time = time.time()
        
        # Validate inputs
        if board is None:
            raise AIDecisionError("Board cannot be None", board, 0)
        
        if not isinstance(player, Player):
            raise AIDecisionError(f"Invalid player type: {type(player)}", board, 0)
        
        # Check if game is already over
        win_result = self.win_detector.check_win(board)
        if win_result:
            raise AIDecisionError(f"Game is already over, winner: {win_result.winner}", board, 0)
        
        # Check if there are any legal moves
        empty_positions = board.get_empty_positions()
        if not empty_positions:
            raise AIDecisionError("No legal moves available", board, 0)
        
        # Try primary strategy first
        try:
            decision = self._select_move_primary(board, player, start_time)
            self._log_decision(decision, "PRIMARY")
            return decision
            
        except Exception as e:
            if self.enable_logging:
                self.logger.warning(f"Primary strategy failed: {str(e)}")
            
            # Try fallback strategies
            for strategy in [FallbackStrategy.REDUCE_DEPTH, 
                           FallbackStrategy.SIMPLE_EVALUATION, 
                           FallbackStrategy.RANDOM_MOVE]:
                try:
                    decision = self._select_move_fallback(board, player, strategy, start_time)
                    self.fallback_count += 1
                    self._log_decision(decision, f"FALLBACK_{strategy.value.upper()}")
                    return decision
                    
                except Exception as fallback_error:
                    if self.enable_logging:
                        self.logger.warning(f"Fallback {strategy.value} failed: {str(fallback_error)}")
                    continue
            
            # All strategies failed
            raise AIDecisionError(
                f"All strategies failed. Last error: {str(e)}", 
                board, 
                self.max_depth
            )
    
    def _select_move_primary(self, board: CircularBoard, player: Player, start_time: float) -> AIDecision:
        """
        Primary move selection using full minimax search.
        
        Args:
            board: Current board state
            player: Player to select move for
            start_time: Time when decision process started
            
        Returns:
            AIDecision with selected move
        """
        # Set the search algorithm's time limit
        self.search_algorithm.set_time_limit(self.time_limit)
        
        # Create a copy of the board with the correct current player
        board_copy = board.copy()
        board_copy.current_player = player
        
        # Perform the search
        search_result = self.search_algorithm.search(
            board_copy, 
            max_depth=self.max_depth,
            time_limit=self.time_limit
        )
        
        if search_result.best_move is None:
            raise AIDecisionError("Search returned no move", board, search_result.depth)
        
        # Calculate confidence based on search quality
        confidence = self._calculate_confidence(search_result, board)
        
        # Generate reasoning
        reasoning = self._generate_reasoning(search_result, board, player)
        
        # Create performance metrics
        move_time = time.time() - start_time
        metrics = AIPerformanceMetrics(
            move_time=move_time,
            search_depth=search_result.depth,
            nodes_evaluated=search_result.nodes_evaluated,
            transposition_hits=self.search_algorithm.transposition_hits,
            cutoffs=self.search_algorithm.cutoffs,
            fallback_used=False,
            fallback_strategy=None,
            evaluation_score=search_result.score
        )
        
        # Update statistics
        self.total_decisions += 1
        self.total_time += move_time
        
        decision = AIDecision(
            move=search_result.best_move,
            metrics=metrics,
            confidence=confidence,
            reasoning=reasoning
        )
        
        self.decision_history.append(decision)
        return decision
    
    def _select_move_fallback(self, board: CircularBoard, player: Player, 
                            strategy: FallbackStrategy, start_time: float) -> AIDecision:
        """
        Fallback move selection using simplified strategies.
        
        Args:
            board: Current board state
            player: Player to select move for
            strategy: Fallback strategy to use
            start_time: Time when decision process started
            
        Returns:
            AIDecision with selected move
        """
        move_time = time.time() - start_time
        
        if strategy == FallbackStrategy.REDUCE_DEPTH:
            return self._fallback_reduce_depth(board, player, start_time)
        
        elif strategy == FallbackStrategy.SIMPLE_EVALUATION:
            return self._fallback_simple_evaluation(board, player, start_time)
        
        elif strategy == FallbackStrategy.RANDOM_MOVE:
            return self._fallback_random_move(board, player, start_time)
        
        else:
            raise AIDecisionError(f"Unknown fallback strategy: {strategy}", board, 0)
    
    def _fallback_reduce_depth(self, board: CircularBoard, player: Player, start_time: float) -> AIDecision:
        """Fallback strategy: Reduce search depth and try again."""
        reduced_depth = max(self.MIN_SEARCH_DEPTH, self.FALLBACK_SEARCH_DEPTH)
        reduced_time = self.FALLBACK_TIME_LIMIT
        
        self.search_algorithm.set_time_limit(reduced_time)
        
        board_copy = board.copy()
        board_copy.current_player = player
        
        search_result = self.search_algorithm.search(
            board_copy,
            max_depth=reduced_depth,
            time_limit=reduced_time
        )
        
        if search_result.best_move is None:
            raise AIDecisionError("Reduced depth search failed", board, reduced_depth)
        
        move_time = time.time() - start_time
        confidence = max(0.3, self._calculate_confidence(search_result, board) * 0.7)
        
        metrics = AIPerformanceMetrics(
            move_time=move_time,
            search_depth=search_result.depth,
            nodes_evaluated=search_result.nodes_evaluated,
            transposition_hits=self.search_algorithm.transposition_hits,
            cutoffs=self.search_algorithm.cutoffs,
            fallback_used=True,
            fallback_strategy=FallbackStrategy.REDUCE_DEPTH,
            evaluation_score=search_result.score
        )
        
        reasoning = f"Used reduced depth search (depth {reduced_depth}) due to time constraints"
        
        decision = AIDecision(
            move=search_result.best_move,
            metrics=metrics,
            confidence=confidence,
            reasoning=reasoning
        )
        
        self.decision_history.append(decision)
        return decision
    
    def _fallback_simple_evaluation(self, board: CircularBoard, player: Player, start_time: float) -> AIDecision:
        """Fallback strategy: Use simple position evaluation without deep search."""
        empty_positions = board.get_empty_positions()
        
        if not empty_positions:
            raise AIDecisionError("No empty positions for simple evaluation", board, 0)
        
        # Check for immediate wins first
        immediate_wins = self.win_detector.find_immediate_wins(board, player)
        if immediate_wins:
            best_position = immediate_wins[0]
            evaluation_score = 1000.0
            reasoning = "Immediate winning move found"
        else:
            # Check for blocks
            opponent = Player.O if player == Player.X else Player.X
            opponent_wins = self.win_detector.find_immediate_wins(board, opponent)
            
            # Find positions that block opponent's wins
            blocking_moves = []
            if opponent_wins:
                for empty_pos in empty_positions:
                    test_board = board.copy()
                    test_move = Move(position=empty_pos.id, player=player)
                    if test_board.make_move(test_move):
                        opponent_wins_after = self.win_detector.find_immediate_wins(test_board, opponent)
                        if len(opponent_wins_after) < len(opponent_wins):
                            blocking_moves.append(empty_pos.id)
            
            if blocking_moves:
                best_position = blocking_moves[0]
                evaluation_score = 500.0
                reasoning = "Blocking opponent's winning move"
            else:
                # Use position evaluator to find best move
                best_moves = self.position_evaluator.get_best_moves(board, player, count=1)
                if not best_moves:
                    raise AIDecisionError("Position evaluator found no moves", board, 0)
                
                best_position, evaluation_score = best_moves[0]
                reasoning = "Best move by position evaluation"
        
        move = Move(position=best_position, player=player, evaluation_score=evaluation_score)
        move_time = time.time() - start_time
        
        metrics = AIPerformanceMetrics(
            move_time=move_time,
            search_depth=1,
            nodes_evaluated=len(empty_positions),
            transposition_hits=0,
            cutoffs=0,
            fallback_used=True,
            fallback_strategy=FallbackStrategy.SIMPLE_EVALUATION,
            evaluation_score=evaluation_score
        )
        
        confidence = 0.5 if evaluation_score > 100 else 0.3
        
        decision = AIDecision(
            move=move,
            metrics=metrics,
            confidence=confidence,
            reasoning=reasoning
        )
        
        self.decision_history.append(decision)
        return decision
    
    def _fallback_random_move(self, board: CircularBoard, player: Player, start_time: float) -> AIDecision:
        """Fallback strategy: Select a random legal move."""
        import random
        
        empty_positions = board.get_empty_positions()
        
        if not empty_positions:
            raise AIDecisionError("No empty positions for random move", board, 0)
        
        # Prefer center positions if available
        center_positions = [pos for pos in empty_positions if pos.ring == 0]
        if center_positions:
            selected_position = random.choice(center_positions)
            reasoning = "Random center position selected"
        else:
            selected_position = random.choice(empty_positions)
            reasoning = "Random position selected as last resort"
        
        move = Move(position=selected_position.id, player=player, evaluation_score=0.0)
        move_time = time.time() - start_time
        
        metrics = AIPerformanceMetrics(
            move_time=move_time,
            search_depth=0,
            nodes_evaluated=1,
            transposition_hits=0,
            cutoffs=0,
            fallback_used=True,
            fallback_strategy=FallbackStrategy.RANDOM_MOVE,
            evaluation_score=0.0
        )
        
        decision = AIDecision(
            move=move,
            metrics=metrics,
            confidence=0.1,
            reasoning=reasoning
        )
        
        self.decision_history.append(decision)
        return decision
    
    def _calculate_confidence(self, search_result: SearchResult, board: CircularBoard) -> float:
        """
        Calculate confidence level in the AI decision.
        
        Args:
            search_result: Result from the search algorithm
            board: Current board state
            
        Returns:
            Confidence level between 0.0 and 1.0
        """
        confidence = 0.5  # Base confidence
        
        # Adjust based on search depth
        if search_result.depth >= 6:
            confidence += 0.3
        elif search_result.depth >= 4:
            confidence += 0.2
        elif search_result.depth >= 2:
            confidence += 0.1
        
        # Adjust based on evaluation score
        abs_score = abs(search_result.score)
        if abs_score >= 1000:  # Winning/losing position
            confidence += 0.2
        elif abs_score >= 500:  # Strong position
            confidence += 0.1
        
        # Adjust based on time used
        if search_result.time_elapsed < self.GOOD_MOVE_TIME_THRESHOLD:
            confidence += 0.1
        
        # Adjust based on nodes evaluated
        if search_result.nodes_evaluated > 10000:
            confidence += 0.1
        
        return min(1.0, confidence)
    
    def _generate_reasoning(self, search_result: SearchResult, board: CircularBoard, player: Player) -> str:
        """
        Generate human-readable reasoning for the AI decision.
        
        Args:
            search_result: Result from the search algorithm
            board: Current board state
            player: Player making the move
            
        Returns:
            Human-readable explanation of the decision
        """
        reasoning_parts = []
        
        # Check for immediate tactical moves
        immediate_wins = self.win_detector.find_immediate_wins(board, player)
        if search_result.best_move.position in immediate_wins:
            reasoning_parts.append("Winning move")
        else:
            opponent = Player.O if player == Player.X else Player.X
            immediate_blocks = self.win_detector.find_immediate_wins(board, opponent)
            if search_result.best_move.position in immediate_blocks:
                reasoning_parts.append("Blocking opponent's win")
        
        # Describe search quality
        if search_result.depth >= 8:
            reasoning_parts.append(f"Deep search (depth {search_result.depth})")
        elif search_result.depth >= 5:
            reasoning_parts.append(f"Medium search (depth {search_result.depth})")
        else:
            reasoning_parts.append(f"Shallow search (depth {search_result.depth})")
        
        # Describe position evaluation
        if search_result.score > 500:
            reasoning_parts.append("Strong advantage")
        elif search_result.score > 100:
            reasoning_parts.append("Slight advantage")
        elif search_result.score < -500:
            reasoning_parts.append("Defensive move")
        elif search_result.score < -100:
            reasoning_parts.append("Equalizing move")
        else:
            reasoning_parts.append("Balanced position")
        
        # Add position information
        position = board.get_position(search_result.best_move.position)
        if position:
            ring_names = ["center", "inner", "middle", "outer"]
            if position.ring < len(ring_names):
                reasoning_parts.append(f"{ring_names[position.ring]} ring")
        
        return ", ".join(reasoning_parts)
    
    def _log_decision(self, decision: AIDecision, strategy_type: str):
        """Log AI decision for performance monitoring."""
        if not self.enable_logging:
            return
        
        self.logger.info(
            f"{strategy_type} - Move: {decision.move.position}, "
            f"Time: {decision.metrics.move_time:.3f}s, "
            f"Depth: {decision.metrics.search_depth}, "
            f"Nodes: {decision.metrics.nodes_evaluated}, "
            f"Score: {decision.metrics.evaluation_score:.2f}, "
            f"Confidence: {decision.confidence:.2f}"
        )
        
        if decision.metrics.fallback_used:
            self.logger.warning(f"Fallback strategy used: {decision.metrics.fallback_strategy.value}")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get summary of AI performance statistics.
        
        Returns:
            Dictionary with performance metrics
        """
        if self.total_decisions == 0:
            return {
                'total_decisions': 0,
                'average_time': 0.0,
                'fallback_rate': 0.0,
                'average_confidence': 0.0,
                'average_depth': 0.0,
                'total_nodes': 0
            }
        
        avg_time = self.total_time / self.total_decisions
        fallback_rate = self.fallback_count / self.total_decisions
        
        confidences = [d.confidence for d in self.decision_history]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        
        depths = [d.metrics.search_depth for d in self.decision_history]
        avg_depth = sum(depths) / len(depths) if depths else 0.0
        
        total_nodes = sum(d.metrics.nodes_evaluated for d in self.decision_history)
        
        return {
            'total_decisions': self.total_decisions,
            'average_time': avg_time,
            'fallback_rate': fallback_rate,
            'average_confidence': avg_confidence,
            'average_depth': avg_depth,
            'total_nodes': total_nodes,
            'fallback_count': self.fallback_count
        }
    
    def reset_performance_tracking(self):
        """Reset all performance tracking data."""
        self.decision_history.clear()
        self.total_decisions = 0
        self.total_time = 0.0
        self.fallback_count = 0
    
    def set_time_limit(self, time_limit: float):
        """
        Set the time limit for AI decisions.
        
        Args:
            time_limit: Time limit in seconds
        """
        if time_limit <= 0:
            raise ValueError("Time limit must be positive")
        
        self.time_limit = time_limit
        self.search_algorithm.set_time_limit(time_limit)
    
    def set_max_depth(self, max_depth: int):
        """
        Set the maximum search depth.
        
        Args:
            max_depth: Maximum search depth
        """
        if max_depth < 1:
            raise ValueError("Max depth must be at least 1")
        
        self.max_depth = max_depth
    
    def clear_transposition_table(self):
        """Clear the transposition table to free memory."""
        self.search_algorithm.clear_transposition_table()