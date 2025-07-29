"""
Minimax algorithm with alpha-beta pruning for circular tic-tac-toe AI.

This module implements the core search algorithm that uses minimax with alpha-beta pruning,
transposition tables, iterative deepening, and move ordering for optimal performance.
"""
import time
import math
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
from ...models.enums import Player
from ...models.move import Move
from ...game.board import CircularBoard
from ..evaluation.position_evaluator import PositionEvaluator
from ..evaluation.win_detector import WinDetector
from ..evaluation.fork_detector import ForkDetector


@dataclass
class SearchResult:
    """
    Result of a minimax search.
    
    Attributes:
        best_move: The best move found by the search
        score: Evaluation score of the best move
        depth: Depth reached in the search
        nodes_evaluated: Number of nodes evaluated during search
        time_elapsed: Time taken for the search in seconds
        principal_variation: Sequence of best moves found
    """
    best_move: Optional[Move]
    score: float
    depth: int
    nodes_evaluated: int
    time_elapsed: float
    principal_variation: List[Move]


@dataclass
class TranspositionEntry:
    """
    Entry in the transposition table for position caching.
    
    Attributes:
        score: Evaluation score for this position
        depth: Depth at which this position was evaluated
        flag: Type of bound (EXACT, LOWER_BOUND, UPPER_BOUND)
        best_move: Best move from this position
    """
    score: float
    depth: int
    flag: str  # 'EXACT', 'LOWER_BOUND', 'UPPER_BOUND'
    best_move: Optional[int]


class SearchAlgorithm:
    """
    Implements minimax algorithm with alpha-beta pruning and optimizations.
    
    Features:
    - Alpha-beta pruning for search tree reduction
    - Transposition table for position caching
    - Iterative deepening for time management
    - Move ordering for better pruning efficiency
    - Quiescence search for tactical positions
    """
    
    # Search constants
    MAX_DEPTH = 12
    TIME_LIMIT = 5.0  # Maximum 3 seconds per move
    TRANSPOSITION_TABLE_SIZE = 100000
    
    # Evaluation bounds
    MATE_SCORE = 10000
    DRAW_SCORE = 0
    
    # Transposition table flags
    EXACT = 'EXACT'
    LOWER_BOUND = 'LOWER_BOUND'
    UPPER_BOUND = 'UPPER_BOUND'
    
    def __init__(self):
        """Initialize the search algorithm."""
        self.position_evaluator = PositionEvaluator()
        self.win_detector = WinDetector()
        self.fork_detector = ForkDetector()
        
        # Search statistics
        self.nodes_evaluated = 0
        self.transposition_hits = 0
        self.cutoffs = 0
        
        # Transposition table
        self.transposition_table: Dict[str, TranspositionEntry] = {}
        
        # Time management
        self.start_time = 0.0
        self.time_limit = self.TIME_LIMIT
        
        # Principal variation
        self.principal_variation: List[Move] = []
        
        # Move ordering history
        self.killer_moves: Dict[int, List[int]] = {}  # depth -> [move1, move2]
        self.history_table: Dict[Tuple[int, int], int] = {}  # (from_pos, to_pos) -> score
    
    def search(self, board: CircularBoard, max_depth: int = None, 
               time_limit: float = None) -> SearchResult:
        """
        Perform iterative deepening search to find the best move.
        
        Args:
            board: Current board state
            max_depth: Maximum search depth (default: MAX_DEPTH)
            time_limit: Time limit in seconds (default: TIME_LIMIT)
            
        Returns:
            SearchResult with the best move and search statistics
        """
        if max_depth is None:
            max_depth = self.MAX_DEPTH
        if time_limit is None:
            time_limit = self.TIME_LIMIT
        
        self.start_time = time.time()
        self.time_limit = time_limit
        self.nodes_evaluated = 0
        self.transposition_hits = 0
        self.cutoffs = 0
        self.principal_variation = []
        
        # Store the root player for consistent evaluation
        self.root_player = board.current_player
        
        # Clear killer moves for this search
        self.killer_moves.clear()
        
        # Clear fork detection cache
        if hasattr(self, '_cached_fork_prevention'):
            delattr(self, '_cached_fork_prevention')
        
        best_move = None
        best_score = float('-inf')
        completed_depth = 0
        
        # Iterative deepening
        for depth in range(1, max_depth + 1):
            if self._is_time_up():
                break
            
            try:
                result = self._minimax_root(board, depth)
                
                if result.best_move is not None:
                    best_move = result.best_move
                    best_score = result.score
                    completed_depth = depth
                    
                    # Update principal variation
                    self.principal_variation = result.principal_variation
                
                # If we found a mate, no need to search deeper
                if abs(best_score) >= self.MATE_SCORE - 100:
                    break
                    
            except TimeoutError:
                break
        
        time_elapsed = time.time() - self.start_time
        
        return SearchResult(
            best_move=best_move,
            score=best_score,
            depth=completed_depth,
            nodes_evaluated=self.nodes_evaluated,
            time_elapsed=time_elapsed,
            principal_variation=self.principal_variation.copy()
        )
    
    def _minimax_root(self, board: CircularBoard, depth: int) -> SearchResult:
        """
        Root node of minimax search.
        
        Args:
            board: Current board state
            depth: Search depth
            
        Returns:
            SearchResult with best move found
        """
        player = board.current_player
        best_move = None
        best_score = float('-inf')
        alpha = float('-inf')
        beta = float('+inf')
        principal_variation = []
        
        # Get ordered moves
        moves = self._get_ordered_moves(board, depth)
        
        if not moves:
            return SearchResult(None, self.DRAW_SCORE, depth, self.nodes_evaluated, 0.0, [])
        
        for move_pos in moves:
            if self._is_time_up():
                raise TimeoutError("Search time limit exceeded")
            
            # Make the move
            move = Move(position=move_pos, player=player)
            board_copy = board.copy()
            if not board_copy.make_move(move):
                continue
            
            # Check for immediate win first
            win_result = self.win_detector.check_win(board_copy)
            if win_result and win_result.winner == player:
                # Immediate win - return immediately with highest score
                return SearchResult(
                    best_move=move,
                    score=self.MATE_SCORE - (self.MAX_DEPTH - depth),
                    depth=depth,
                    nodes_evaluated=self.nodes_evaluated,
                    time_elapsed=time.time() - self.start_time,
                    principal_variation=[move]
                )
            
            # Check if this blocks opponent's immediate win or fork threats
            opponent = Player.O if player == Player.X else Player.X
            opponent_wins_before = self.win_detector.find_immediate_wins(board, opponent)
            opponent_wins_after = self.win_detector.find_immediate_wins(board_copy, opponent)
            
            # Check for fork threats using the dedicated fork detector
            fork_prevention_moves = self.fork_detector.find_fork_prevention_moves(board, opponent)
            is_fork_prevention = move_pos in fork_prevention_moves
            
            if len(opponent_wins_before) > len(opponent_wins_after):
                # This move blocks at least one immediate threat - give it very high priority
                score = self.MATE_SCORE - 100  # Slightly less than immediate win
            elif is_fork_prevention:
                # This move blocks a fork threat - also very high priority
                score = self.MATE_SCORE - 200  # Slightly less than blocking immediate win
            else:
                # Search this move normally
                score = -self._minimax(board_copy, depth - 1, -beta, -alpha, False)
            
            if score > best_score:
                best_score = score
                best_move = move
                alpha = max(alpha, score)
                
                # Update principal variation
                principal_variation = [move] + self.principal_variation
        
        return SearchResult(
            best_move=best_move,
            score=best_score,
            depth=depth,
            nodes_evaluated=self.nodes_evaluated,
            time_elapsed=time.time() - self.start_time,
            principal_variation=principal_variation
        )
    
    def _minimax(self, board: CircularBoard, depth: int, alpha: float, beta: float, 
                maximizing: bool) -> float:
        """
        Minimax algorithm with alpha-beta pruning.
        
        Args:
            board: Current board state
            depth: Remaining search depth
            alpha: Alpha value for pruning
            beta: Beta value for pruning
            maximizing: True if maximizing player, False if minimizing
            
        Returns:
            Evaluation score for this position
        """
        self.nodes_evaluated += 1
        
        if self._is_time_up():
            raise TimeoutError("Search time limit exceeded")
        
        # Check transposition table
        board_hash = board.get_board_state_hash()
        tt_entry = self.transposition_table.get(board_hash)
        
        if tt_entry and tt_entry.depth >= depth:
            self.transposition_hits += 1
            
            if tt_entry.flag == self.EXACT:
                return tt_entry.score
            elif tt_entry.flag == self.LOWER_BOUND and tt_entry.score >= beta:
                return tt_entry.score
            elif tt_entry.flag == self.UPPER_BOUND and tt_entry.score <= alpha:
                return tt_entry.score
        
        # Terminal node checks
        win_result = self.win_detector.check_win(board)
        if win_result:
            # Someone won the game
            # In minimax, we need to return the score from the perspective of the current node
            # The root always maximizes for the root player
            # At each level, maximizing alternates
            
            # The key insight: if maximizing=True, we want high scores for good outcomes
            # If maximizing=False, we want low scores for good outcomes (from opponent's perspective)
            
            # Since the game is over, we return a mate score
            # Positive mate score = good for the maximizing player
            # Negative mate score = bad for the maximizing player
            
            if maximizing:
                # We're maximizing - return positive score (good outcome)
                return self.MATE_SCORE - (self.MAX_DEPTH - depth)
            else:
                # We're minimizing - return negative score (bad outcome for maximizer)
                return -self.MATE_SCORE + (self.MAX_DEPTH - depth)
        
        if board.is_full():
            return self.DRAW_SCORE
        
        if depth == 0:
            return self._quiescence_search(board, alpha, beta, maximizing, 3)
        
        # Get ordered moves
        moves = self._get_ordered_moves(board, depth)
        
        if not moves:
            return self.DRAW_SCORE
        
        best_score = float('-inf') if maximizing else float('+inf')
        best_move = None
        original_alpha = alpha
        
        for move_pos in moves:
            # Make the move
            player = board.current_player
            move = Move(position=move_pos, player=player)
            board_copy = board.copy()
            
            if not board_copy.make_move(move):
                continue
            
            # Recursive search
            score = self._minimax(board_copy, depth - 1, alpha, beta, not maximizing)
            
            if maximizing:
                if score > best_score:
                    best_score = score
                    best_move = move_pos
                alpha = max(alpha, score)
                
                if beta <= alpha:
                    self.cutoffs += 1
                    self._update_killer_moves(depth, move_pos)
                    break
            else:
                if score < best_score:
                    best_score = score
                    best_move = move_pos
                beta = min(beta, score)
                
                if beta <= alpha:
                    self.cutoffs += 1
                    self._update_killer_moves(depth, move_pos)
                    break
        
        # Store in transposition table
        self._store_transposition(board_hash, best_score, depth, original_alpha, beta, best_move)
        
        return best_score
    
    def _quiescence_search(self, board: CircularBoard, alpha: float, beta: float, 
                          maximizing: bool, depth: int) -> float:
        """
        Quiescence search to handle tactical positions.
        
        Args:
            board: Current board state
            alpha: Alpha value for pruning
            beta: Beta value for pruning
            maximizing: True if maximizing player
            depth: Remaining quiescence depth
            
        Returns:
            Evaluation score
        """
        self.nodes_evaluated += 1
        
        if self._is_time_up() or depth <= 0:
            # Always evaluate from the root player's perspective
            # The root player is stored when search begins
            root_player = getattr(self, 'root_player', board.current_player)
            eval_score = self.position_evaluator.evaluate_position(board, root_player)
            return eval_score if maximizing else -eval_score
        
        # Stand pat evaluation
        root_player = getattr(self, 'root_player', board.current_player)
        stand_pat = self.position_evaluator.evaluate_position(board, root_player)
        if not maximizing:
            stand_pat = -stand_pat
        
        if maximizing:
            if stand_pat >= beta:
                return beta
            alpha = max(alpha, stand_pat)
        else:
            if stand_pat <= alpha:
                return alpha
            beta = min(beta, stand_pat)
        
        # Only search tactical moves (threats and captures)
        tactical_moves = self._get_tactical_moves(board)
        
        for move_pos in tactical_moves:
            player = board.current_player
            move = Move(position=move_pos, player=player)
            board_copy = board.copy()
            
            if not board_copy.make_move(move):
                continue
            
            score = self._quiescence_search(board_copy, alpha, beta, not maximizing, depth - 1)
            
            if maximizing:
                alpha = max(alpha, score)
                if beta <= alpha:
                    break
            else:
                beta = min(beta, score)
                if beta <= alpha:
                    break
        
        return alpha if maximizing else beta
    
    def _get_ordered_moves(self, board: CircularBoard, depth: int) -> List[int]:
        """
        Get moves ordered for better alpha-beta pruning.
        
        Args:
            board: Current board state
            depth: Current search depth
            
        Returns:
            List of position IDs ordered by likely strength
        """
        empty_positions = [pos.id for pos in board.get_empty_positions()]
        
        if not empty_positions:
            return []
        
        player = board.current_player
        move_scores = []
        
        # Check transposition table for best move
        board_hash = board.get_board_state_hash()
        tt_entry = self.transposition_table.get(board_hash)
        tt_move = tt_entry.best_move if tt_entry else None
        
        # Get killer moves for this depth
        killer_moves = self.killer_moves.get(depth, [])
        
        for pos_id in empty_positions:
            score = 0
            
            # Transposition table move gets highest priority
            if pos_id == tt_move:
                score += 10000
            
            # Killer moves get high priority
            elif pos_id in killer_moves:
                score += 5000 + (1000 * (len(killer_moves) - killer_moves.index(pos_id)))
            
            # Immediate wins get very high priority
            test_board = board.copy()
            move = Move(position=pos_id, player=player)
            if test_board.make_move(move):
                win_result = self.win_detector.check_win(test_board)
                if win_result and win_result.winner == player:
                    score += 20000
                
                # Blocking opponent wins gets high priority
                opponent = Player.O if player == Player.X else Player.X
                opponent_wins_before = self.win_detector.find_immediate_wins(board, opponent)
                opponent_wins_after = self.win_detector.find_immediate_wins(test_board, opponent)
                
                # If opponent had immediate wins before but not after, this is a blocking move
                if len(opponent_wins_before) > len(opponent_wins_after):
                    score += 15000
                
                # Check for fork threats and prevention (cached for performance)
                if not hasattr(self, '_cached_fork_prevention'):
                    self._cached_fork_prevention = self.fork_detector.find_fork_prevention_moves(board, opponent)
                
                if pos_id in self._cached_fork_prevention:
                    score += 12000  # High priority for fork prevention
                
                # Skip expensive fork creation check for move ordering to improve performance
                # Fork creation will be handled in the main search evaluation
                
                # Position evaluation
                eval_score = self.position_evaluator.evaluate_position(test_board, player)
                score += eval_score
            
            # History table bonus
            history_score = self.history_table.get((pos_id, pos_id), 0)
            score += history_score
            
            move_scores.append((pos_id, score))
        
        # Sort by score (highest first)
        move_scores.sort(key=lambda x: x[1], reverse=True)
        
        return [pos_id for pos_id, _ in move_scores]
    
    def _get_tactical_moves(self, board: CircularBoard) -> List[int]:
        """
        Get tactical moves for quiescence search.
        
        Only include moves that are critical for the current player to consider.
        This prevents the search from helping the opponent find good moves.
        
        Args:
            board: Current board state
            
        Returns:
            List of tactical move positions
        """
        player = board.current_player
        opponent = Player.O if player == Player.X else Player.X
        root_player = getattr(self, 'root_player', player)
        
        tactical_moves = set()
        
        # Only explore tactical moves if current player is the root player
        # This prevents helping the opponent find good responses
        if player == root_player:
            # Immediate wins for root player
            wins = self.win_detector.find_immediate_wins(board, player)
            tactical_moves.update(wins)
            
            # High-value threats for root player
            threats = self.win_detector.find_threats(board, player)
            for threat in threats:
                if threat.severity >= 3:  # Only very high threats
                    tactical_moves.update(threat.completing_moves)
            
            # Fork creation moves for root player
            fork_threats = self.fork_detector.find_fork_threats(board, player)
            for fork_threat in fork_threats:
                if fork_threat.is_unstoppable():
                    tactical_moves.add(fork_threat.trigger_position)
        else:
            # If it's opponent's turn, only consider defensive moves
            # Immediate wins for opponent (we need to be aware of these)
            opponent_wins = self.win_detector.find_immediate_wins(board, player)
            if opponent_wins:
                # If opponent has immediate wins, this is a critical position
                # Include the winning moves so we can see the threat
                tactical_moves.update(opponent_wins)
        
        # Always include fork prevention moves regardless of whose turn it is
        # This ensures we're aware of critical defensive positions
        fork_prevention_moves = self.fork_detector.find_fork_prevention_moves(board, opponent)
        tactical_moves.update(fork_prevention_moves)
        
        return list(tactical_moves)
    
    def _update_killer_moves(self, depth: int, move: int):
        """
        Update killer moves table.
        
        Args:
            depth: Search depth where cutoff occurred
            move: Move that caused the cutoff
        """
        if depth not in self.killer_moves:
            self.killer_moves[depth] = []
        
        killers = self.killer_moves[depth]
        
        # Add move if not already present
        if move not in killers:
            killers.insert(0, move)
            # Keep only top 2 killer moves per depth
            if len(killers) > 2:
                killers.pop()
    
    def _store_transposition(self, board_hash: str, score: float, depth: int, 
                           original_alpha: float, beta: float, best_move: Optional[int]):
        """
        Store position in transposition table.
        
        Args:
            board_hash: Hash of the board position
            score: Evaluation score
            depth: Search depth
            original_alpha: Original alpha value
            beta: Beta value
            best_move: Best move from this position
        """
        # Determine flag type
        if score <= original_alpha:
            flag = self.UPPER_BOUND
        elif score >= beta:
            flag = self.LOWER_BOUND
        else:
            flag = self.EXACT
        
        # Store entry
        entry = TranspositionEntry(
            score=score,
            depth=depth,
            flag=flag,
            best_move=best_move
        )
        
        self.transposition_table[board_hash] = entry
        
        # Limit table size
        if len(self.transposition_table) > self.TRANSPOSITION_TABLE_SIZE:
            # Remove oldest entries (simple FIFO)
            keys_to_remove = list(self.transposition_table.keys())[:1000]
            for key in keys_to_remove:
                del self.transposition_table[key]
    
    def _is_time_up(self) -> bool:
        """Check if search time limit has been exceeded."""
        return time.time() - self.start_time >= self.time_limit
    
    def get_search_statistics(self) -> Dict[str, int]:
        """Get search statistics."""
        return {
            'nodes_evaluated': self.nodes_evaluated,
            'transposition_hits': self.transposition_hits,
            'cutoffs': self.cutoffs,
            'transposition_table_size': len(self.transposition_table)
        }
    
    def clear_transposition_table(self):
        """Clear the transposition table."""
        self.transposition_table.clear()
    
    def set_time_limit(self, time_limit: float):
        """Set the time limit for searches."""
        self.time_limit = time_limit