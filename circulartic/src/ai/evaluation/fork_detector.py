"""
Fork detection system for circular tic-tac-toe AI.

This module detects fork threats - situations where a player can create
multiple simultaneous winning threats that cannot all be blocked.
"""
from typing import List, Dict, Set, Tuple
from dataclasses import dataclass

from ...models.enums import Player
from ...models.move import Move
from ...game.board import CircularBoard
from .win_detector import WinDetector


@dataclass
class ForkThreat:
    """
    Represents a fork threat on the board.
    
    Attributes:
        player: Player who has the fork threat
        trigger_position: Position that would create the fork
        resulting_threats: Positions that would become immediate wins after the fork
        severity: How dangerous this fork is (number of simultaneous threats)
        prevention_moves: Positions that would prevent this fork
    """
    player: Player
    trigger_position: int
    resulting_threats: List[int]
    severity: int
    prevention_moves: List[int]
    
    def is_unstoppable(self) -> bool:
        """Check if this fork creates more threats than opponent can block."""
        return self.severity >= 2


class ForkDetector:
    """
    Detects fork threats and suggests prevention strategies.
    
    A fork occurs when a player can create multiple simultaneous winning
    threats that cannot all be blocked by the opponent.
    """
    
    def __init__(self):
        """Initialize the fork detector."""
        self.win_detector = WinDetector()
    
    def find_fork_threats(self, board: CircularBoard, player: Player) -> List[ForkThreat]:
        """
        Find all fork threats for a specific player.
        
        Args:
            board: Current board state
            player: Player to find fork threats for
            
        Returns:
            List of fork threats sorted by severity (most dangerous first)
        """
        fork_threats = []
        empty_positions = [pos.id for pos in board.get_empty_positions()]
        
        # Test each empty position to see if it creates a fork
        for test_position in empty_positions:
            # Create a test board with the player's move
            test_board = board.copy()
            test_board.current_player = player
            test_move = Move(position=test_position, player=player)
            
            if test_board.make_move(test_move):
                # Check how many immediate wins this creates
                immediate_wins = self.win_detector.find_immediate_wins(test_board, player)
                
                if len(immediate_wins) >= 2:
                    # This is a fork! Find prevention moves
                    prevention_moves = self._find_prevention_moves(
                        board, player, test_position, immediate_wins
                    )
                    
                    fork_threat = ForkThreat(
                        player=player,
                        trigger_position=test_position,
                        resulting_threats=immediate_wins,
                        severity=len(immediate_wins),
                        prevention_moves=prevention_moves
                    )
                    
                    fork_threats.append(fork_threat)
        
        # Sort by severity (most dangerous first)
        fork_threats.sort(key=lambda f: f.severity, reverse=True)
        return fork_threats
    
    def find_fork_prevention_moves(self, board: CircularBoard, opponent: Player) -> List[int]:
        """
        Find moves that prevent opponent's fork threats.
        
        Args:
            board: Current board state
            opponent: Opponent player to defend against
            
        Returns:
            List of positions that prevent fork threats
        """
        fork_threats = self.find_fork_threats(board, opponent)
        prevention_moves = set()
        
        for threat in fork_threats:
            if threat.is_unstoppable():
                # This is a critical fork threat
                prevention_moves.update(threat.prevention_moves)
        
        return list(prevention_moves)
    
    def _find_prevention_moves(self, board: CircularBoard, player: Player, 
                             fork_position: int, resulting_threats: List[int]) -> List[int]:
        """
        Find moves that would prevent a specific fork.
        
        Args:
            board: Original board state (before fork move)
            player: Player creating the fork
            fork_position: Position that creates the fork
            resulting_threats: Immediate wins created by the fork
            
        Returns:
            List of positions that prevent this fork
        """
        prevention_moves = []
        opponent = Player.O if player == Player.X else Player.X
        empty_positions = [pos.id for pos in board.get_empty_positions()]
        
        # Test each empty position to see if it prevents the fork
        for test_position in empty_positions:
            if test_position == fork_position:
                # Playing at the fork position directly prevents it
                prevention_moves.append(test_position)
                continue
            
            # Test if playing here prevents the fork
            test_board = board.copy()
            test_board.current_player = opponent
            test_move = Move(position=test_position, player=opponent)
            
            if test_board.make_move(test_move):
                # Now test if the original fork still works
                test_board.current_player = player
                fork_move = Move(position=fork_position, player=player)
                
                if test_board.make_move(fork_move):
                    new_threats = self.win_detector.find_immediate_wins(test_board, player)
                    
                    # If the number of threats is reduced, this is a prevention move
                    if len(new_threats) < len(resulting_threats):
                        prevention_moves.append(test_position)
        
        return prevention_moves
    
    def evaluate_fork_danger(self, board: CircularBoard, player: Player) -> float:
        """
        Evaluate the overall fork danger level for a player.
        
        Args:
            board: Current board state
            player: Player to evaluate fork danger for
            
        Returns:
            Fork danger score (higher = more dangerous)
        """
        fork_threats = self.find_fork_threats(board, player)
        danger_score = 0.0
        
        for threat in fork_threats:
            if threat.is_unstoppable():
                # Unstoppable forks are extremely dangerous
                danger_score += 1000.0 * threat.severity
            else:
                # Regular fork threats are moderately dangerous
                danger_score += 200.0 * threat.severity
        
        return danger_score
    
    def has_critical_fork_threats(self, board: CircularBoard, player: Player) -> bool:
        """
        Check if a player has critical fork threats (2+ simultaneous threats).
        
        Args:
            board: Current board state
            player: Player to check
            
        Returns:
            True if player has critical fork threats
        """
        fork_threats = self.find_fork_threats(board, player)
        return any(threat.is_unstoppable() for threat in fork_threats)