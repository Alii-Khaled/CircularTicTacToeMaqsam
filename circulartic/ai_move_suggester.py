#!/usr/bin/env python3
"""
AI Move Suggester Script

This script takes a board representation as input and returns the AI's suggested move
for the circular tic-tac-toe game.

Usage:
    python ai_move_suggester.py <board_string> <current_player> [options]

Board String Format:
    32-character string representing positions 0-31, where:
    - 'X' = X player occupies this position
    - 'O' = O player occupies this position  
    - '_' = Empty position

Example:
    python ai_move_suggester.py "X_______________________________" X
    python ai_move_suggester.py "XO______________________________" X --time-limit 5.0
"""

import sys
import argparse
import json
from typing import Dict, Any

from src.game.board import CircularBoard
from src.models.move import Move
from src.models.enums import Player
from src.ai.engine import AIEngine


def parse_board_string(board_string: str) -> CircularBoard:
    """
    Parse a board string into a CircularBoard object.
    
    Args:
        board_string: 32-character string representing the board state
        
    Returns:
        CircularBoard object with the specified state
        
    Raises:
        ValueError: If board string is invalid
    """
    if len(board_string) != 32:
        raise ValueError(f"Board string must be exactly 32 characters, got {len(board_string)}")
    
    # Validate characters
    valid_chars = {'X', 'O', '_'}
    for i, char in enumerate(board_string):
        if char not in valid_chars:
            raise ValueError(f"Invalid character '{char}' at position {i}. Use 'X', 'O', or '_'")
    
    # Create empty board
    board = CircularBoard()
    
    # Reconstruct the board state by making moves in order
    moves = []
    for position_id, char in enumerate(board_string):
        if char != '_':
            player = Player.X if char == 'X' else Player.O
            moves.append((position_id, player))
    
    # Sort moves to ensure proper turn order (X goes first)
    x_moves = [(pos, player) for pos, player in moves if player == Player.X]
    o_moves = [(pos, player) for pos, player in moves if player == Player.O]
    
    # Validate move counts (X should have equal or one more move than O)
    if len(x_moves) < len(o_moves) or len(x_moves) > len(o_moves) + 1:
        raise ValueError(f"Invalid move count: X has {len(x_moves)} moves, O has {len(o_moves)} moves")
    
    # Interleave moves starting with X
    ordered_moves = []
    for i in range(max(len(x_moves), len(o_moves))):
        if i < len(x_moves):
            ordered_moves.append(x_moves[i])
        if i < len(o_moves):
            ordered_moves.append(o_moves[i])
    
    # Apply moves to board
    for position_id, player in ordered_moves:
        move = Move(position=position_id, player=player)
        if not board.make_move(move):
            raise ValueError(f"Invalid move: position {position_id} for player {player.value}")
    
    return board


def format_output(decision, format_type: str = 'human') -> str:
    """
    Format the AI decision output.
    
    Args:
        decision: AIDecision object
        format_type: Output format ('human', 'json', 'simple')
        
    Returns:
        Formatted output string
    """
    if format_type == 'json':
        output = {
            'suggested_move': decision.move.position,
            'player': decision.move.player.value,
            'confidence': decision.confidence,
            'reasoning': decision.reasoning,
            'metrics': {
                'move_time': decision.metrics.move_time,
                'search_depth': decision.metrics.search_depth,
                'nodes_evaluated': decision.metrics.nodes_evaluated,
                'fallback_used': decision.metrics.fallback_used,
                'fallback_strategy': decision.metrics.fallback_strategy.value if decision.metrics.fallback_strategy else None,
                'evaluation_score': decision.metrics.evaluation_score
            }
        }
        return json.dumps(output, indent=2)
    
    elif format_type == 'simple':
        return str(decision.move.position)
    
    else:  # human format
        output = []
        output.append(f"AI Suggested Move: {decision.move.position}")
        output.append(f"Player: {decision.move.player.value}")
        output.append(f"Confidence: {decision.confidence:.2f}")
        output.append(f"Reasoning: {decision.reasoning}")
        output.append("")
        output.append("Performance Metrics:")
        output.append(f"  Time taken: {decision.metrics.move_time:.3f}s")
        output.append(f"  Search depth: {decision.metrics.search_depth}")
        output.append(f"  Nodes evaluated: {decision.metrics.nodes_evaluated}")
        output.append(f"  Evaluation score: {decision.metrics.evaluation_score:.2f}")
        
        if decision.metrics.fallback_used:
            output.append(f"  Fallback strategy used: {decision.metrics.fallback_strategy.value}")
        
        return "\n".join(output)


def main():
    """Main function to handle command line arguments and run AI move suggestion."""
    parser = argparse.ArgumentParser(
        description="Get AI move suggestion for circular tic-tac-toe",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Get move for X on empty board
  python ai_move_suggester.py "________________________________" X
  
  # Get move for O after X played position 0
  python ai_move_suggester.py "X_______________________________" O
  
  # Get move with custom time limit and JSON output
  python ai_move_suggester.py "XO______________________________" X --time-limit 5.0 --format json
  
  # Simple output (just the position number)
  python ai_move_suggester.py "XO______________________________" X --format simple
        """
    )
    
    parser.add_argument(
        'board_string',
        help='32-character board representation (X/O/_ for each position 0-31)'
    )
    
    parser.add_argument(
        'current_player',
        choices=['X', 'O'],
        help='Current player to move (X or O)'
    )
    
    parser.add_argument(
        '--time-limit',
        type=float,
        default=5.0,
        help='Time limit for AI thinking in seconds (default: 3.0)'
    )
    
    parser.add_argument(
        '--max-depth',
        type=int,
        default=12,
        help='Maximum search depth (default: 12)'
    )
    
    parser.add_argument(
        '--format',
        choices=['human', 'json', 'simple'],
        default='human',
        help='Output format (default: human)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    try:
        # Parse inputs
        board = parse_board_string(args.board_string)
        current_player = Player.X if args.current_player == 'X' else Player.O
        
        # Validate that it's the correct player's turn
        if board.current_player != current_player:
            print(f"Error: Board indicates it's {board.current_player.value}'s turn, but you specified {current_player.value}", 
                  file=sys.stderr)
            sys.exit(1)
        
        # Create AI engine
        ai_engine = AIEngine(
            time_limit=args.time_limit,
            max_depth=args.max_depth,
            enable_logging=args.verbose
        )
        
        # Get AI decision
        decision = ai_engine.select_move(board, current_player)
        
        # Output result
        output = format_output(decision, args.format)
        print(output)
        
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()