from __future__ import annotations

from typing import Optional, Dict, List, Tuple
import chess
import torch
import random

from net import load_policy_value_net, ChessNet
from mcts import MCTS

__all__ = ["ChessRLAgent"]

# Rough material values (centipawns) for quick heuristics / material accounting
VAL = {1: 100, 2: 300, 3: 310, 4: 500, 5: 900, 6: 10_000}  # PAWN..KING


def _rep_key(b: chess.Board) -> str:
    """Key used to count repetitions (FEN without clocks)."""
    f = b.fen().split()
    return " ".join(f[:4]) if len(f) >= 4 else b.board_fen() + (" w" if b.turn else " b")


def _pos_key(b: chess.Board) -> str:
    """Key for current piece layout + side to move."""
    return f"{b.board_fen()} {'w' if b.turn else 'b'}"


def _build_microbook() -> Dict[str, List[str]]:
    """
    Micro opening book
    - Position-keyed mapping: pos_key -> list of recommended UCI moves.
    - add_line(...) takes a full UCI move list and registers the next move at
      every intermediate position along the line (both White and Black turns).
    - We keep them principled (castle, develop, centre) to avoid cheap tactics.
    """
    book: Dict[str, List[str]] = {}

    def add(pos: chess.Board, uci: str) -> None:
        k = _pos_key(pos)
        book.setdefault(k, [])
        if uci not in book[k]:
            book[k].append(uci)

    def add_line(ucis: List[str]) -> None:
        b = chess.Board()
        for u in ucis:
            # register 'u' as the recommended move at position b
            add(b, u)
            try:
                b.push_uci(u)
            except Exception:
                break  # illegal in this branch; ignore the rest of the line

    # Italian
    # 1.e4 e5 2.Nf3 Nc6 3.Bc4 Bc5 4.d3 Nf6 5.O-O d6 6.c3 a6 7.Bb3 O-O 8.Re1
    add_line(["e2e4","e7e5","g1f3","b8c6","f1c4","f8c5","d2d3","g8f6",
              "e1g1","d7d6","c2c3","a7a6","c4b3","e8g8","f1e1"])

    # Ruy Lopez
    # 1.e4 e5 2.Nf3 Nc6 3.Bb5 a6 4.Ba4 Nf6 5.O-O Be7 6.Re1 b5 7.Bb3 d6 8.c3 O-O
    add_line(["e2e4","e7e5","g1f3","b8c6","f1b5","a7a6","b5a4","g8f6",
              "e1g1","f8e7","f1e1","b7b5","a4b3","d7d6","c2c3","e8g8"])

    # Sicilian Scheveningen
    # 1.e4 c5 2.Nf3 d6 3.d4 cxd4 4.Nxd4 Nf6 5.Nc3 e6 6.Be2 Be7 7.O-O O-O 8.Be3
    add_line(["e2e4","c7c5","g1f3","d7d6","d2d4","c5d4","f3d4","g8f6",
              "b1c3","e7e6","f1e2","f8e7","e1g1","e8g8","c1e3"])

    # Caro-Kann Classical
    # 1.e4 c6 2.d4 d5 3.Nc3 dxe4 4.Nxe4 Bf5 5.Ng3 Bg6 6.Nf3 Nd7 7.Bd3 e6 8.O-O
    add_line(["e2e4","c7c6","d2d4","d7d5","b1c3","d5e4","c3e4","c8f5",
              "e4g3","f5g6","g1f3","b8d7","f1d3","e7e6","e1g1"])

    # Queen’s Gambit Declined
    # 1.d4 d5 2.c4 e6 3.Nc3 Nf6 4.Nf3 Be7 5.Bg5 O-O 6.e3 Nbd7 7.Rc1 c6 8.Bd3
    add_line(["d2d4","d7d5","c2c4","e7e6","b1c3","g8f6","g1f3","f8e7",
              "c1g5","e8g8","e2e3","b8d7","a1c1","c7c6","f1d3"])

    # Slav / Semi-Slav
    # 1.d4 d5 2.c4 c6 3.Nf3 Nf6 4.Nc3 e6 5.e3 Nbd7 6.Bd3 Bd6 7.O-O O-O
    add_line(["d2d4","d7d5","c2c4","c7c6","g1f3","g8f6","b1c3","e7e6",
              "e2e3","b8d7","f1d3","f8d6","e1g1","e8g8"])

    # King’s Indian
    # 1.d4 Nf6 2.c4 g6 3.Nc3 Bg7 4.e4 d6 5.Nf3 O-O 6.Be2 e5 7.O-O Nc6
    add_line(["d2d4","g8f6","c2c4","g7g6","b1c3","f8g7","e2e4","d7d6",
              "g1f3","e8g8","f1e2","e7e5","e1g1","b8c6"])

    # Nimzo-Indian
    # 1.d4 Nf6 2.c4 e6 3.Nc3 Bb4 4.e3 O-O 5.Bd3 d5 6.Nf3 c5 7.O-O
    add_line(["d2d4","g8f6","c2c4","e7e6","b1c3","f8b4","e2e3","e8g8",
              "f1d3","d7d5","g1f3","c7c5","e1g1"])

    # English
    # 1.c4 e5 2.Nc3 Nf6 3.g3 d5 4.cxd5 Nxd5 5.Bg2 Nb6 6.Nf3 Nc6 7.O-O Be7
    add_line(["c2c4","e7e5","b1c3","g8f6","g2g3","d7d5","c4d5","f6d5",
              "f1g2","d5b6","g1f3","b8c6","e1g1","f8e7"])

    # London
    # 1.d4 d5 2.Nf3 Nf6 3.Bf4 e6 4.e3 Bd6 5.Bxd6 cxd6 6.Nbd2 O-O 7.Bd3 Nbd7 8.O-O
    add_line(["d2d4","d7d5","g1f3","g8f6","c1f4","e7e6","e2e3","f8d6",
              "f4d6","c7c6","b1d2","e8g8","f1d3","b8d7","e1g1"])
    
    # French Defence
    # 1.e4 e6 2.d4 d5 3.Nc3 Nf6 4.Bg5 Be7 5.e5 Nfd7 6.f4 c5 7.Nf3
    add_line(["e2e4","e7e6","d2d4","d7d5","b1c3","g8f6","c1g5","f8e7","e4e5","f6d7","f2f4","c7c5","g1f3"])

    # Alternative French: Tarrasch 3.Nd2 … 4.Ngf3 5.c3 6.Bd3 7.O-O
    add_line(["e2e4","e7e6","d2d4","d7d5","b1d2","g8f6","g1f3","c7c5","c2c3","f8e7","f1d3","e8g8"])

    # Scandinavian
    # 1.e4 d5 2.exd5 Qxd5 3.Nc3 Qa5 4.d4 Nf6 5.Nf3 c6 6.Bd3 e6 7.O-O
    add_line(["e2e4","d7d5","e4d5","d8d5","b1c3","d5a5","d2d4","g8f6","g1f3","c7c6","f1d3","e7e6","e1g1"])

    # Pirc/Modern
    # 1.e4 d6 2.d4 Nf6 3.Nc3 g6 4.Nf3 Bg7 5.Be2 O-O 6.O-O e5 7.Re1
    add_line(["e2e4","d7d6","d2d4","g8f6","b1c3","g7g6","g1f3","f8g7","f1e2","e8g8","e1g1","e7e5","f1e1"])

    # 1.d4 e6 2.c4 d5 3.Nc3 Nf6 4.Nf3 Be7 5.Bg5 O-O 6.e3 Nbd7
    add_line(["d2d4","e7e6","c2c4","d7d5","b1c3","g8f6","g1f3","f8e7","c1g5","e8g8","e2e3","b8d7"])

    # London transposition
    add_line(["d2d4","e7e6","g1f3","d7d5","c1f4","g8f6","e2e3","f8d6","f4d6","c7c6","e1g1"])

    return book


class ChessRLAgent:
    """
    RL-guided agent that uses a policy–value net with MCTS plus safety filters.
    Flow: (book / quick tactics) → safe MCTS choice → heuristic fallback.
    """
    def __init__(
        self,
        model_path: str,
        mcts_sims: int = 160,
        device: Optional[str] = None,
        value_white_pov: bool = True,
        logit_temperature: float = 1.25,
        c_puct: float = 1.25,
    ) -> None:
        """Load the net, set up MCTS, and prepare small helpers."""
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.net: ChessNet = load_policy_value_net(model_path, device=self.device)
        self.mcts = MCTS(
            net=self.net,
            sims=mcts_sims,
            c_puct=c_puct,
            logit_temperature=logit_temperature,
            value_white_pov=value_white_pov,
            device=self.device,
        )
        self._book = _build_microbook()
        self._seen: Dict[str, int] = {}  # repetition memory

    # Capture/recapture chain evaluator
    def _see_legal_chain(self, b: chess.Board, mv: chess.Move) -> int:
        """Evaluate the capture/recapture sequence on mv.to_square using only legal moves."""
        if not b.is_capture(mv):
            return 0

        to = mv.to_square
        bb = b.copy()

        # value of the first victim (handle en passant)
        if bb.is_en_passant(mv):
            first_val = VAL[chess.PAWN]
        else:
            victim = bb.piece_at(to)
            first_val = 0 if victim is None else VAL[victim.piece_type]

        gains = [first_val]

        # promotion material delta on the initial move (e.g., PxX=Q adds ~800 over a pawn)
        if mv.promotion:
            gains[0] += VAL[mv.promotion] - VAL[chess.PAWN]

        bb.push(mv)

        # always capture back on 'to' with the least valuable attacker
        while True:
            caps = [m for m in bb.legal_moves if m.to_square == to and bb.is_capture(m)]
            if not caps:
                break
            caps.sort(key=lambda m: VAL[bb.piece_at(m.from_square).piece_type])
            reply = caps[0]

            cur = bb.piece_at(to)
            cur_val = 0 if cur is None else VAL[cur.piece_type]
            gains.append(cur_val)

            if reply.promotion:
                gains[-1] += VAL[reply.promotion] - VAL[chess.PAWN]

            bb.push(reply)

        # negamax fold (swap-list propagation)
        for i in range(len(gains) - 2, -1, -1):
            gains[i] = max(gains[i], -gains[i + 1])

        return gains[0]
    
    def _bishop_into_g_pawn_snap(self, b: chess.Board, mv: chess.Move) -> bool:
        p = b.piece_at(mv.from_square)
        if not p or p.piece_type != chess.BISHOP or b.fullmove_number > 15:
            return False
        to = mv.to_square
        bad_targets = (chess.H6, chess.H3) if b.turn == chess.WHITE else (chess.H3, chess.H6)
        if to not in bad_targets:
            return False
        bb = b.copy()
        try:
            bb.push(mv)
        except Exception:
            return True
        g_from = chess.G7 if b.turn == chess.WHITE else chess.G2
        for m2 in bb.legal_moves:
            if m2.from_square == g_from and m2.to_square == to and bb.is_capture(m2):
                return self._see_legal_chain(bb, m2) >= 0
        return False

    # Development / structure helpers
    def _dev_debt(self, b: chess.Board) -> int:
        """Count undeveloped minors on back rank + unspent castling right early."""
        us = b.turn
        debt = 0
        if us == chess.WHITE:
            if b.piece_at(chess.B1) == chess.Piece(chess.KNIGHT, chess.WHITE): debt += 1
            if b.piece_at(chess.G1) == chess.Piece(chess.KNIGHT, chess.WHITE): debt += 1
            if b.piece_at(chess.C1) == chess.Piece(chess.BISHOP, chess.WHITE): debt += 1
            if b.piece_at(chess.F1) == chess.Piece(chess.BISHOP, chess.WHITE): debt += 1
        else:
            if b.piece_at(chess.B8) == chess.Piece(chess.KNIGHT, chess.BLACK): debt += 1
            if b.piece_at(chess.G8) == chess.Piece(chess.KNIGHT, chess.BLACK): debt += 1
            if b.piece_at(chess.C8) == chess.Piece(chess.BISHOP, chess.BLACK): debt += 1
            if b.piece_at(chess.F8) == chess.Piece(chess.BISHOP, chess.BLACK): debt += 1
        if b.fullmove_number <= 20 and self._has_any_castling_rights(b, us):
            debt += 1
        return debt

    def _reduces_dev_debt(self, b: chess.Board, mv: chess.Move) -> bool:
        """True if the move develops a minor or castles."""
        p = b.piece_at(mv.from_square)
        if not p:
            return False
        if p.piece_type in (chess.BISHOP, chess.KNIGHT):
            if p.color == chess.WHITE and chess.square_rank(mv.from_square) == 0 and chess.square_rank(mv.to_square) > 0:
                return True
            if p.color == chess.BLACK and chess.square_rank(mv.from_square) == 7 and chess.square_rank(mv.to_square) < 7:
                return True
        return p.piece_type == chess.KING and b.is_castling(mv)

    def _invites_bishop_snap(self, b: chess.Board, mv: chess.Move) -> bool:
        """Detect N→a3/h3/a6/h6 that can be safely taken by enemy bishop at once."""
        p = b.piece_at(mv.from_square)
        if not p or p.piece_type != chess.KNIGHT:
            return False
        to = mv.to_square
        if to not in (chess.A3, chess.H3, chess.A6, chess.H6):
            return False
        bb = b.copy()
        try:
            bb.push(mv)
        except Exception:
            return False
        them = not b.turn
        for m2 in bb.legal_moves:
            if m2.to_square == to and bb.is_capture(m2):
                pc2 = bb.piece_at(m2.from_square)
                if pc2 and pc2.color == them and pc2.piece_type == chess.BISHOP:
                    if self._see_legal_chain(bb, m2) >= 0:
                        return True
        return False

    
    def _early_ng5_without_threat(self, b: chess.Board, mv: chess.Move) -> bool:
        if b.fullmove_number > 12: 
            return False
        p = b.piece_at(mv.from_square)
        if not p or p.piece_type != chess.KNIGHT:
            return False
        if mv.to_square not in (chess.G5, chess.B5):  # mirror idea on both wings
            return False
        # Require immediate check, winning SEE capture, or mate threat next move
        bb = b.copy(); bb.push(mv)
        if bb.is_check():
            return False
        # try one-ply forcing follow-ups
        forcing = any(bb.is_capture(m2) or bb.gives_check(m2) for m2 in bb.legal_moves)
        return not forcing

    def _qc2_b3_knight_tactic_risk(self, b: chess.Board, mv: chess.Move) -> bool:
        """Early Qc2/Qb3 can run into Na4/Nb4; mark as risky in first 10 moves."""
        p = b.piece_at(mv.from_square)
        if not p or p.piece_type != chess.QUEEN or b.fullmove_number > 10:
            return False
        return mv.to_square in (chess.C2, chess.B3)

    # King-walk and flank-pawn vetoes
    def _queens_on_board(self, b: chess.Board) -> bool:
        """Quick check if any queen exists."""
        return any(pc.piece_type == chess.QUEEN for pc in b.piece_map().values())

    def _unsafe_king_walk(self, b: chess.Board, mv: chess.Move) -> bool:
        """
        Hard veto for non-castling king moves that increase exposure.
        Stricter in opening/middlegame, looser in true endgames.
        """
        p = b.piece_at(mv.from_square)
        if not p or p.piece_type != chess.KING or b.is_castling(mv):
            return False

        # Extra pre-check
        if self._queens_on_board(b) and b.fullmove_number <= 30:
            to_f, to_r = chess.square_file(mv.to_square), chess.square_rank(mv.to_square)
            if to_f in (2, 3, 4, 5) and to_r in (2, 3, 4, 5):
                return True

        bb = b.copy()
        try:
            bb.push(mv)
        except Exception:
            # Illegal
            return True

        if bb.is_checkmate():
            return False

        # Opening/middlegame
        if b.fullmove_number <= 20:
            them = not b.turn
            if b.is_check():
                # Still in check after moving the king → unsafe
                if bb.is_check():
                    return True
                # Allow capturing out of check if SEE says it’s not losing
                if b.is_capture(mv) and self._see_legal_chain(b, mv) >= 0:
                    return False
                # Safe if destination square isn’t attacked
                if not bb.is_attacked_by(them, mv.to_square):
                    return False
                return True
            else:
                # Not in check
                if b.is_capture(mv) and self._see_legal_chain(b, mv) >= 0 and not bb.is_check():
                    return False
                return True

        # Long-diagonal cheap trap
        if self._queens_on_board(b) and not self._is_endgame(b):
            us = b.turn
            them = not us
            q_sq = next((s for s, pc in bb.piece_map().items()
                        if pc.color == them and pc.piece_type == chess.QUEEN), None)
            if q_sq is not None:
                corners = (chess.A1, chess.H1) if us == chess.WHITE else (chess.A8, chess.H8)
                qf, qr = chess.square_file(q_sq), chess.square_rank(q_sq)
                for c in corners:
                    cf, cr = chess.square_file(c), chess.square_rank(c)
                    # Coarse: same diagonal → likely airy ray (e.g., …Qg7-a1)
                    if abs(qf - cf) == abs(qr - cr):
                        return True

        if self._queens_on_board(b) or not self._is_endgame(b):
            us, them = b.turn, not b.turn
            if bb.is_attacked_by(them, mv.to_square) and not bb.is_attacked_by(us, mv.to_square):
                return True
            f = chess.square_file(mv.to_square)
            if f in (2, 3, 4, 5):
                return True

        return False


    def _forbid_flank_pawn_while_undeveloped(self, b: chess.Board, mv: chess.Move) -> bool:
        """Block a/h pawn pushes in first 10 moves if development is behind and it's not tactical."""
        p = b.piece_at(mv.from_square)
        if not p or p.piece_type != chess.PAWN:
            return False

        file_from = chess.square_file(mv.from_square)

        # a/h pawns early when behind in development
        if b.fullmove_number <= 10 and file_from in (0, 7):
            if not (b.is_capture(mv) or b.gives_check(mv) or self._reduces_dev_debt(b, mv)):
                return self._dev_debt(b) > 1

        # NEW: f/g pawns before castling while queens are on (unless forcing)
        if b.fullmove_number <= 12 and file_from in (5, 6) and self._queens_on_board(b):
            if not (b.is_capture(mv) or b.gives_check(mv)):
                if self._has_any_castling_rights(b, b.turn):
                    return True
        
        if file_from in (6, 7) and self._queens_on_board(b):
            ksq = b.king(b.turn)
            if ksq in (chess.G1, chess.G8) and not (b.is_capture(mv) or b.gives_check(mv)):
                # allow one-step luft (…h3/…h6), block the two-step lunge (…g4/…h4)
                if mv.from_square in (chess.G2, chess.H2, chess.G7, chess.H7) and \
                mv.to_square in (chess.G4, chess.H4):
                    return True

        return False
    
    def _corner_rook_bishop_trap(self, b: chess.Board, mv: chess.Move) -> bool:
        p = b.piece_at(mv.from_square)
        if not p or p.piece_type != chess.BISHOP or not b.is_capture(mv):
            return False
        if mv.to_square not in (chess.A1, chess.H1, chess.A8, chess.H8):
            return False
        bb = b.copy(); bb.push(mv)
        us, them = b.turn, not b.turn
        for m2 in bb.legal_moves:
            if m2.from_square == mv.to_square:
                bbb = bb.copy(); bbb.push(m2)
                if not (bbb.is_attacked_by(them, m2.to_square) and not bbb.is_attacked_by(us, m2.to_square)):
                    return False
        return True

    # One-ply mate / big-loss filters
    def _opponent_has_mate_in_one(self, b: chess.Board) -> bool:
        """True if opponent can checkmate immediately."""
        for mv in b.legal_moves:
            bb = b.copy(); bb.push(mv)
            if bb.is_checkmate():
                return True
        return False
    
    def _is_ping_pong(self, b: chess.Board, mv: chess.Move) -> bool:
        """
        True if this move exactly undoes our previous own move (A→B then B→A).
        (Quiet return moves are the main repetition culprit.)
        """
        last = self._last_own_moves(b, 1)
        if not last:
            return False
        prev = last[0]
        return (prev.from_square == mv.to_square and prev.to_square == mv.from_square)
    
    def _is_bishop_ping_pong(self, b: chess.Board, mv: chess.Move) -> bool:
        """
        Bishop-specific ping-pong, only for non-forcing moves so tactical retreats are allowed.
        Blocks loops like Be3↔Bd2 or Bg5↔Bh4 when they don’t capture/give check/reduce debt.
        """
        p = b.piece_at(mv.from_square)
        if not p or p.piece_type != chess.BISHOP:
            return False
        if b.is_capture(mv) or b.gives_check(mv):
            return False
        if self._reduces_dev_debt(b, mv):
            return False
        return self._is_ping_pong(b, mv)

    def _opponent_wins_big_in_one(self, b_after_our_move: chess.Board, loss_cp: int = 450) -> bool:
        """True if opponent has a forcing reply gaining ≈ minor piece or more."""
        worst = -10_000
        for mv in b_after_our_move.legal_moves:
            is_cap = b_after_our_move.is_capture(mv)
            is_check = b_after_our_move.gives_check(mv)
            if not (is_cap or is_check):
                continue

            gain = 0

            if is_cap:
                # SEE for the capture (non-negative only)
                gain = max(0, self._see_legal_chain(b_after_our_move, mv))

                # Early centre-pawn bonus on captures (incl. en passant)
                to = mv.to_square
                is_ep = b_after_our_move.is_en_passant(mv)
                if is_ep:
                    is_pawn_capture = True
                    file_ = chess.square_file(to)  # EP target square shares the captured pawn's file
                else:
                    victim = b_after_our_move.piece_at(to)
                    is_pawn_capture = (victim is not None and victim.piece_type == chess.PAWN)
                    file_ = chess.square_file(to)

                if is_pawn_capture and b_after_our_move.fullmove_number <= 12 and file_ in (3, 4):
                    gain += 120

            if is_check:
                gain += 60

            worst = max(worst, gain)
            if worst >= loss_cp:
                return True

        return False


    def _is_move_safe_for_one_ply(self, b: chess.Board, mv: chess.Move, loss_cp: int = 450) -> bool:
        """Move is safe if it avoids mate-in-one and big immediate loss."""
        bb = b.copy()
        try:
            bb.push(mv)
        except Exception:
            return False
        if self._opponent_has_mate_in_one(bb):
            return False
        if self._opponent_wins_big_in_one(bb, loss_cp=loss_cp):
            return False
        if self._trap_loss_in_two(b, mv, min_loss_cp=loss_cp):
            return False
        return True

    def _filter_moves_avoiding_mate_in_one(self, b: chess.Board, moves: List[chess.Move]) -> List[chess.Move]:
        """Keep moves that do not allow a mate-in-one reply."""
        safe = []
        for mv in moves:
            bb = b.copy(); bb.push(mv)
            if not self._opponent_has_mate_in_one(bb):
                safe.append(mv)
        return safe

    def _filter_moves_avoiding_big_loss(self, b: chess.Board, moves: List[chess.Move], loss_cp: int = 450) -> List[chess.Move]:
        """Keep moves that do not allow a huge immediate material swing."""
        safe = []
        for mv in moves:
            bb = b.copy(); bb.push(mv)
            if not self._opponent_wins_big_in_one(bb, loss_cp=loss_cp):
                safe.append(mv)
        return safe

    # Immediate material-loss veto (capture-chain based)
    def _loses_material_immediately(self, b: chess.Board, mv: chess.Move, min_loss_cp: int = 240) -> bool:
        """True if opponent can snap material worth ≥ min_loss_cp right after our move."""
        bb = b.copy()
        try:
            bb.push(mv)
        except Exception:
            return True
        them = bb.turn
        worst_gain = -10_000
        for omv in bb.legal_moves:
            if not bb.is_capture(omv):
                continue
            gain = self._see_legal_chain(bb, omv)
            worst_gain = max(worst_gain, gain)
            if omv.to_square == mv.to_square and gain >= min_loss_cp:
                return True
        return worst_gain >= min_loss_cp
    
    def _early_queen_overextension(self, b: chess.Board, mv: chess.Move) -> bool:
        p = b.piece_at(mv.from_square)
        if not p or p.piece_type != chess.QUEEN or b.fullmove_number > 12:
            return False
        if b.is_capture(mv) or b.gives_check(mv):
            return False
        # entering enemy half early, or a big lateral leap, is suspicious
        r = chess.square_rank(mv.to_square)
        enemy_half = (r >= 4) if b.turn == chess.WHITE else (r <= 3)
        if enemy_half:
            return True
        f0 = chess.square_file(mv.from_square)
        f1 = chess.square_file(mv.to_square)
        return abs(f1 - f0) >= 3
    
    def _creates_strong_pin(self, b: chess.Board, mv: chess.Move) -> bool:
        """Return True if the bishop move creates a king pin on a knight immediately."""
        pc = b.piece_at(mv.from_square)
        if not pc or pc.piece_type != chess.BISHOP:
            return False
        bb = b.copy()
        try:
            bb.push(mv)
        except Exception:
            return False
        them = not b.turn
        # Pins to the king only (python-chess limitation)
        for sq, p2 in bb.piece_map().items():
            if p2.color == them and p2.piece_type == chess.KNIGHT and bb.is_pinned(them, sq):
                return True
        return False

    # Opening sanity checks
    def _fails_opening_sanity(self, b: chess.Board, mv: chess.Move) -> bool:
        """Early rules: avoid queen/rook drifts, king walks, and lazy flank pawn moves."""
        p = b.piece_at(mv.from_square)
        if not p:
            return False

        early10 = b.fullmove_number <= 10

        if early10:
            if self._early_ng5_without_threat(b, mv):
                return True
            if self._bishop_into_g_pawn_snap(b, mv):
                return True

            if p.piece_type == chess.BISHOP and b.gives_check(mv) and not b.is_capture(mv):
                if not self._creates_strong_pin(b, mv):
                    return True

            if p.piece_type == chess.QUEEN and b.gives_check(mv):
                f = chess.square_file(mv.to_square); r = chess.square_rank(mv.to_square)
                on_rim = (f in (0, 7)) or (r in (0, 7))
                if on_rim:
                    bb = b.copy(); bb.push(mv)
                    if any((not bb.is_capture(m2)) and (bb.piece_at(m2.from_square).piece_type != chess.KING)
                        for m2 in bb.legal_moves):
                        return True

            # Forcing moves are allowed; otherwise be strict about early wandering
            if b.gives_check(mv) or b.is_capture(mv) or (p.piece_type == chess.KING and b.is_castling(mv)):
                pass
            else:
                if self._early_queen_overextension(b, mv):
                    return True
                if p.piece_type == chess.KING and not b.is_castling(mv):
                    return True
                if p.piece_type in (chess.ROOK, chess.QUEEN):
                    return True
                if p.piece_type == chess.PAWN and chess.square_file(mv.from_square) in (0, 7):
                    return True
                if p.piece_type == chess.KNIGHT:
                    f = chess.square_file(mv.to_square)
                    if f in (0, 7):
                        return True

        # Knights to the rim discouraged through move 20
        if b.fullmove_number <= 20 and p.piece_type == chess.KNIGHT:
            f = chess.square_file(mv.to_square); r = chess.square_rank(mv.to_square)
            if f in (0, 7) or r in (0, 7):
                return True

        return False



    # Tactical/forcing checks
    def _find_critical_threat(self, b: chess.Board) -> Optional[Tuple[int, List[int]]]:
        """Return (our threatened square, attacker squares) if we are losing something valuable."""
        us, them = b.turn, not b.turn
        worst = None
        for sq in chess.SquareSet(b.occupied_co[us]):
            p = b.piece_at(sq)
            if not p:
                continue
            attackers = list(b.attackers(them, sq))
            if not attackers:
                continue
            best_opp_gain = max((self._see_legal_chain(b, chess.Move(frm, sq)) for frm in attackers), default=-10_000)
            if best_opp_gain > 0:
                val = VAL.get(p.piece_type, 0)
                if worst is None or val > worst[0]:
                    worst = (val, sq, attackers)
        if worst is None:
            return None
        return (worst[1], worst[2])

    def _threat_response(self, b: chess.Board, threat_sq: int, attackers_from: List[int]) -> Optional[chess.Move]:
        """Try safe capture of the attacker, else safe escape to a central-ish square."""
        us, them = b.turn, not b.turn

        best_cap_gain, best_cap = -10_000, None
        for mv in b.legal_moves:
            if mv.to_square not in attackers_from:
                continue
            gain = self._see_legal_chain(b, mv)
            if gain >= 0:
                bb = b.copy(); bb.push(mv)
                if not (bb.is_attacked_by(them, mv.to_square) and not bb.is_attacked_by(us, mv.to_square)):
                    if gain > best_cap_gain:
                        best_cap_gain, best_cap = gain, mv
        if best_cap:
            return best_cap

        best_escape, best_score = None, -1e9
        for mv in b.legal_moves:
            if mv.from_square != threat_sq:
                continue
            bb = b.copy()
            try:
                bb.push(mv)
            except Exception:
                continue
            safe = not bb.is_attacked_by(them, mv.to_square)
            if not safe:
                opp_caps = [m for m in bb.legal_moves if m.to_square == mv.to_square and bb.is_capture(m)]
                worst_opp = max((self._see_legal_chain(bb, m) for m in opp_caps), default=0)
                safe = worst_opp <= 0
            if not safe:
                continue
            f = chess.square_file(mv.to_square); r = chess.square_rank(mv.to_square)
            centre = 2 - (abs(3.5 - f) + abs(3.5 - r)) * 0.25
            score = 1.0 + centre
            if score > best_score:
                best_score, best_escape = score, mv
        return best_escape

    def _opponent_forcing_refutation_score(self, bb: chess.Board) -> int:
        """Approximate how many forcing resources opponent has after our move."""
        them = bb.turn
        us = not them
        best = 0
        for mv in list(bb.legal_moves):
            score = 0
            is_cap = bb.is_capture(mv)
            is_check = bb.gives_check(mv)
            pc = bb.piece_at(mv.from_square)

            if is_cap:
                score += 100 + max(0, self._see_legal_chain(bb, mv))
            if is_check:
                score += 220

            bbb = bb.copy(); bbb.push(mv)

            if bbb.is_checkmate():
                best = max(best, score + 10_000)
                continue

            if pc and pc.piece_type == chess.QUEEN and is_cap:
                q_sq = mv.to_square

                def _rook_corner_for(us_color: bool, q_target: int) -> int:
                    if us_color == chess.WHITE:
                        if q_target == chess.G2: return chess.H1
                        if q_target == chess.B2: return chess.A1
                    else:
                        if q_target == chess.G7: return chess.H8
                        if q_target == chess.B7: return chess.A8
                    return -1

                if mv.to_square in (chess.G2, chess.G7, chess.B2, chess.B7):
                    rook_corner = _rook_corner_for(us, mv.to_square)
                    rescue = False
                    for m2 in bbb.legal_moves:
                        if m2.to_square == q_sq and bbb.is_capture(m2): rescue = True; break
                        if rook_corner != -1 and m2.from_square == rook_corner: rescue = True; break
                    if not rescue:
                        score += 900

            if pc and pc.piece_type == chess.PAWN and not is_cap:
                rank = chess.square_rank(mv.to_square)
                dist = rank if them == chess.BLACK else (7 - rank)
                if dist <= 1:
                    score += 1200
                elif dist == 2:
                    score += 550

            ksq = bbb.king(us)
            if pc and pc.piece_type == chess.QUEEN and is_check and ksq is not None:
                f = chess.square_file(ksq)
                r = chess.square_rank(ksq)
                if f in (2, 3, 4) and 1 <= r <= 5:
                    qf = chess.square_file(mv.to_square)
                    qr = chess.square_rank(mv.to_square)
                    if qr in (3, 4, 5) or qf in (1, 6):
                        score += 600

            best = max(best, score)

        return best

    # Simple tactics and book gates
    def _free_capture(self, b: chess.Board) -> Optional[chess.Move]:
        """Pick a safe capture with best gain if available."""
        best_gain, best = -10_000, None
        for mv in b.legal_moves:
            if not b.is_capture(mv):
                continue
            gain = self._see_legal_chain(b, mv)
            if gain < 0:
                continue
            bb = b.copy(); bb.push(mv)
            if bb.is_attacked_by(not b.turn, mv.to_square) and not bb.is_attacked_by(b.turn, mv.to_square):
                continue
            if gain > best_gain:
                best_gain, best = gain, mv
        return best

    def _book_move(self, b: chess.Board) -> str:
        """Random micro-book move (only in the opening phase), filtered to legal."""
        if b.fullmove_number > 20:  # allow up to 20 full moves of book
            return ""
        cand = self._book.get(_pos_key(b), [])
        if not cand:
            return ""
        legals: List[str] = []
        for u in cand:
            try:
                m = chess.Move.from_uci(u)
                if m in b.legal_moves:
                    legals.append(u)
            except Exception:
                continue
        return random.choice(legals) if legals else ""
    
    def _best_opp_gain_on_square(self, b: chess.Board, target_sq: int) -> int:
        best = 0
        for mv in b.legal_moves:
            if mv.to_square == target_sq and b.is_capture(mv):
                best = max(best, max(0, self._see_legal_chain(b, mv)))
        return best
    
    def _has_safe_escape_for_piece(self, b: chess.Board, sq: int, color: bool, min_loss_cp: int) -> bool:
        us, them = color, (not color)
        piece = b.piece_at(sq)
        if piece is None:
            return True  # already gone (someone captured earlier)
        # 1) Try moving the piece away to a square that isn’t losing on SEE
        for mv in b.legal_moves:
            if mv.from_square != sq:
                continue
            b2 = b.copy()
            try:
                b2.push(mv)
            except Exception:
                continue
            to2 = mv.to_square
            # safe if not attacked OR opp’s best capture on that square doesn’t win
            if (not b2.is_attacked_by(them, to2)) or (self._best_opp_gain_on_square(b2, to2) < min_loss_cp):
                return True
        # 2) Try parry by capturing (any attacker of sq) with non-losing SEE
        atk_from = list(b.attackers(them, sq))
        if atk_from:
            for mv in b.legal_moves:
                if mv.to_square in atk_from and b.is_capture(mv):
                    if self._see_legal_chain(b, mv) >= 0:
                        return True
        return False
    
    def _trap_loss_in_two(self, b: chess.Board, mv: chess.Move, min_loss_cp: int = 450) -> bool:
        """After our move, allow *quiet* replies that create a winning capture on our moved piece next turn."""
        piece = b.piece_at(mv.from_square)
        if piece is None:
            return False
        piece_val = VAL.get(piece.piece_type, 0)

        bb = b.copy()
        try:
            bb.push(mv)
        except Exception:
            return True  # illegal anyway

        to = mv.to_square
        them = bb.turn  # opponent to move

        # If we already blundered immediately, the one-ply checks will catch it.
        # Here we allow quiet replies that *set up* a win next move.
        for omv in bb.legal_moves:
            b2 = bb.copy(); b2.push(omv)

            # if our moved piece was captured *now*, that’s one-ply and handled elsewhere
            still_there = (b2.piece_at(to) is not None) and (b2.piece_at(to).color != them)
            if not still_there:
                continue

            # If after their quiet move our piece is attacked and we have no safe save, treat as a trap.
            if b2.is_attacked_by(them, to):
                if not self._has_safe_escape_for_piece(b2, to, color=(not them), min_loss_cp=min_loss_cp):
                    # count it as losing at least the piece's value
                    return piece_val >= min_loss_cp
        return False

    # Root-scoring and fallback
    def _root_score(self, b: chess.Board, mv: chess.Move, visits: float) -> Tuple[bool, float]:
        """Score candidate at root using simple features; return (hard_block, score)."""
        hard = False
        s = float(visits)
        p = b.piece_at(mv.from_square)
        early = b.fullmove_number <= 12

        # discourage purposeless re-moves of the same piece
        chain = self._own_move_chain_len(b, mv, max_lookback=4)
        if chain >= 2 and not (b.is_capture(mv) or b.gives_check(mv) or self._reduces_dev_debt(b, mv)):
            s -= 6.0 * chain

        if self._is_bishop_ping_pong(b, mv):
            s -= 9999.0
            hard = True

        if self._is_ping_pong(b, mv) and not (b.is_capture(mv) or b.gives_check(mv)):
            s -= 3.0

        # Nudge toward classical opening moves in the first few plies
        if b.fullmove_number <= 6:
            if b.move_stack:
                last = b.peek()
                if last.to_square == mv.from_square and b.fullmove_number <= 12:
                    if not (b.is_capture(mv) or b.gives_check(mv)):
                        s -= 1.6
            if p:
                to = mv.to_square
                if p.piece_type == chess.PAWN and to in (chess.E4, chess.D4, chess.C4):
                    s += 0.8
                if p.piece_type == chess.KNIGHT and to in (chess.F3, chess.C3):
                    s += 0.6

        # Early queen overextension → hard veto
        if self._early_queen_overextension(b, mv):
            s -= 9999.0; hard = True

        # Tactics sweeteners
        if b.is_capture(mv): s += 1.6 + 0.01 * max(-300, min(300, self._see_legal_chain(b, mv)))
        if b.gives_check(mv): s += 0.5
        if mv.promotion: s += 1.2

        # Early non-forcing queen/rook drifts
        if p and p.piece_type == chess.QUEEN and early and not (b.is_capture(mv) or b.gives_check(mv)):
            s -= 4.8
        # Early king walks or unsafe king moves
        if p and p.piece_type == chess.KING and not b.is_castling(mv) and b.fullmove_number <= 20:
            s -= 7.0; hard = True
        if self._unsafe_king_walk(b, mv):
            s -= 9999.0; hard = True

        # Development and castling bonuses
        if self._reduces_dev_debt(b, mv): s += 1.4
        if p and p.piece_type == chess.KING and b.is_castling(mv): s += 2.0

        # Knights to the rim early (and bishop snap motif)
        if p and p.piece_type == chess.KNIGHT and early and chess.square_file(mv.to_square) in (0, 7):
            s -= 2.6
            if self._invites_bishop_snap(b, mv): s -= 3.2

        # Early passive rooks
        if p and p.piece_type == chess.ROOK and early and not b.is_capture(mv) and not b.gives_check(mv):
            s -= 2.8

        # Early f-pawn loosening without force
        if p and p.piece_type == chess.PAWN and early and mv.from_square == (chess.F2 if b.turn == chess.WHITE else chess.F7):
            if not (b.is_capture(mv) or b.gives_check(mv)):
                s -= 2.2

        # Qc2/Qb3 knight-tactic risk
        if self._qc2_b3_knight_tactic_risk(b, mv):
            s -= 4.0; hard = True

        # Don’t hang
        hang_val = self._move_hangs_value(b, mv)
        if hang_val > 0:
            if p and p.piece_type == chess.QUEEN:
                s -= 9999.0; hard = True
            else:
                s -= min(7.5, 0.012 * hang_val)

        # Landing square loose to opponent
        bb = b.copy(); bb.push(mv)
        to = mv.to_square
        if bb.is_attacked_by(not b.turn, to) and not bb.is_attacked_by(b.turn, to):
            s -= 2.8

        # Opponent forcing refutation capacity
        ref = self._opponent_forcing_refutation_score(bb)
        if ref >= 350:
            s -= 5.2; hard = True
        else:
            s -= 0.01 * ref

        # Pawn structure nudge
        s += 0.01 * self._pawn_push_bonus(b, mv)

        # Anti-repetition memory
        seen = self._seen.get(_rep_key(bb), 0)
        if seen >= 2: hard = True
        elif seen == 1: s -= 1.1

        if b.fullmove_number <= 15:
            if self._has_any_castling_rights(b, b.turn):
                if not self._has_any_castling_rights(bb, b.turn) and not b.is_castling(mv):
                    s -= 1.8

        return hard, s

    
    def _last_own_moves(self, b: chess.Board, k: int) -> List[chess.Move]:
        """
        Return the last k moves played by the side to move (i.e., 'our' last moves).
        If it's our turn now, our most-recent move is 2 plies ago.
        """
        stack = b.move_stack
        res: List[chess.Move] = []
        idx = len(stack) - 2  # last move by our side
        while idx >= 0 and len(res) < k:
            res.append(stack[idx])
            idx -= 2
        return res
    
    def _emergency_sane_move(self, b: chess.Board) -> str:
        """Pick a sane emergency move instead of 'first legal'."""
        try_list = [
            "e2e4","d2d4","c2c4","e2e3","d2d3","c2c3",  # centre pawns
            "g1f3","b1c3"                               # natural knights
        ]
        cand = []
        for u in try_list:
            try:
                m = chess.Move.from_uci(u)
                if m in b.legal_moves \
                and self._is_move_safe_for_one_ply(b, m, loss_cp=450) \
                and not self._loses_material_immediately(b, m, min_loss_cp=240) \
                and self._move_hangs_value(b, m) == 0 \
                and not (b.fullmove_number <= 10 and self._fails_opening_sanity(b, m)):
                    cand.append(m)
            except Exception:
                pass

        # Minor development & castling as next best
        for m in b.legal_moves:
            p = b.piece_at(m.from_square)
            if p and (self._reduces_dev_debt(b, m) or (p.piece_type == chess.KING and b.is_castling(m))):
                if self._is_move_safe_for_one_ply(b, m, loss_cp=450) \
                and not self._loses_material_immediately(b, m, min_loss_cp=240) \
                and self._move_hangs_value(b, m) == 0 \
                and not (b.fullmove_number <= 10 and self._fails_opening_sanity(b, m)):
                    cand.append(m)

        if cand:
            return cand[0].uci()

        for m in b.legal_moves:
            if not (b.fullmove_number <= 10 and self._fails_opening_sanity(b, m)):
                return m.uci()
        return next(iter(b.legal_moves)).uci()

    def _pick_sane_move(self, b: chess.Board) -> str:
        """Heuristic move choice when MCTS is not trusted."""
        legal = list(b.legal_moves)
        if not legal:
            return ""
        best_s, best_mv = -1e18, None
        us, them = b.turn, not b.turn
        early = b.fullmove_number <= 12
        endgame = self._is_endgame(b)
        debt_now = self._dev_debt(b)

        filtered = []
        for mv in legal:
            if self._bishop_into_g_pawn_snap(b, mv):
                continue
            if b.fullmove_number <= 10 and self._fails_opening_sanity(b, mv):
                continue
            if self._unsafe_king_walk(b, mv):
                continue
            if self._forbid_flank_pawn_while_undeveloped(b, mv):
                continue
            if self._loses_material_immediately(b, mv, min_loss_cp=240):
                continue
            if self._corner_rook_bishop_trap(b, mv):
                continue
            if self._early_queen_overextension(b, mv):
                continue
            if self._is_bishop_ping_pong(b, mv):
                continue
            if self._is_ping_pong(b, mv) and not (b.is_capture(mv) or b.gives_check(mv)):
                continue
            # discourage purposeless re-moves by the same piece
            chain = self._own_move_chain_len(b, mv, max_lookback=4)
            if chain >= 2 and not b.is_capture(mv) and not b.gives_check(mv) and not self._reduces_dev_debt(b, mv):
                continue

            filtered.append(mv)
        if not filtered:
            filtered = legal

        safe_pawn_push_exists = False
        for mv in filtered:
            pc = b.piece_at(mv.from_square)
            if pc and pc.piece_type == chess.PAWN:
                bonus = self._pawn_push_bonus(b, mv)
                if bonus >= 120:
                    bb = b.copy(); bb.push(mv)
                    if not (bb.is_attacked_by(them, mv.to_square) and not bb.is_attacked_by(us, mv.to_square)):
                        safe_pawn_push_exists = True
                        break

        for mv in filtered:
            s = 0.0
            p = b.piece_at(mv.from_square)
            bb = b.copy()
            try:
                bb.push(mv)
            except Exception:
                continue

            if self._loses_material_immediately(b, mv, min_loss_cp=240):
                continue

            if self._move_hangs_value(b, mv) > 0:
                continue

            if p and p.piece_type == chess.QUEEN:
                if bb.is_attacked_by(them, mv.to_square) and not bb.is_attacked_by(us, mv.to_square):
                    s = -1e9
                    continue

            if b.is_capture(mv): s += 800 + 0.01 * self._see_legal_chain(b, mv)
            if b.gives_check(mv): s += 120
            if mv.promotion: s += 300

            if self._reduces_dev_debt(b, mv): s += 160

            if p and p.piece_type == chess.QUEEN and early and not b.is_capture(mv):
                s -= 220 + 60 * max(0, debt_now)

            if p and p.piece_type == chess.KING and b.is_castling(mv): s += 520
            if p and p.piece_type == chess.KING and not b.is_castling(mv):
                if b.fullmove_number <= 20:
                    s -= 900
                elif endgame:
                    if safe_pawn_push_exists:
                        s -= 260
                    f = chess.square_file(mv.to_square); r = chess.square_rank(mv.to_square)
                    s += 20 - 3 * (abs(3.5 - f) + abs(3.5 - r))

            if p and p.piece_type == chess.ROOK and early and not b.is_capture(mv) and not b.gives_check(mv):
                s -= 200

            if (p and p.piece_type == chess.KNIGHT and early and b.is_capture(mv)):
                cap = b.piece_at(mv.to_square)
                f = chess.square_file(mv.to_square)
                if cap and cap.piece_type == chess.PAWN and f in (0, 1, 6, 7) and not b.gives_check(mv):
                    s -= 260

            if p and p.piece_type == chess.KNIGHT and b.fullmove_number <= 20 and chess.square_file(mv.to_square) in (0, 7):
                if self._invites_bishop_snap(b, mv):
                    s -= 320
            
            if b.fullmove_number <= 6 and p:
                to = mv.to_square
                if p.piece_type == chess.PAWN and to in (chess.E4, chess.D4, chess.C4):
                    s += 120
                if p.piece_type == chess.KNIGHT and to in (chess.F3, chess.C3):
                    s += 90

            to = mv.to_square

            if any(bb.is_castling(m2) for m2 in bb.legal_moves): s += 80

            if bb.is_attacked_by(them, to) and not bb.is_attacked_by(us, to):
                s -= 200
                if p:
                    s -= 0.8 * (VAL.get(p.piece_type, 0) / 100.0)

            f = chess.square_file(to); r = chess.square_rank(to)
            s += 20 - 3 * (abs(3.5 - f) + abs(3.5 - r))

            s += min(40, len(list(bb.legal_moves)))

            if self._dev_debt(bb) < debt_now: s += 50

            qga_pawn_on_c4 = any(sq == chess.C4 and pc2.piece_type == chess.PAWN and pc2.color == them
                                for sq, pc2 in b.piece_map().items())
            if qga_pawn_on_c4 and to == chess.C4 and b.is_capture(mv):
                s += 220

            if b.fullmove_number <= 12 and debt_now > 0 and not self._reduces_dev_debt(b, mv) and not b.is_capture(mv) and not b.gives_check(mv):
                s -= 80

            ref = self._opponent_forcing_refutation_score(bb)
            s -= 0.02 * ref
            if ref >= 350: s -= 1200

            s += self._pawn_push_bonus(b, mv)

            if b.fullmove_number <= 15:
                if self._has_any_castling_rights(b, b.turn) and not self._has_any_castling_rights(bb, b.turn) and not b.is_castling(mv):
                    s -= 180

            if self._needs_luft(b):
                if p and p.piece_type == chess.PAWN:
                    target_sqs = (chess.H3, chess.G3) if b.turn == chess.WHITE else (chess.H6, chess.G6)
                    if mv.to_square in target_sqs and not b.is_capture(mv):
                        if not (bb.is_attacked_by(them, mv.to_square) and not bb.is_attacked_by(us, mv.to_square)):
                            s += 140

            if s > best_s:
                best_s, best_mv = s, mv

        return best_mv.uci() if best_mv else (list(b.legal_moves)[0].uci() if legal else "")

    
    def _own_move_chain_len(self, b: chess.Board, mv: chess.Move, max_lookback: int = 4) -> int:
        """How many consecutive last own moves involve this same piece continuing its path?"""
        chain = 0
        cur_to = mv.from_square
        for m in self._last_own_moves(b, max_lookback):
            if m.to_square == cur_to:
                chain += 1
                cur_to = m.from_square
            else:
                break
        return chain

    # Main API
    @torch.no_grad()
    def best_move_uci(self, fen: str) -> str:
        """Return a safe UCI move using (book/tactics) → MCTS → fallback."""
        b = chess.Board(fen)

        def _safe_alternatives(bd: chess.Board) -> List[chess.Move]:
            """Generate a list of safe-ish legal moves."""
            leg = list(bd.legal_moves)
            if not leg:
                return []
            if bd.fullmove_number <= 10:
                leg = [m for m in leg if not self._fails_opening_sanity(bd, m)]
            leg = [m for m in leg if not self._forbid_flank_pawn_while_undeveloped(bd, m)]
            leg = [m for m in leg if not self._unsafe_king_walk(bd, m)]
            leg = [m for m in leg if not self._loses_material_immediately(bd, m, min_loss_cp=240)]
            leg = self._filter_moves_avoiding_mate_in_one(bd, leg)
            leg = self._filter_moves_avoiding_big_loss(bd, leg, loss_cp=450) or leg
            leg = [m for m in leg if not self._corner_rook_bishop_trap(bd, m)]
            return leg

        def _return(uci: str) -> str:
            """
            Final gate before returning a move.
            Ensures: legal, no mate-in-one blunder, no huge immediate loss,
            no unsafe king walk, no early flank-pawn silliness,
            and no hanging the moved piece.
            """
            try:
                m = chess.Move.from_uci(uci)
                if m in b.legal_moves:
                    if (self._is_move_safe_for_one_ply(b, m, loss_cp=450)
                        and not self._unsafe_king_walk(b, m)
                        and not (b.fullmove_number <= 10 and self._fails_opening_sanity(b, m))
                        and not self._forbid_flank_pawn_while_undeveloped(b, m)
                        and not self._loses_material_immediately(b, m, min_loss_cp=240)
                        and self._move_hangs_value(b, m) == 0
                        and not self._corner_rook_bishop_trap(b, m)):
                        bb = b.copy(); bb.push(m)
                        self._seen[_rep_key(bb)] = self._seen.get(_rep_key(bb), 0) + 1
                        return uci
                    # try safe alternatives if the chosen move fails a gate
                    alts = _safe_alternatives(b)
                    if alts:
                        for m2 in alts:
                            if self._move_hangs_value(b, m2) == 0:
                                bb = b.copy(); bb.push(m2)
                                self._seen[_rep_key(bb)] = self._seen.get(_rep_key(bb), 0) + 1
                                return m2.uci()
            except Exception:
                pass

            # fallback through heuristic safe move(s)
            safe = self._pick_sane_move(b)
            try:
                m2 = chess.Move.from_uci(safe)
                if m2 in b.legal_moves:
                    # if the heuristic pick also fails safety, try alternatives
                    if (not self._is_move_safe_for_one_ply(b, m2, loss_cp=450)
                        or self._unsafe_king_walk(b, m2)
                        or (b.fullmove_number <= 10 and self._fails_opening_sanity(b, m2))
                        or self._forbid_flank_pawn_while_undeveloped(b, m2)
                        or self._loses_material_immediately(b, m2, min_loss_cp=240)
                        or self._move_hangs_value(b, m2) > 0
                        or self._corner_rook_bishop_trap(b, m2)):
                        alts = _safe_alternatives(b)
                        if alts:
                            for m3 in alts:
                                if self._move_hangs_value(b, m3) == 0:
                                    bb = b.copy(); bb.push(m3)
                                    self._seen[_rep_key(bb)] = self._seen.get(_rep_key(bb), 0) + 1
                                    return m3.uci()
                    bb = b.copy(); bb.push(m2)
                    if self._move_hangs_value(b, m2) == 0:
                        self._seen[_rep_key(bb)] = self._seen.get(_rep_key(bb), 0) + 1
                        return safe
            except Exception:
                pass

            return self._emergency_sane_move(b)

        legal = list(b.legal_moves)
        if not legal:
            return ""

        # pre-filter obvious issues
        pre = legal
        if b.fullmove_number <= 10:
            pre = [m for m in pre if not self._fails_opening_sanity(b, m)]
        pre = [m for m in pre if not self._forbid_flank_pawn_while_undeveloped(b, m)]
        pre = [m for m in pre if not self._unsafe_king_walk(b, m)]
        pre = [m for m in pre if not self._loses_material_immediately(b, m, min_loss_cp=240)]
        if not pre:
            pre = legal

        # micro-book if still safe
        u = self._book_move(b)
        if u:
            try:
                mv = chess.Move.from_uci(u)
                if mv in b.legal_moves \
                   and mv in pre \
                   and self._is_move_safe_for_one_ply(b, mv, loss_cp=450):
                    bb = b.copy(); bb.push(mv)
                    self._seen[_rep_key(bb)] = self._seen.get(_rep_key(bb), 0) + 1
                    return u
            except Exception:
                pass

        # one-ply safety filters
        safe_moves = self._filter_moves_avoiding_mate_in_one(b, pre)
        safe_moves2 = self._filter_moves_avoiding_big_loss(b, safe_moves, loss_cp=450)
        filtered_moves = safe_moves2 or safe_moves or pre

        # handle critical threats first
        crit = self._find_critical_threat(b)
        if crit is not None:
            sq, attackers = crit
            mv = self._threat_response(b, sq, attackers)
            if mv and mv in filtered_moves:
                return _return(mv.uci())

        # quick free capture if safe
        cap = self._free_capture(b)
        if cap is not None and cap in filtered_moves:
            if self._is_move_safe_for_one_ply(b, cap, loss_cp=450) and not self._corner_rook_bishop_trap(b, cap):
                return _return(cap.uci())

        # try MCTS
        try:
            root = self.mcts.run(b)
        except Exception:
            return _return(self._pick_sane_move(b))

        if not getattr(root, "children", None):
            return _return(self._pick_sane_move(b))

        phase_tight = b.fullmove_number <= 14 and not self._is_endgame(b)
        top_k = 4 if phase_tight else 6
        mv_vis = self._best_by_visits_with_safety(b, root, top_k=top_k)
        if mv_vis is not None and mv_vis in filtered_moves:
            return _return(mv_vis.uci())

        # tie-break with root heuristics
        scored, backed_off = [], []
        for mv, node in root.children.items():
            if mv not in filtered_moves:
                continue
            N = getattr(node, "N", 0.0)
            hard, score = self._root_score(b, mv, N)
            (backed_off if hard else scored).append((score, mv))
        pool = scored if scored else backed_off
        if not pool:
            return _return(self._pick_sane_move(b))
        pool.sort(key=lambda t: t[0], reverse=True)
        return _return(pool[0][1].uci())

    def _best_by_visits_with_safety(self, b: chess.Board, root, top_k: int = 6) -> Optional[chess.Move]:
        items = sorted(root.children.items(), key=lambda kv: getattr(kv[1], "N", 0), reverse=True)
        if not items:
            return None
        early = b.fullmove_number <= 10
        for mv, _ in items[:top_k]:
            # discourage purposeless re-moves by the same piece
            chain = self._own_move_chain_len(b, mv, max_lookback=4)
            if chain >= 2 and not b.is_capture(mv) and not b.gives_check(mv) and not self._reduces_dev_debt(b, mv):
                continue

            if self._bishop_into_g_pawn_snap(b, mv):
                continue
            if self._trap_loss_in_two(b, mv, min_loss_cp=240):
                continue
            if self._early_queen_overextension(b, mv):
                continue
            if early and self._fails_opening_sanity(b, mv):
                continue
            if self._unsafe_king_walk(b, mv):
                continue
            if self._loses_material_immediately(b, mv, min_loss_cp=240):
                continue
            if self._move_hangs_value(b, mv) > 0:
                continue
            if self._corner_rook_bishop_trap(b, mv):
                continue
            if self._is_bishop_ping_pong(b, mv):
                continue
            if self._is_ping_pong(b, mv) and not (b.is_capture(mv) or b.gives_check(mv)):
                continue

            bb = b.copy(); bb.push(mv)

            if bb.is_attacked_by(not b.turn, mv.to_square) and not bb.is_attacked_by(b.turn, mv.to_square):
                if self._see_legal_chain(b, mv) < 0:
                    continue

            # queen-specific post-move looseness
            pc = b.piece_at(mv.from_square)
            if pc and pc.piece_type == chess.QUEEN:
                if bb.is_attacked_by(not b.turn, mv.to_square) and not bb.is_attacked_by(b.turn, mv.to_square):
                    continue

            if bb.is_attacked_by(not b.turn, mv.to_square) and not bb.is_attacked_by(b.turn, mv.to_square):
                if self._see_legal_chain(b, mv) < 0:
                    continue

            if self._opponent_forcing_refutation_score(bb) >= 350:
                continue
            return mv
        return None


    # Phase/structure helpers
    def _phase_score(self, b: chess.Board) -> float:
        """Return a rough middlegame score based on remaining material."""
        material = 0
        for pc in b.piece_map().values():
            if pc.piece_type != chess.KING:
                material += VAL.get(pc.piece_type, 0)
        return min(1.0, material / 7840.0)

    def _is_endgame(self, b: chess.Board) -> bool:
        """True if low material (simple threshold)."""
        return self._phase_score(b) < 0.35

    def _is_passed_pawn(self, b: chess.Board, sq: int, color: bool) -> bool:
        """Check for a passed pawn ignoring minor edge cases."""
        f = chess.square_file(sq)
        r = chess.square_rank(sq)
        step = 1 if color == chess.WHITE else -1
        enemy = not color
        for df in (-1, 0, 1):
            ff = f + df
            if ff < 0 or ff > 7:
                continue
            rr = r + step
            while 0 <= rr <= 7:
                t = chess.square(ff, rr)
                p = b.piece_at(t)
                if p and p.color == enemy and p.piece_type == chess.PAWN:
                    return False
                rr += step
        return True

    def _move_hangs_value(self, b: chess.Board, mv: chess.Move) -> int:
        """Return value of the moved piece if it ends up hanging; else 0."""
        pc = b.piece_at(mv.from_square)
        if not pc:
            return 0
        bb = b.copy()
        try:
            bb.push(mv)
        except Exception:
            return VAL.get(pc.piece_type, 0)

        them = not b.turn
        to = mv.to_square
        if not bb.is_attacked_by(them, to):
            return 0
        piece_cp = VAL.get(pc.piece_type, 0)
        worst = self._best_opp_gain_on_square(bb, to)
        return piece_cp if worst >= int(0.8 * piece_cp) else 0

    def _pawn_push_bonus(self, b: chess.Board, mv: chess.Move) -> int:
        """Give points to safe pawn pushes, more if passed and closer to promotion."""
        p = b.piece_at(mv.from_square)
        if not p or p.piece_type != chess.PAWN:
            return 0
        bb = b.copy(); bb.push(mv)
        if bb.is_attacked_by(not b.turn, mv.to_square) and not bb.is_attacked_by(b.turn, mv.to_square):
            return 0
        rank = chess.square_rank(mv.to_square)
        dist = (7 - rank) if p.color == chess.WHITE else rank
        base = 60
        passed = self._is_passed_pawn(b, mv.from_square, p.color) or self._is_passed_pawn(bb, mv.to_square, p.color)
        if passed:
            base += 220
        if dist <= 1:
            base += 160
        elif dist == 2:
            base += 90
        if not b.is_capture(mv):
            base += 20
        return base

    def _has_any_castling_rights(self, b: chess.Board, color: bool) -> bool:
        """True if side still has either castling right."""
        try:
            return b.has_castling_rights(color)
        except Exception:
            if color == chess.WHITE:
                return b.has_kingside_castling_rights(chess.WHITE) or b.has_queenside_castling_rights(chess.WHITE)
            else:
                return b.has_kingside_castling_rights(chess.BLACK) or b.has_queenside_castling_rights(chess.BLACK)

    def _needs_luft(self, b: chess.Board) -> bool:
        ksq = b.king(b.turn)
        if ksq is None:
            return False
        if b.turn == chess.WHITE and ksq == chess.G1:
            return (b.piece_at(chess.H2) is not None) and (b.piece_at(chess.H3) is None) and (b.piece_at(chess.G3) is None)
        if b.turn == chess.BLACK and ksq == chess.G8:
            return (b.piece_at(chess.H7) is not None) and (b.piece_at(chess.H6) is None) and (b.piece_at(chess.G6) is None)
        return False

