from __future__ import annotations

import sys
import threading
import queue
from typing import Mapping, Tuple, Optional, List

import pygame
import chess
import chess.engine

from config import (
    BOARD_SIZE, WINDOW_HEIGHT, HUD_TOP, FPS, TILE,
    PTH_PATH, STOCKFISH_PATH, STOCKFISH_ELO_DEFAULT,
)
from chess_agent_rl import ChessRLAgent
from ui import (
    show_start_screen, ask_stockfish_elo,
    draw_board, draw_pieces, draw_banners,
    load_piece_images, xy_to_square,
    endgame_message, endgame_modal,
)

_PVAL = {
    chess.PAWN: 100, chess.KNIGHT: 300, chess.BISHOP: 310,
    chess.ROOK: 500, chess.QUEEN: 900, chess.KING: 10_000,
}

# Global AI state shared across screens
agent: ChessRLAgent | None = None
_ai_thread: threading.Thread | None = None
_ai_queue: "queue.Queue[Optional[chess.Move]]" = queue.Queue(maxsize=1)
_ai_fen_inflight: str | None = None
MOVE_GAP_MS = 1000


def _see_legal_chain_local(b: chess.Board, mv: chess.Move) -> int:
    """Evaluate the capture/recapture sequence on mv.to_square using only legal moves."""
    if not b.is_capture(mv):
        return 0

    to = mv.to_square
    bb = b.copy()

    # First victim (handle en passant)
    if bb.is_en_passant(mv):
        first_val = _PVAL[chess.PAWN]
    else:
        victim = bb.piece_at(to)
        first_val = 0 if victim is None else _PVAL[victim.piece_type]

    gains = [first_val]

    # Promotion delta on the initial move (e.g., PxX=Q adds over a pawn)
    if mv.promotion:
        gains[0] += _PVAL[mv.promotion] - _PVAL[chess.PAWN]

    bb.push(mv)

    # Always capture back on 'to' with the least valuable attacker
    while True:
        caps = [m for m in bb.legal_moves if m.to_square == to and bb.is_capture(m)]
        if not caps:
            break
        caps.sort(key=lambda m: _PVAL[bb.piece_at(m.from_square).piece_type])
        reply = caps[0]

        cur = bb.piece_at(to)
        cur_val = 0 if cur is None else _PVAL[cur.piece_type]
        gains.append(cur_val)

        if reply.promotion:
            gains[-1] += _PVAL[reply.promotion] - _PVAL[chess.PAWN]

        bb.push(reply)

    # Negamax fold (swap list propagation)
    for i in range(len(gains) - 2, -1, -1):
        gains[i] = max(gains[i], -gains[i + 1])

    return gains[0]

def _pick_sane_move_local(board: chess.Board) -> chess.Move:
    """Heuristic move: mate > checks > captures > develop/centre; avoid bad SEE."""
    legal: List[chess.Move] = list(board.legal_moves)
    if not legal:
        return chess.Move.null()

    scored: List[tuple[float, chess.Move]] = []
    fullmove = board.fullmove_number

    for mv in legal:
        sc = 0.0

        bb = board.copy(stack=False)
        bb.push(mv)
        if bb.is_checkmate():
            return mv
        
        if bb.is_attacked_by(not board.turn, mv.to_square) and not bb.is_attacked_by(board.turn, mv.to_square):
            continue

        if board.gives_check(mv):
            sc += 40.0

        if board.is_capture(mv):
            sc += 100.0 + 0.5 * max(-300, min(300, _see_legal_chain_local(board, mv)))

        p = board.piece_at(mv.from_square)
        if p:
            if p.piece_type in (chess.KNIGHT, chess.BISHOP):
                start_rank = 1 if p.color == chess.WHITE else 8
                r_from = chess.square_rank(mv.from_square) + 1
                if r_from == start_rank and fullmove <= 12:
                    sc += 35.0
            if p.piece_type == chess.PAWN:
                file_to = chess.square_file(mv.to_square)
                if file_to in (3, 4):
                    sc += 10.0
            if p.piece_type == chess.KING:
                if board.is_castling(mv):
                    sc += 60.0
                elif fullmove <= 15:
                    sc -= 80.0

        if _see_legal_chain_local(board, mv) < -50:
            sc -= 120.0

        scored.append((sc, mv))

    scored.sort(key=lambda t: t[0], reverse=True)
    return scored[0][1]


def _ai_worker(fen_snapshot: str) -> None:
    """Compute a move for the given FEN on a background thread and queue it."""
    assert agent is not None
    board = chess.Board(fen_snapshot)

    uci = agent.best_move_uci(fen_snapshot)
    if uci:
        cand = chess.Move.from_uci(uci)
        mv = cand if cand in board.legal_moves else _pick_sane_move_local(board)
    else:
        mv = _pick_sane_move_local(board)

    while not _ai_queue.empty():
        _ai_queue.get_nowait()
    _ai_queue.put(mv)


def start_ai_think(board: chess.Board) -> None:
    """Start the worker thread for the current FEN (skip if already thinking on this FEN)."""
    global _ai_thread, _ai_fen_inflight
    fen = board.fen()
    if _ai_thread and _ai_thread.is_alive() and _ai_fen_inflight == fen:
        return
    _ai_fen_inflight = fen

    while not _ai_queue.empty():
        _ai_queue.get_nowait()

    _ai_thread = threading.Thread(target=_ai_worker, args=(fen,), daemon=True)
    _ai_thread.start()


def poll_ai_move() -> Optional[chess.Move]:
    """Non-blocking read of AI move."""
    return _ai_queue.get_nowait() if not _ai_queue.empty() else None


def new_game() -> Tuple[chess.Board, chess.Color, chess.Color, bool, Optional[int]]:
    """Create a new game; randomise human colour; draw human at the bottom."""
    import random
    board = chess.Board()
    human = chess.WHITE if random.randrange(2) == 0 else chess.BLACK
    bot_colour = chess.BLACK if human == chess.WHITE else chess.WHITE
    white_bottom = (human == chess.WHITE)
    return board, human, bot_colour, white_bottom, None


def _cfg_stockfish_strength(engine: chess.engine.SimpleEngine, elo: int) -> None:
    """Configure Stockfish strength via Elo or skill."""
    opts = engine.options
    if "UCI_LimitStrength" in opts and "UCI_Elo" in opts:
        engine.configure({"UCI_LimitStrength": True, "UCI_Elo": int(elo)})
    elif "Skill Level" in opts:
        sf_min, sf_max = 1350, 2850
        skill = int(round((elo - sf_min) / max(1, (sf_max - sf_min) / 20)))
        engine.configure({"Skill Level": max(0, min(20, skill))})


def run_rl_vs_stockfish(
    screen: pygame.Surface,
    piece_imgs: Mapping[str, pygame.Surface],
    elo: int = STOCKFISH_ELO_DEFAULT,
) -> None:
    """Run RL (White) vs Stockfish (Black) with a fixed 1s gap between moves."""
    clock = pygame.time.Clock()
    last_move_ts = pygame.time.get_ticks() - MOVE_GAP_MS

    engine = chess.engine.SimpleEngine.popen_uci(str(STOCKFISH_PATH))
    _cfg_stockfish_strength(engine, elo)

    while True:
        board = chess.Board()
        white_bottom = True

        if board.turn == chess.WHITE:
            start_ai_think(board)

        while True:
            clock.tick(FPS)

            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    engine.quit()
                    pygame.quit()
                    sys.exit(0)
                if ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE:
                    engine.quit()
                    return

            can_act = (pygame.time.get_ticks() - last_move_ts) >= MOVE_GAP_MS

            if board.outcome() is None:
                if board.turn == chess.WHITE:
                    mv = poll_ai_move()
                    if can_act and mv is not None:
                        if mv == chess.Move.null() or mv not in board.legal_moves:
                            mv = _pick_sane_move_local(board)
                        board.push(mv)
                        last_move_ts = pygame.time.get_ticks()
                    elif can_act and ((_ai_thread is None) or (not _ai_thread.is_alive())):
                        mv = _pick_sane_move_local(board)
                        board.push(mv)
                        last_move_ts = pygame.time.get_ticks()
                else:
                    if can_act:
                        res = engine.play(board, chess.engine.Limit(time=0.05))
                        if res.move:
                            board.push(res.move)
                            last_move_ts = pygame.time.get_ticks()
                            if board.outcome() is None and board.turn == chess.WHITE:
                                start_ai_think(board)

            draw_board(screen, None, None, white_bottom, HUD_TOP)
            draw_pieces(screen, board, piece_imgs, white_bottom, HUD_TOP)
            draw_banners(screen, f"Stockfish (Elo {elo})", "GrandMasker's Chess Bot")
            pygame.display.flip()

            if board.outcome() is not None:
                msg = endgame_message(board.outcome())
                action = endgame_modal(
                    screen, board, piece_imgs, white_bottom,
                    top_label=f"Stockfish (Elo {elo})",
                    bottom_label="GrandMasker's Chess Bot",
                    result_text=msg,
                )
                if action == "play_again":
                    break
                engine.quit()
                return


def run_human_vs_rl(
    screen: pygame.Surface,
    piece_imgs: Mapping[str, pygame.Surface],
) -> None:
    """Run Human vs RL; human pieces at the bottom; show modal on game end."""
    clock = pygame.time.Clock()

    while True:
        board, human, bot, white_bottom, selected_sq = new_game()
        if board.turn == bot and board.outcome() is None:
            start_ai_think(board)

        while True:
            clock.tick(FPS)

            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit(0)
                if ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE:
                    return
                if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1 and board.outcome() is None:
                    mx, my = pygame.mouse.get_pos()
                    by = my - HUD_TOP
                    if 0 <= by < BOARD_SIZE:
                        sq = xy_to_square(mx // TILE, by // TILE, white_bottom)
                        if selected_sq is None:
                            piece = board.piece_at(sq)
                            if piece and piece.color == human and board.turn == human:
                                selected_sq = sq
                        else:
                            move = chess.Move(selected_sq, sq)
                            if move not in board.legal_moves:
                                for promo in (chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT):
                                    pm = chess.Move(selected_sq, sq, promotion=promo)
                                    if pm in board.legal_moves:
                                        move = pm
                                        break
                            if move in board.legal_moves and board.turn == human:
                                board.push(move)
                                selected_sq = None
                                if board.outcome() is None and board.turn == bot:
                                    start_ai_think(board)
                            else:
                                piece = board.piece_at(sq)
                                selected_sq = sq if (piece and piece.color == human and board.turn == human) else None

            if board.outcome() is None and board.turn == bot:
                mv = poll_ai_move()
                if mv is not None:
                    if mv == chess.Move.null() or mv not in board.legal_moves:
                        mv = _pick_sane_move_local(board)
                    board.push(mv)
                else:
                    if (_ai_thread is None) or (not _ai_thread.is_alive()):
                        mv = _pick_sane_move_local(board)
                        board.push(mv)

            legal_targets = [m.to_square for m in board.legal_moves if selected_sq is not None and m.from_square == selected_sq] or None
            draw_board(screen, selected_sq, legal_targets, white_bottom, HUD_TOP)
            draw_pieces(screen, board, piece_imgs, white_bottom, HUD_TOP)
            draw_banners(screen, "GrandMasker's Chess Bot", "You")
            pygame.display.flip()

            if board.outcome() is not None:
                msg = endgame_message(board.outcome())
                action = endgame_modal(
                    screen, board, piece_imgs, white_bottom,
                    top_label="GrandMasker's Chess Bot",
                    bottom_label="You",
                    result_text=msg,
                )
                if action == "play_again":
                    break
                return


def main() -> None:
    """Load RL agent, open window, show menu, run chosen mode."""
    global agent
    agent = ChessRLAgent(str(PTH_PATH), mcts_sims=512, value_white_pov=True, logit_temperature=0.5)

    pygame.init()
    screen = pygame.display.set_mode((BOARD_SIZE, WINDOW_HEIGHT))
    pygame.display.set_caption("GrandMasker's Chess Bot")
    piece_imgs = load_piece_images()

    choice = show_start_screen(screen)
    if choice == "exit":
        pygame.quit()
        return

    if choice == "rl_vs_sf":
        elo = ask_stockfish_elo(screen, initial=STOCKFISH_ELO_DEFAULT)
        if elo is None:
            return main()
        run_rl_vs_stockfish(screen, piece_imgs, elo=elo)
        return main()

    run_human_vs_rl(screen, piece_imgs)
    return main()


if __name__ == "__main__":
    main()