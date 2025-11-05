from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, List, Tuple, Dict

import pygame
import chess

from config import (
    TILE, BOARD_SIZE, LIGHT, DARK, HIGHLIGHT,
    HUD_TOP, HUD_BOTTOM, WINDOW_HEIGHT, PIECES_DIR,
    FALLBACK_TO_LETTERS, FPS,
    STOCKFISH_ELO_MIN, STOCKFISH_ELO_MAX,
)

def square_to_xy(sq: int, white_bottom: bool = True) -> Tuple[int, int]:
    """Convert a board square index (0–63) to (x, y) tile coordinates for drawing."""
    f, r = chess.square_file(sq), chess.square_rank(sq)
    # Flip ranks if White is at the bottom so '0,0' is the top-left visually.
    return (f, 7 - r) if white_bottom else (7 - f, r)

def xy_to_square(x: int, y: int, white_bottom: bool = True) -> int:
    """Convert (x, y) tile coordinates back to a board square index."""
    # Inverse of square_to_xy; used for mouse clicks → squares.
    return chess.square(x, 7 - y) if white_bottom else chess.square(7 - x, y)

def draw_board(
    screen: pygame.Surface,
    selected_sq: Optional[int] = None,
    legal_targets: Optional[List[int]] = None,
    white_bottom: bool = True,
    y_offset: int = 0,
) -> None:
    """Draw the 8×8 board, optional selection outline, and optional legal-move dots."""
    # Tiles
    for y in range(8):
        for x in range(8):
            colour = LIGHT if (x + y) % 2 == 0 else DARK
            pygame.draw.rect(screen, colour, (x * TILE, y_offset + y * TILE, TILE, TILE))

    # Selection outline
    if selected_sq is not None:
        fx, fy = square_to_xy(selected_sq, white_bottom)
        pygame.draw.rect(screen, HIGHLIGHT, (fx * TILE, y_offset + fy * TILE, TILE, TILE), 5)

    # Legal targets (small dots in the centre of destination tiles)
    if legal_targets:
        for sq in legal_targets:
            tx, ty = square_to_xy(sq, white_bottom)
            cx, cy = tx * TILE + TILE // 2, y_offset + ty * TILE + TILE // 2
            pygame.draw.circle(screen, (30, 30, 30), (cx, cy), 8)

def draw_banners(screen: pygame.Surface, top_text: str, bottom_text: str) -> None:
    """Draw simple black banners with white text above and below the board area."""
    font = pygame.font.SysFont("arial", 22, bold=True)

    # Top banner
    pygame.draw.rect(screen, (0, 0, 0), (0, 0, BOARD_SIZE, HUD_TOP))
    top_surf = font.render(top_text, True, (255, 255, 255))
    screen.blit(top_surf, top_surf.get_rect(center=(BOARD_SIZE // 2, HUD_TOP // 2)))

    # Bottom banner
    bottom_y = WINDOW_HEIGHT - HUD_BOTTOM
    pygame.draw.rect(screen, (0, 0, 0), (0, bottom_y, BOARD_SIZE, HUD_BOTTOM))
    bsurf = font.render(bottom_text, True, (255, 255, 255))
    screen.blit(bsurf, bsurf.get_rect(center=(BOARD_SIZE // 2, bottom_y + HUD_BOTTOM // 2)))

def legal_dests(board: chess.Board, from_sq: int) -> List[int]:
    """Return a list of legal destination squares from a given source square."""
    return [m.to_square for m in board.legal_moves if m.from_square == from_sq]

def load_piece_images() -> Dict[str, pygame.Surface]:
    """Load and scale piece sprites; return {} to fall back to letter rendering."""
    if FALLBACK_TO_LETTERS:
        return {}
    labels = {
        "wp": "wp.png", "wr": "wr.png", "wn": "wn.png", "wb": "wb.png", "wq": "wq.png", "wk": "wk.png",
        "bp": "bp.png", "br": "br.png", "bn": "bn.png", "bb": "bb.png", "bq": "bq.png", "bk": "bk.png",
    }
    base = Path(PIECES_DIR)
    images: Dict[str, pygame.Surface] = {}

    # Only proceed if all expected files exist (avoids half-missing sets).
    all_exist = all((base / fname).exists() for fname in labels.values())
    if not all_exist:
        return {}

    # Load and scale to the current TILE size.
    for k, fname in labels.items():
        img = pygame.image.load(str(base / fname)).convert_alpha()
        images[k] = pygame.transform.smoothscale(img, (TILE, TILE))

    # Require a complete set of 12 keys; else fall back to letters.
    return images if len(images) == 12 else {}

def draw_pieces(
    screen: pygame.Surface,
    board: chess.Board,
    piece_images: Dict[str, pygame.Surface],
    white_bottom: bool = True,
    y_offset: int = 0,
) -> None:
    """Draw pieces as sprites if available, otherwise render single-letter glyphs."""
    use_images = len(piece_images) == 12
    font = None if use_images else pygame.font.SysFont("arial", int(TILE * 0.8), bold=True)

    for sq in chess.SQUARES:
        p = board.piece_at(sq)
        if not p:
            continue
        x, y = square_to_xy(sq, white_bottom)
        if use_images:
            key = f"{'w' if p.color else 'b'}{p.symbol().lower()}"
            screen.blit(piece_images[key], (x * TILE, y_offset + y * TILE))
        else:
            # Letter fallback (upper-case = White, lower-case = Black)
            s = p.symbol()
            col = (245, 245, 245) if s.isupper() else (15, 15, 15)
            surf = font.render(s.upper(), True, col)
            rect = surf.get_rect(center=(x * TILE + TILE // 2, y_offset + y * TILE + TILE // 2))
            screen.blit(surf, rect)

def endgame_message(outcome: chess.Outcome) -> str:
    """Format a short, readable message for the game outcome."""
    t = outcome.termination
    if t == chess.Termination.CHECKMATE:
        return f"Checkmate — {'White' if outcome.winner else 'Black'} wins!"
    if t == chess.Termination.STALEMATE:
        return "Stalemate — draw"
    if t == chess.Termination.INSUFFICIENT_MATERIAL:
        return "Draw — insufficient material"
    if t in (chess.Termination.THREEFOLD_REPETITION, chess.Termination.FIVEFOLD_REPETITION):
        return "Draw — repetition"
    if t in (chess.Termination.FIFTY_MOVES, chess.Termination.SEVENTYFIVE_MOVES):
        return "Draw — 50/75-move rule"
    return f"Game over — {outcome.result()} ({t.name})"

def endgame_modal(
    screen: pygame.Surface,
    board: chess.Board,
    piece_images: Dict[str, pygame.Surface],
    white_bottom: bool,
    top_label: Optional[str],
    bottom_label: Optional[str],
    result_text: str = "Game over",
) -> str:
    """Display a translucent modal with the final position and two buttons."""
    clock = pygame.time.Clock()

    # Semi-transparent overlay above the board area
    overlay = pygame.Surface((BOARD_SIZE, BOARD_SIZE), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 140))

    # Card/dialog geometry
    card_w, card_h = int(BOARD_SIZE * 0.72), int(WINDOW_HEIGHT * 0.46)
    card_x, card_y = (BOARD_SIZE - card_w) // 2, (WINDOW_HEIGHT - card_h) // 2
    card = pygame.Rect(card_x, card_y, card_w, card_h)

    # Buttons: Play Again (green) and Exit (red)
    btn_w, btn_h, gap = int(card_w * 0.38), 56, int(card_w * 0.08)
    btn_y = card_y + card_h - btn_h - 28
    btn_again = pygame.Rect(card_x + gap, btn_y, btn_w, btn_h)
    btn_menu = pygame.Rect(card_x + card_w - gap - btn_w, btn_y, btn_w, btn_h)

    # Fonts
    title_font = pygame.font.SysFont("arial", 36, bold=True)
    body_font = pygame.font.SysFont("arial", 22)
    btn_font = pygame.font.SysFont("arial", 24, bold=True)

    top_label = top_label or "Top"
    bottom_label = bottom_label or "Bottom"

    while True:
        clock.tick(60)

        # Minimal event handling to exit or choose an action
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                return "menu"
            if ev.type == pygame.KEYDOWN:
                if ev.key in (pygame.K_ESCAPE, pygame.K_q):
                    return "menu"
                if ev.key in (pygame.K_RETURN, pygame.K_SPACE):
                    return "play_again"
            if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                if btn_again.collidepoint(ev.pos):
                    return "play_again"
                if btn_menu.collidepoint(ev.pos):
                    return "menu"

        # Draw the frozen final position behind the modal
        draw_board(screen, None, None, white_bottom, HUD_TOP)
        draw_pieces(screen, board, piece_images, white_bottom, HUD_TOP)
        draw_banners(screen, top_label, bottom_label)

        # Overlay + card
        screen.blit(overlay, (0, HUD_TOP))
        pygame.draw.rect(screen, (245, 245, 245), card, border_radius=18)
        pygame.draw.rect(screen, (40, 40, 40), card, 2, border_radius=18)

        # Title and subtext
        title = title_font.render(result_text, True, (20, 20, 20))
        screen.blit(title, title.get_rect(center=(BOARD_SIZE // 2, card_y + 70)))
        sub = body_font.render("What would you like to do?", True, (70, 70, 70))
        screen.blit(sub, sub.get_rect(center=(BOARD_SIZE // 2, card_y + 115)))

        # Buttons
        for rect, label, bg in ((btn_again, "Play Again", (70, 160, 75)),
                                (btn_menu, "Exit", (185, 70, 70))):
            pygame.draw.rect(screen, bg, rect, border_radius=12)
            pygame.draw.rect(screen, (30, 30, 30), rect, 2, border_radius=12)
            t = btn_font.render(label, True, (255, 255, 255))
            screen.blit(t, t.get_rect(center=rect.center))

        pygame.display.flip()

def ask_stockfish_elo(screen: pygame.Surface, initial: Optional[int] = None) -> Optional[int]:
    """Numeric prompt for Stockfish Elo; Return None to cancel."""
    clock = pygame.time.Clock()

    # Fonts
    title_font = pygame.font.SysFont("arial", 48, bold=True)
    label_font = pygame.font.SysFont("arial", 24, bold=True)
    input_font = pygame.font.SysFont("arial", 36, bold=True)
    btn_font = pygame.font.SysFont("arial", 24, bold=True)

    width, height = screen.get_width(), screen.get_height()
    text = str(int(STOCKFISH_ELO_MIN if initial is None else initial))

    # Controls: input box + two buttons
    box_rect = pygame.Rect(0, 0, 260, 64); box_rect.center = (width // 2, height // 2 - 10)
    btn_w, btn_h, gap = 220, 56, 16
    start_rect = pygame.Rect(0, 0, btn_w, btn_h); start_rect.center = (width // 2, height // 2 + 70)
    back_rect  = pygame.Rect(0, 0, btn_w, btn_h); back_rect.center  = (width // 2, height // 2 + 70 + btn_h + gap)

    # Simple helpers
    parse_elo = lambda s: int(s) if s.strip().isdigit() else None
    clamp_elo = lambda v: max(STOCKFISH_ELO_MIN, min(STOCKFISH_ELO_MAX, v))

    while True:
        clock.tick(FPS)

        # Events: typing, arrows to adjust, enter to confirm, back to menu.
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                pygame.quit(); sys.exit(0)
            if ev.type == pygame.KEYDOWN:
                if ev.key in (pygame.K_ESCAPE, pygame.K_q): return None
                if ev.key == pygame.K_RETURN:
                    elo = parse_elo(text)
                    if elo is not None and STOCKFISH_ELO_MIN <= elo <= STOCKFISH_ELO_MAX: return elo
                if ev.key == pygame.K_BACKSPACE: text = text[:-1]
                elif ev.key == pygame.K_UP:   text = str(clamp_elo((parse_elo(text) or STOCKFISH_ELO_MIN) + 50))
                elif ev.key == pygame.K_DOWN: text = str(clamp_elo((parse_elo(text) or STOCKFISH_ELO_MIN) - 50))
                else:
                    ch = ev.unicode
                    if ch.isdigit() and len(text) < 4: text += ch
            if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                if back_rect.collidepoint(ev.pos): return None
                if start_rect.collidepoint(ev.pos):
                    elo = parse_elo(text)
                    if elo is not None and STOCKFISH_ELO_MIN <= elo <= STOCKFISH_ELO_MAX: return elo

        # Background: chessboard tiling
        rows = (WINDOW_HEIGHT + TILE - 1) // TILE
        for y in range(rows):
            for x in range(8):
                colour = LIGHT if (x + y) % 2 == 0 else DARK
                pygame.draw.rect(screen, colour, (x * TILE, y * TILE, TILE, TILE))

        # Title + shadow for legibility
        title_text = "Enter Stockfish Difficulty"
        title = title_font.render(title_text, True, (255, 255, 255))
        shadow = title_font.render(title_text, True, (0, 0, 0))
        title_rect = title.get_rect(center=(width // 2, height // 2 - 140))
        screen.blit(shadow, title_rect.move(2, 2))
        screen.blit(title, title_rect)

        # Range label
        range_text = f"{STOCKFISH_ELO_MIN}–{STOCKFISH_ELO_MAX} (Elo)"
        rt = label_font.render(range_text, True, (255, 255, 255))
        rs = label_font.render(range_text, True, (0, 0, 0))
        range_rect = rt.get_rect(center=(width // 2, height // 2 - 70))
        screen.blit(rs, range_rect.move(2, 2))
        screen.blit(rt, range_rect)

        # Input box (red outline when invalid)
        elo_val = parse_elo(text)
        valid = (elo_val is not None) and (STOCKFISH_ELO_MIN <= elo_val <= STOCKFISH_ELO_MAX)
        pygame.draw.rect(screen, (245, 245, 245), box_rect, border_radius=12)
        pygame.draw.rect(screen, (40, 40, 40) if valid or text == "" else (180, 60, 60), box_rect, 2, border_radius=12)

        # Typed value
        typed = input_font.render(text, True, (20, 20, 20))
        typed_rect = typed.get_rect(center=box_rect.center)
        screen.blit(typed, typed_rect)

        # Blinking caret
        if (pygame.time.get_ticks() // 500) % 2 == 0:
            x = typed_rect.right + 4
            y1, y2 = box_rect.top + 4, box_rect.bottom - 4
            pygame.draw.line(screen, (20, 20, 20), (x, y1), (x, y2), 2)

        mouse = pygame.mouse.get_pos()

        def draw_btn(rect: pygame.Rect, label: str, base: Tuple[int, int, int], enabled: bool = True) -> None:
            """Small helper to draw rounded buttons with a subtle hover state."""
            hovered = rect.collidepoint(mouse) and enabled
            bg = tuple(min(c + 15, 255) for c in base) if hovered else base
            if not enabled: bg = (150, 150, 150)
            pygame.draw.rect(screen, bg, rect, border_radius=12)
            pygame.draw.rect(screen, (30, 30, 30), rect, 2, border_radius=12)
            t = btn_font.render(label, True, (255, 255, 255))
            screen.blit(t, t.get_rect(center=rect.center))

        # Buttons
        draw_btn(start_rect, "Start", (70, 160, 75), enabled=valid)
        draw_btn(back_rect, "Back to Menu", (185, 70, 70), enabled=True)

        pygame.display.flip()

def show_start_screen(screen: pygame.Surface) -> str:
    """Main menu: returns 'play', 'rl_vs_sf', or 'exit' based on the clicked button."""
    clock = pygame.time.Clock()
    title_font = pygame.font.SysFont("arial", 60, bold=True)
    btn_font = pygame.font.SysFont("arial", 28, bold=True)

    # Layout three stacked buttons centred under the title
    btn_w, btn_h, gap = 300, 64, 18
    play_rect = pygame.Rect(0, 0, btn_w, btn_h)
    vs_rect   = pygame.Rect(0, 0, btn_w, btn_h)
    exit_rect = pygame.Rect(0, 0, btn_w, btn_h)

    cx, cy = BOARD_SIZE // 2, WINDOW_HEIGHT // 2 + 10
    play_rect.center, vs_rect.center, exit_rect.center = (cx, cy), (cx, cy + btn_h + gap), (cx, cy + 2 * (btn_h + gap))

    while True:
        clock.tick(FPS)

        # Mouse click → route to selection
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                pygame.quit(); sys.exit(0)
            if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                if play_rect.collidepoint(ev.pos): return "play"
                if vs_rect.collidepoint(ev.pos):   return "rl_vs_sf"
                if exit_rect.collidepoint(ev.pos): return "exit"

        # Background chessboard pattern
        rows = (WINDOW_HEIGHT + TILE - 1) // TILE
        for y in range(rows):
            for x in range(8):
                colour = LIGHT if (x + y) % 2 == 0 else DARK
                pygame.draw.rect(screen, colour, (x * TILE, y * TILE, TILE, TILE))

        # Title with a faint drop shadow for contrast
        title = title_font.render("GrandMasker's Chess Bot", True, (255, 255, 255))
        shadow = title_font.render("GrandMasker's Chess Bot", True, (0, 0, 0))
        rect   = title.get_rect(center=(BOARD_SIZE // 2, WINDOW_HEIGHT // 2 - 120))
        screen.blit(shadow, rect.move(2, 2))
        screen.blit(title, rect)

        mouse = pygame.mouse.get_pos()

        def draw_btn(rect: pygame.Rect, label: str, base: Tuple[int, int, int]) -> None:
            """Menu button with subtle hover brightening and a dark outline."""
            hovered = rect.collidepoint(mouse)
            bg = tuple(min(c + 15, 255) for c in base) if hovered else base
            pygame.draw.rect(screen, bg, rect, border_radius=14)
            pygame.draw.rect(screen, (30, 30, 30), rect, 2, border_radius=14)
            txt = btn_font.render(label, True, (255, 255, 255))
            screen.blit(txt, txt.get_rect(center=rect.center))

        # Green = play vs bot; Blue = bot vs Stockfish; Red = exit
        draw_btn(play_rect, "Play against Chess Bot", (70, 160, 75))
        draw_btn(vs_rect,   "Chess Bot vs Stockfish", (70, 120, 185))
        draw_btn(exit_rect, "Exit", (185, 70, 70))

        pygame.display.flip()
