from typing import Optional, Tuple
import torch
import chess

# Policy head size: 73 planes × 64 squares (8×8)
POLICY_SIZE = 4672

# Map piece type → base channel (0..5 for white, 6..11 for black)
PIECE2PLANE = {
    chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
    chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5,
}

# Sliding directions and knight jumps
DIRS = [(1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)]
KNIGHTS = [(2,1),(2,-1),(-2,1),(-2,-1),(1,2),(1,-2),(-1,2),(-1,-2)]
KNIGHT_IDX = {d: i for i, d in enumerate(KNIGHTS)}


def _rf(sq: int) -> Tuple[int, int]:
    """Return (rank, file) with 0..7 indexing."""
    return divmod(sq, 8)


def _flip_rank(sq: int) -> int:
    """Flip square vertically (useful for Black POV)."""
    r, f = _rf(sq)
    return (7 - r) * 8 + f


def _orient(sq: int, stm_white: bool) -> int:
    """Orient square so side-to-move is treated as 'white'."""
    return sq if stm_white else _flip_rank(sq)


def board_to_planes(board: chess.Board) -> torch.Tensor:
    """
    Encode board state into 18 planes [C, 8, 8].

    0..11 : piece one-hot (white 0..5, black 6..11)
    12..15: castling rights (Wk, Wq, Bk, Bq) — absolute
    16    : side-to-move (1 for White, 0 for Black)
    17    : spare (all zeros)

    Squares are oriented so the side-to-move is 'playing up'.
    """
    planes = torch.zeros((18, 8, 8), dtype=torch.float32)
    stm_white = (board.turn == chess.WHITE)

    # Pieces (place 1.0 at [ch, row, col])
    for colour in (chess.WHITE, chess.BLACK):
        off = 0 if colour == chess.WHITE else 6
        for pt in (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING):
            for sq in board.pieces(pt, colour):
                ch = PIECE2PLANE[pt] + off
                # Orient square to STM POV then map to row,col (row 0 is top)
                sq_o = _orient(sq, stm_white)
                r, f = _rf(sq_o)
                planes[ch, 7 - r, f] = 1.0

    # Castling rights as four full planes (absolute)
    rights = (
        board.has_kingside_castling_rights(chess.WHITE),
        board.has_queenside_castling_rights(chess.WHITE),
        board.has_kingside_castling_rights(chess.BLACK),
        board.has_queenside_castling_rights(chess.BLACK),
    )
    for i, ok in enumerate(rights):
        if ok:
            planes[12 + i].fill_(1.0)

    # Side to move plane
    if board.turn == chess.WHITE:
        planes[16].fill_(1.0)
    # Plane 17 remains zeros (spare)

    return planes


def move_to_index(board: chess.Board, move: chess.Move) -> Optional[int]:
    """
    Map a legal move to [0..4671] using AlphaZero-style encoding.

    0..55  : sliders (8 dirs × 7 distances) × 64 from-squares
    56..63 : knight jumps (8 patterns) × 64 from-squares
    64..72 : underpromotions (R,B,N) × (left, straight, right) × 64 from-squares
    """
    stm_white = board.turn
    fr = _orient(move.from_square, stm_white)
    to = _orient(move.to_square,   stm_white)
    r1, f1 = _rf(fr); r2, f2 = _rf(to)
    dr, df = r2 - r1, f2 - f1

    # Knight jumps → planes 56..63
    if (dr, df) in KNIGHT_IDX:
        return (56 + KNIGHT_IDX[(dr, df)]) * 64 + fr

    # Underpromotions (rook/bishop/knight) → planes 64..72
    if move.promotion in (chess.ROOK, chess.BISHOP, chess.KNIGHT):
        df_dir = 1 if df > 0 else (-1 if df < 0 else 0)  # left/straight/right from STM POV
        dir_idx = {-1: 0, 0: 1, 1: 2}[df_dir]
        promo_idx = {chess.ROOK: 0, chess.BISHOP: 1, chess.KNIGHT: 2}[move.promotion]
        return (64 + promo_idx * 3 + dir_idx) * 64 + fr

    # Helper for direction sign
    def sgn(x: int) -> int:
        return 0 if x == 0 else (1 if x > 0 else -1)

    # Sliders (rook/bishop/queen moves) → planes 0..55
    if dr == 0 or df == 0 or abs(dr) == abs(df):
        try:
            d = DIRS.index((sgn(dr), sgn(df)))  # which of the 8 directions
        except ValueError:
            return None
        steps = max(abs(dr), abs(df))          # distance 1..7
        if 1 <= steps <= 7:
            return (d * 7 + (steps - 1)) * 64 + fr

    # Anything else (e.g., unsupported promos) → None
    return None


def legal_mask(board: chess.Board) -> torch.Tensor:
    """Return boolean mask over POLICY_SIZE marking legal moves as True."""
    m = torch.zeros(POLICY_SIZE, dtype=torch.bool)
    for mv in board.legal_moves:
        idx = move_to_index(board, mv)
        if idx is not None:
            m[idx] = True
    return m


def mask_logits_to_legal(logits: torch.Tensor, board: chess.Board) -> torch.Tensor:
    """Replace illegal move logits with -inf; keep original shape."""
    single = logits.dim() == 1
    if single:
        logits = logits[None, :]
    m = legal_mask(board)[None, :].to(logits.device)
    out = logits.masked_fill(~m, float("-inf"))
    return out[0] if single else out
