from __future__ import annotations
from pathlib import Path
import sys

BASE = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))

def res_path(rel: str) -> Path:
    return BASE / rel

# Resources at project root
PTH_PATH       = res_path("policy_value_net.pth")
PIECES_DIR     = res_path("pieces") 
STOCKFISH_PATH = res_path("stockfish.exe")     

# UI / game constants
TILE = 80
BOARD_SIZE = TILE * 8

HUD_TOP = 44
HUD_BOTTOM = 44
WINDOW_HEIGHT = BOARD_SIZE + HUD_TOP + HUD_BOTTOM

FPS = 30

LIGHT = (240, 217, 181)
DARK = (181, 136, 99)
HIGHLIGHT = (246, 246, 105)

FALLBACK_TO_LETTERS = False

STOCKFISH_ELO_DEFAULT = 1350
STOCKFISH_ELO_MIN = 1350
STOCKFISH_ELO_MAX = 2850
