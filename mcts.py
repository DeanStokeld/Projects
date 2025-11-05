import math
from typing import Optional, Tuple, Dict

import chess
import torch

from net import ChessNet
from encode import board_to_planes, mask_logits_to_legal, move_to_index


class Node:
    """
    Lightweight MCTS node:
      - prior: network policy prior π(a|s) for the move from the parent
      - N: visit count
      - W: total backed-up value
      - Q: mean value (W / N)
      - children: map of legal child moves → Node
      - move: move taken from the parent to reach this node
      - terminal: True if the position at this node is game-over
      - value: leaf evaluation (from net) or terminal value used for backup
    """
    __slots__ = ("prior", "N", "W", "Q", "children", "move", "terminal", "value")

    def __init__(self, prior: float, move: Optional[chess.Move], terminal: bool):
        self.prior = float(prior)
        self.N = 0
        self.W = 0.0
        self.Q = 0.0
        self.children: Dict[chess.Move, "Node"] = {}
        self.move = move
        self.terminal = terminal
        self.value = 0.0


class MCTS:
    """
    AlphaZero-style Monte Carlo Tree Search guided by a policy–value network.

    Args:
        net: policy–value network (returns logits over moves and a scalar value)
        sims: number of simulations (selection→expansion/eval→backup) per root move
        c_puct: exploration constant for PUCT
        logit_temperature: temperature τ applied to policy logits before softmax
        value_white_pov: if True, values are normalised to White's perspective
        device: torch device string ("cpu" or "cuda")
    """

    def __init__(
        self,
        net: ChessNet,
        sims: int = 160,
        c_puct: float = 1.25,
        logit_temperature: float = 1.25,
        value_white_pov: bool = True,
        device: str = "cpu",
    ):
        self.net = net.eval()
        self.sims = sims
        self.c_puct = c_puct
        self.tau = logit_temperature
        self.v_white = value_white_pov
        self.device = device

    @torch.no_grad()
    def _eval(self, b: chess.Board) -> Tuple[torch.Tensor, float]:
        """
        Forward pass of the policy–value net.

        Returns:
            p: probability vector over encoded move indices (masked to legal)
            val: scalar value in [-1, 1] (win=+1, draw=0, loss=-1) from the configured POV
        """
        # Encode board → tensor [1, C, 8, 8]
        x = board_to_planes(b).unsqueeze(0).to(self.device, dtype=torch.float32)

        # Network heads
        logits, v = self.net(x)

        # Mask illegal moves and (optionally) apply temperature to logits
        masked = mask_logits_to_legal(logits[0], b)
        if self.tau != 1.0:
            masked = masked / self.tau

        # Normalise to a proper distribution over legal moves
        p = torch.softmax(masked, dim=-1).cpu()

        # Value: flip sign when evaluating from White POV but it's Black to move
        val = float(v.item())
        if self.v_white and b.turn == chess.BLACK:
            val = -val
        return p, val

    def run(self, board: chess.Board) -> Node:
        """
        Build a search tree from 'board' and return the root node with populated stats.
        """
        root = Node(1.0, None, board.is_game_over())
        if root.terminal:
            # If terminal at the root, just set its value and return
            root.value = self._terminal_value(board)
            return root

        # Expand root using policy priors
        priors, _ = self._eval(board)
        for mv in board.legal_moves:
            idx = move_to_index(board, mv)
            if idx is not None:
                root.children[mv] = Node(priors[idx].item(), mv, False)

        # Run simulations (selection → expansion/eval → backup)
        for _ in range(self.sims):
            self._simulate(board, root)
        return root

    def _simulate(self, board: chess.Board, root: Node) -> None:
        """
        One MCTS iteration:
          1) SELECTION via PUCT until a leaf/terminal node,
          2) EXPANSION + EVALUATION (network) at the leaf,
          3) BACKUP the value up the visited path with sign flips per ply.
        """
        b = board.copy()
        node = root
        visited = []

        # Selection (PUCT)
        while node.children and not node.terminal:
            # Sum of visits used in the exploration term
            N_sum = sum(ch.N for ch in node.children.values()) + 1
            # Choose child maximising Q + U
            mv, node = max(
                node.children.items(),
                key=lambda kv: kv[1].Q + self.c_puct * kv[1].prior * math.sqrt(N_sum) / (1 + kv[1].N),
            )
            visited.append(node)
            b.push(mv)
            if b.is_game_over():
                node.terminal = True
                break

        # Expansion + Evaluation OR terminal handling
        if not node.terminal and not node.children:
            # Leaf: expand with priors and evaluate value
            priors, v = self._eval(b)
            for mv in b.legal_moves:
                idx = move_to_index(b, mv)
                if idx is not None:
                    node.children[mv] = Node(priors[idx].item(), mv, False)
            node.value = v
        else:
            # Terminal: compute terminal value directly
            node.value = self._terminal_value(b)

        # Backup
        # Propagate value back up; alternate sign each ply (opponent's perspective)
        v = node.value
        for n in reversed(visited):
            n.N += 1
            n.W += v
            n.Q = n.W / max(1, n.N)  # guard against div-by-zero on first update
            v = -v

    def _terminal_value(self, b: chess.Board) -> float:
        """
        Return terminal value from the side-to-move perspective at 'b':
          +1 = win for side to move, -1 = loss, 0 = draw.
        """
        out = b.outcome(claim_draw=True)
        if out is None or out.winner is None:
            return 0.0
        return 1.0 if out.winner == b.turn else -1.0

    def best_move(self, board: chess.Board) -> Optional[chess.Move]:
        """
        Run MCTS from 'board' and pick the root child with the highest visit count.
        """
        root = self.run(board)
        return max(root.children.items(), key=lambda kv: kv[1].N)[0] if root.children else None
