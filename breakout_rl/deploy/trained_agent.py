import argparse
import asyncio
import time
from typing import Optional, Dict, Any
import numpy as np
import torch

from agents.base_agent import BaseAgent
from breakout_rl.agents.networks import DuelingMLP
from breakout_rl.env.observation import ObservationBuilder, OBS_DIM
from breakout_rl.env.high_level_env import HIGH_OBS_DIM
from breakout_rl.controllers.aim_controller import choose_action, brick_mass_offset
from server.logic import Breakout
from breakout_rl.constants import ACTION_WEST, ACTION_EAST, DECISION_LINE_Y

_ACTION_MSG = {
    ACTION_WEST: {"action": "move", "direction": "WEST"},
    ACTION_EAST: {"action": "move", "direction": "EAST"},
}


def _load(net_path: str, in_dim: int, hidden: int = 128) -> DuelingMLP:
    net = DuelingMLP(in_dim, 3, hidden)
    net.load_state_dict(torch.load(net_path, map_location="cpu"))
    net.eval()
    return net


class FlatDQNAgent(BaseAgent):
    def __init__(self, net_path: str, **kw):
        super().__init__(**kw)
        self.net = _load(net_path, OBS_DIM)
        self.obs_builder = ObservationBuilder()
        self._last_t: Optional[float] = None

    async def deliberate(self) -> Optional[Dict[str, Any]]:
        st = self.current_state
        if not st or st.get("game_over"):
            return None
        now = time.monotonic()
        dt = (now - self._last_t) if self._last_t else (1.0 / 30.0)  # measured dt
        self._last_t = now
        obs = self.obs_builder.build(st, dt)
        with torch.no_grad():
            a = int(self.net(torch.as_tensor(obs).unsqueeze(0)).argmax(1).item())
        return _ACTION_MSG.get(a)  # None for NOOP


def _mirror_from_state(st: Dict[str, Any]) -> Breakout:
    """Build a local Breakout matching the wire state, for the controller's predictor.
    Velocity is unknown on the wire, so estimate it from the builder's finite diff is
    not available here; instead we reconstruct vx/vy from two successive states in the
    hierarchical agent (see below)."""
    g = Breakout(width=int(st["width"]), height=int(st["height"]))
    g.paddle_width = st["paddle_width"]
    g.paddle_x = st["paddle_x"]
    g.ball_radius = st["ball_radius"]
    g.ball_x = st["ball_x"]
    g.ball_y = st["ball_y"]
    active_idx = {b["index"] for b in st.get("bricks", [])}
    for b in g.bricks:
        b.active = b.index in active_idx
    g._sync_bricks_to_numpy()
    return g


class HierarchicalAgent(BaseAgent):
    def __init__(self, net_path: str, **kw):
        super().__init__(**kw)
        self.net = _load(net_path, HIGH_OBS_DIM)
        self.obs_builder = ObservationBuilder()
        self._prev: Optional[Dict[str, Any]] = None
        self._prev_t: Optional[float] = None
        self._region = 1  # default CENTER
        self._prev_vy = 0.0

    async def deliberate(self) -> Optional[Dict[str, Any]]:
        st = self.current_state
        if not st or st.get("game_over"):
            return None
        now = time.monotonic()
        dt = (now - self._prev_t) if self._prev_t else (1.0 / 30.0)
        # estimate ball velocity from two successive wire states
        if self._prev is not None and dt > 0:
            vx = (st["ball_x"] - self._prev["ball_x"]) / dt
            vy = (st["ball_y"] - self._prev["ball_y"]) / dt
        else:
            vx, vy = 0.0, 0.0
        g = _mirror_from_state(st)
        g.ball_vx, g.ball_vy = vx, vy

        # new high-level decision at a clean descent decision point
        if g.ball_y > DECISION_LINE_Y and vy > 0 and self._prev_vy <= 0:
            # match the training observation: shared 23-dim + brick-centroid offset
            obs = self.obs_builder.build(st, dt)
            obs = np.concatenate([obs, [brick_mass_offset(g)]]).astype(np.float32)
            with torch.no_grad():
                self._region = int(
                    self.net(torch.as_tensor(obs).unsqueeze(0)).argmax(1).item()
                )
        else:
            self.obs_builder.build(st, dt)  # keep history warm

        self._prev, self._prev_t, self._prev_vy = st, now, vy
        a = choose_action(g, self._region)
        return _ACTION_MSG.get(a)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["flat", "hier"], default="flat")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--uri", default="ws://localhost:8765/ws")
    args = p.parse_args()
    cls = FlatDQNAgent if args.mode == "flat" else HierarchicalAgent
    asyncio.run(cls(args.checkpoint, server_uri=args.uri).run())
