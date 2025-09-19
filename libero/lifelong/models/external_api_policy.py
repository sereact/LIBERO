import os
import time
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import numpy as np
import requests
import msgpack
from libero.lifelong.models.base_policy import BasePolicy

def pack_msg(obj: Dict[str, Any]) -> bytes:
    # Convert Python dict into msgpack bytes
    return msgpack.packb(obj, use_bin_type=True)


def unpack_msg(buf: bytes) -> Dict[str, Any]:
    # Convert msgpack bytes back into Python dict
    return msgpack.unpackb(buf, raw=False)


class LerobotPolicyClient:
    """
    Thin HTTP client for the external policy server.

    Expects a POST /predict endpoint accepting msgpack-encoded observations and
    returning msgpack-encoded results (e.g., {"action": [...]}).
    """
    def __init__(self, server_url: str = "http://localhost:8000", timeout: float = 30.0):
        self.server_url = server_url.rstrip("/")
        self.session = requests.Session()
        self.timeout = timeout

    def predict(self, obs_dict: Dict[str, Any]) -> Dict[str, Any]:
        data = pack_msg(obs_dict)
        resp = self.session.post(
            f"{self.server_url}/predict",
            data=data,
            headers={"Content-Type": "application/octet-stream"},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return unpack_msg(resp.content)

    def predict_with_timing(self, obs_dict: Dict[str, Any]) -> Tuple[Dict[str, Any], float]:
        t0 = time.time()
        out = self.predict(obs_dict)
        return out, time.time() - t0

    def close(self):
        try:
            self.session.close()
        except Exception:
            pass


def _to_serializable_array(x: np.ndarray) -> Any:
    """
    Make numpy array JSON/msgpack-friendly. Using .tolist() here for maximum
    compatibility with typical Python servers. If you control the server and want
    speed, switch to bytes + shape/dtype metadata.
    """
    if isinstance(x, np.ndarray):
        return x.tolist()
    return x


def _tensor_last_step(t: torch.Tensor) -> torch.Tensor:
    """
    Given a tensor shaped [B, T, ...], return the last time-step [B, ...].
    If tensor has shape [B, ...], returns as-is.
    """
    if t.dim() >= 3:
        return t[:, -1]
    return t


def _maybe_first(elem: Any) -> Any:
    """
    If elem has a batch dimension (e.g., [B, ...]) and B==1, strip it.
    """
    if isinstance(elem, np.ndarray) and elem.ndim >= 1 and elem.shape[0] == 1:
        return elem[0]
    if isinstance(elem, list) and len(elem) == 1:
        return elem[0]
    return elem


class ExternalAPIPolicy(BasePolicy):
    """
    A LIBERO-compatible policy that delegates action computation to an external server.

    Expected input keys (matching your datasets / evaluation loop):
      - agentview_rgb: [B, T, C, H, W] or [B, C, H, W] uint8/float
      - eye_in_hand_rgb: same as above (if used)
      - joint_states: [B, T, D] or [B, D]
      - gripper_states: [B, T, D] or [B, D]
      - task_emb: [B, E] (optional but recommended)
      - lang: [B, T, L] or [B, L] (optional)

    Forward returns a dict:
      { "action": torch.FloatTensor of shape [B, A] }
    """
    def __init__(self, cfg, shape_meta=None):
        super().__init__(cfg, shape_meta)
        # Device from cfg if provided (e.g., "cuda:0")
        self.device = getattr(cfg, "device", "cpu")
        # Policy config overrides (Hydra style): cfg.policy may exist
        pconf = getattr(cfg, "policy", None)

        # Allow overrides from env vars or cfg
        server_url = None
        timeout = 30.0

        if pconf is not None:
            # If passed via policy.external_api.server_url or policy.server_url
            server_url = getattr(pconf, "server_url", None) or getattr(pconf, "external_api", None)
            if isinstance(server_url, (dict,)):
                server_url = server_url.get("server_url", None)
            timeout = float(getattr(pconf, "timeout", 30.0))

        # Also allow env var override
        server_url = os.environ.get("EXTERNAL_POLICY_SERVER_URL", server_url or "http://localhost:8000")
        timeout = float(os.environ.get("EXTERNAL_POLICY_TIMEOUT", timeout))

        self.client = LerobotPolicyClient(server_url=server_url, timeout=timeout)

        # Keep shape_meta if needed downstream; not strictly required for remote calls
        self.shape_meta = shape_meta

    def to(self, device):
        self.device = device if isinstance(device, str) else str(device)
        return super().to(device)

    def _build_obs_payload(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert input batch tensors to a serializable observation dictionary
        for the external server. We take the last time step for sequence inputs.
        """
        payload: Dict[str, Any] = {}

        def tensor_to_np(t: Optional[torch.Tensor]) -> Optional[np.ndarray]:
            if t is None:
                return None
            t = _tensor_last_step(t)  # [B, ...] or [...]
            if t.is_cuda:
                t = t.detach().cpu()
            return t.numpy() if isinstance(t, torch.Tensor) else t

        # Images: convert CHW to HWC uint8 if necessary
        for key in ["agentview_rgb", "eye_in_hand_rgb"]:
            if key in batch and batch[key] is not None:
                t = _tensor_last_step(batch[key])  # [B, C, H, W] or [C, H, W]
                if isinstance(t, torch.Tensor):
                    if t.dim() == 5:
                        # [B, T, C, H, W] -> handled by _tensor_last_step => [B, C, H, W]
                        pass
                    if t.is_cuda:
                        t = t.detach().cpu()
                    arr = t.numpy()
                else:
                    arr = t
                # If batch present, take B=1 by default for evaluation
                arr = _maybe_first(arr)
                # Convert to HWC for typical servers
                if arr.ndim == 3 and arr.shape[0] in (1, 3):
                    # assume CHW, move to HWC
                    arr = np.transpose(arr, (1, 2, 0))
                # Scale to uint8 if float in [0,1]
                if arr.dtype != np.uint8:
                    arr = np.clip(arr, 0.0, 1.0) * 255.0 if arr.dtype.kind == "f" else arr
                    arr = arr.astype(np.uint8)
                payload[key] = _to_serializable_array(arr)

        # Low-dim states
        for key_in, key_out in [
            ("joint_states", "joint_states"),
            ("gripper_states", "gripper_states"),
            ("proprio", "proprio"),  # if your batch uses a combined proprio key
        ]:
            if key_in in batch and batch[key_in] is not None:
                arr = tensor_to_np(batch[key_in])
                arr = _maybe_first(arr)
                if arr is not None:
                    payload[key_out] = _to_serializable_array(np.asarray(arr))

        # Task / language embeddings if present
        if "task_emb" in batch and batch["task_emb"] is not None:
            arr = tensor_to_np(batch["task_emb"])
            arr = _maybe_first(arr)
            if arr is not None:
                payload["task_emb"] = _to_serializable_array(np.asarray(arr))

        if "lang" in batch and batch["lang"] is not None:
            arr = tensor_to_np(batch["lang"])
            arr = _maybe_first(arr)
            if arr is not None:
                payload["lang"] = _to_serializable_array(np.asarray(arr))

        return payload

    @torch.no_grad()
    def forward(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Forward pass that delegates to the external API.
        For batch size > 1, this loops over batch items.
        Returns:
            {"action": torch.FloatTensor [B, A]}
        """
        # Normalize to a list of per-sample payloads
        # We attempt to infer batch size from any present tensor
        bsz = None
        for v in batch.values():
            if isinstance(v, torch.Tensor):
                # Could be [B, T, ...] or [B, ...]
                if v.dim() >= 1:
                    bsz = v.shape[0]
                    break
        if bsz is None:
            bsz = 1

        actions = []
        if bsz == 1:
            payload = self._build_obs_payload(batch)
            result = self.client.predict(payload)
            act = np.asarray(result.get("action", []), dtype=np.float32)
            actions.append(torch.from_numpy(act).to(self.device))
        else:
            # Split along batch dimension and call the server per item
            for i in range(bsz):
                one = {}
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor) and v.dim() >= 1 and v.shape[0] == bsz:
                        one[k] = v[i:i+1]
                    else:
                        one[k] = v
                payload = self._build_obs_payload(one)
                result = self.client.predict(payload)
                act = np.asarray(result.get("action", []), dtype=np.float32)
                actions.append(torch.from_numpy(act).to(self.device))

        # Stack to [B, A]; if server returns 1D action we ensure correct shape
        actions = [a.view(1, -1) if a.dim() == 1 else a for a in actions]
        action_tensor = torch.cat(actions, dim=0)
        return {"action": action_tensor}

    def close(self):
        self.client.close()

    def __del__(self):
        self.close()