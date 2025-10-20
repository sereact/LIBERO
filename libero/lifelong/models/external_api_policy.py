import os
import time
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import numpy as np
import requests
import msgpack
from libero.lifelong.models.base_policy import BasePolicy
from libero.lifelong.models.lerobot_messaging import (
    decoder_data, 
    encoder_data,
    json_like_encoder
)

def envelope(msg: Any) -> Dict:
    """Wrap a message in an envelope. Just a dict for future json serialization.

    Args:
        msg (Any): Any serializable message, might be (int,string,float,list,dict,tuple)

    Returns:
        dict: Returns the msg wrapped in an envelope.
    """
    return {"payload": msg, "monotonic_time": time.monotonic()}


def rm_envelope(msg: dict) -> Any:
    """Remove the envelope from a message.

    Args:
        msg (dict): The wrapped message

    Returns:
        Any: The payload of the message
    """
    return msg["payload"]


def pack_msg(msg: Any, json_like=False) -> bytes:
    """Packs msg to a bytearray following the msgpack specification.

    Args:
        msg (Any): Any message to send.

    Returns:
        bytearray: [description]
    """
    encoder = json_like_encoder if json_like else encoder_data
    msg = envelope(msg)  # Wrap message in an envelope
    return msgpack.packb(msg, default=encoder, use_bin_type=True)  # Pack to byte array using msgpack


def unpack_msg(packed: bytes, with_header: bool = False) -> Any:
    """Unpack an image message message.

    Args:
        packed (bytearray): bytearray containin a msgpack message
    """
    unpacked = msgpack.unpackb(
        packed, raw=False, object_hook=decoder_data
    )
    if with_header:
        if isinstance(unpacked["monotonic_time"], list):
            unpacked["monotonic_time"] = unpacked["monotonic_time"][0]
        return unpacked
    else:
        return rm_envelope(unpacked)

class LerobotPolicyClient:
    """
    Thin HTTP client for the external policy server.

    Expects a POST /predict endpoint accepting msgpack-encoded observations and
    returning msgpack-encoded results (e.g., {"action": [...]}).
    """
    def __init__(self, server_url: str = "http://localhost:8000", timeout: float = 30.0, attempts: int = 3):
        self.server_url = server_url.rstrip("/")
        self.session = requests.Session()
        self.timeout = timeout
        self.attempts = attempts

    def predict(self, obs_dict: Dict[str, Any]) -> Dict[str, Any]:
        data = pack_msg(obs_dict)
        for attempt in range(1, self.attempts + 1):
            try:
                resp = self.session.post(
                    f"{self.server_url}/predict",
                    data=data,
                    headers={"Content-Type": "application/octet-stream"},
                    timeout=self.timeout,
                )
                resp.raise_for_status()
                return unpack_msg(resp.content)
            
            except Exception as e:
                print(f"Predict failed, Error occurred: {e}")
                last_exc = e
            
            if attempt < self.attempts:
                time.sleep(1.0)
        
        raise last_exc

    def predict_with_timing(self, obs_dict: Dict[str, Any]) -> Tuple[Dict[str, Any], float]:
        t0 = time.time()
        out = self.predict(obs_dict)
        return out, time.time() - t0

    def close(self):
        try:
            self.session.close()
        except Exception:
            pass


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
    def __init__(self, cfg, shape_meta=None, use_buffer=True):
        super().__init__(cfg, shape_meta)
        # Device from cfg if provided (e.g., "cuda:0")

        server_url = cfg.policy.server_url
        timeout = 30.0

        self.client = LerobotPolicyClient(server_url=server_url, timeout=timeout)

        # Keep shape_meta if needed downstream; not strictly required for remote calls
        self.shape_meta = shape_meta

        self.use_buffer = use_buffer
        self._buffer = torch.tensor([[]])
        self._i = 0

    def get_action(self, data):
        
        if not self.use_buffer or self._i == self._buffer.shape[1]:
            self._buffer = self.client.predict(data)
            self._i = 0

        if self.use_buffer:
            action = self._buffer[:, self._i, :]
            self._i += 1
        else: 
            action = self._buffer[:, 0, :]

        return action.cpu().numpy()

    def close(self):
        self.client.close()

    def __del__(self):
        self.close()