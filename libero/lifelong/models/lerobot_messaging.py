import time
from typing import Any, Dict
import msgpack
import logging
from typing import Any, Dict, List, Union, Tuple
import numpy as np
import torch
import requests

class TorchSerialize:
    def encodes(self, o: Union[torch.Tensor, np.ndarray]) -> dict:
        if isinstance(o, torch.Tensor):
            np_data = o.numpy()
            return {
                "data": np_data.tobytes(), "dtype": np_data.dtype.str, "encoding": "raw_bytes", "shape": o.shape, "type": "tensor"
            }
        elif isinstance(o, np.ndarray):
            return {
                "data": o.tobytes(), "shape": o.shape, "dtype": o.dtype.str, "encoding": "raw_bytes", "type": "array"
            }
        else:
            return o

    def decodes(self, o: Dict) -> Union[torch.Tensor, np.ndarray]:
        dtype = o["dtype"]
        t = o["type"]
        arr = np.frombuffer(o["data"], dtype=dtype)
        arr = arr.reshape(o["shape"])

        if t == "tensor":
            retval = torch.as_tensor(arr)
        elif t == "array":
            retval = arr
        return retval

class JsonLikeSerialize:
    def encodes(self, o: Union[torch.Tensor, np.ndarray, Any]) -> dict:
        if not isinstance(o, list):
            data = o.tolist()
            return {
                "data": data, "dtype": "tensor", "encoding": "json"
            }


img_serializer = TorchSerialize()

class TensorEncoder:

    def __init__(self, image_data_function) -> None:
        self.image_data_function = image_data_function

    def __call__(self, obj: Any) -> Any:
        if isinstance(obj, (np.ndarray, torch.Tensor)):
            return self.image_data_function(obj)
        else:
            return obj


class TensorDecoder:
    def __init__(self, image_data_function) -> None:
        self.image_data_function = image_data_function

    def __call__(self, obj: Any) -> Any:
        if '__image_data__' in obj:
            return self.image_data_function(obj)
        else:
            return obj


def encode_image_data(obj: Union[torch.Tensor, np.ndarray]) -> Dict:
    return {"__image_data__": True, "as_str": img_serializer.encodes(obj)}

def encode_image_data_json(obj: Union[torch.Tensor, np.ndarray]) -> List:
    return img_serializer.encodes(obj)

def decode_image_data(obj: Dict) -> Union[torch.Tensor, np.ndarray]:
    return img_serializer.decodes(obj["as_str"])

encoder_data = TensorEncoder(image_data_function=encode_image_data)
decoder_data = TensorDecoder(image_data_function=decode_image_data)
json_like_encoder = TensorEncoder(image_data_function=encode_image_data_json)


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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LerobotPolicyClient:
    def __init__(self, server_url: str = "http://localhost:8000"):
        """
        Initialize the MoT Policy Client
        
        Args:
            server_url: URL of the MoT Policy Server
        """
        self.server_url = server_url.rstrip('/')
        self.session = requests.Session()
        
        # Set timeout for requests
        self.timeout = 30
        
    def predict(self, obs_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send observation data to server for prediction
        
        Args:
            obs_dict: Dictionary containing observation data
            
        Returns:
            Dictionary containing prediction results
        """
        try:
            # Pack the observation data using msgpack
            packed_data = pack_msg(obs_dict)
            
            # Send POST request to prediction endpoint
            response = self.session.post(
                f"{self.server_url}/predict",
                data=packed_data,
                headers={'Content-Type': 'application/octet-stream'},
                timeout=self.timeout
            )
            
            response.raise_for_status()
            
            # Unpack the response
            result = unpack_msg(response.content)
            
            return result
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise e
        
    def predict_with_timing(self, obs_dict: Dict[str, Any]) -> Tuple[Dict[str, Any], float]:
        """
        Send observation data to server for prediction with timing
        
        Args:
            obs_dict: Dictionary containing observation data
            
        Returns:
            Tuple of (prediction results, elapsed time in seconds)
        """
        st = time.time()
        result = self.predict(obs_dict)
        cprint(f"Prediction time: {time.time() - st:.3f} seconds", "blue")
        return result
    
    def close(self):
        """Close the session"""
        self.session.close()

