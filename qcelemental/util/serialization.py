import json
from typing import Any, Union

import numpy as np

try:
    from pydantic.v1.json import pydantic_encoder
except ImportError:  # Will also trap ModuleNotFoundError
    from pydantic.json import pydantic_encoder

from .importing import which_import

try:
    import msgpack
except ModuleNotFoundError:
    pass

_msgpack_which_msg = "Please install via `conda install msgpack-python`."

## MSGPackExt


def msgpackext_encode(obj: Any) -> Any:
    r"""
    Encodes an object using pydantic and NumPy array serialization techniques suitable for msgpack.

    Parameters
    ----------
    obj : Any
        Any object that can be serialized with pydantic and NumPy encoding techniques.

    Returns
    -------
    Any
        A msgpack compatible form of the object.
    """

    # First try pydantic base objects
    try:
        return pydantic_encoder(obj)
    except TypeError:
        pass

    if isinstance(obj, np.ndarray):
        if obj.shape:
            data = {b"_nd_": True, b"dtype": obj.dtype.str, b"data": np.ascontiguousarray(obj).tobytes()}
            if len(obj.shape) > 1:
                data[b"shape"] = obj.shape
            return data

        else:
            # Converts np.array(5) -> 5
            return obj.tolist()

    return obj


def msgpackext_decode(obj: Any) -> Any:
    r"""
    Decodes a msgpack objects from a dictionary representation.

    Parameters
    ----------
    obj : Any
        An encoded object, likely a dictionary.

    Returns
    -------
    Any
        The decoded form of the object.
    """

    if b"_nd_" in obj:
        arr = np.frombuffer(obj[b"data"], dtype=obj[b"dtype"])
        if b"shape" in obj:
            arr.shape = obj[b"shape"]

        return arr

    return obj


def msgpackext_dumps(data: Any) -> bytes:
    r"""Safe serialization of a Python object to msgpack binary representation using all known encoders.
    For NumPy, encodes a specialized object format to encode all shape and type data.

    Parameters
    ----------
    data : Any
        A encodable python object.

    Returns
    -------
    bytes
        A msgpack representation of the data in bytes.
    """
    which_import("msgpack", raise_error=True, raise_msg=_msgpack_which_msg)

    return msgpack.dumps(data, default=msgpackext_encode, use_bin_type=True)


def msgpackext_loads(data: bytes) -> Any:
    r"""Deserializes a msgpack byte representation of known objects into those objects.

    Parameters
    ----------
    data : bytes
        The serialized msgpack byte array.

    Returns
    -------
    Any
        The deserialized Python objects.
    """
    which_import("msgpack", raise_error=True, raise_msg=_msgpack_which_msg)

    return msgpack.loads(data, object_hook=msgpackext_decode, raw=False)


## JSON Ext


class JSONExtArrayEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        try:
            return pydantic_encoder(obj)
        except TypeError:
            pass

        if isinstance(obj, np.ndarray):
            if obj.shape:
                data = {"_nd_": True, "dtype": obj.dtype.str, "data": np.ascontiguousarray(obj).tobytes().hex()}
                if len(obj.shape) > 1:
                    data["shape"] = obj.shape
                return data

            else:
                # Converts np.array(5) -> 5
                return obj.tolist()

        return json.JSONEncoder.default(self, obj)


def jsonext_decode(obj: Any) -> Any:
    if "_nd_" in obj:
        arr = np.frombuffer(bytes.fromhex(obj["data"]), dtype=obj["dtype"])
        if "shape" in obj:
            arr.shape = obj["shape"]

        return arr

    return obj


def jsonext_dumps(data: Any) -> str:
    r"""Safe serialization of Python objects to JSON string representation using all known encoders.
    The JSON serializer uses a custom array syntax rather than flat JSON lists.

    Parameters
    ----------
    data : Any
        A encodable python object.

    Returns
    -------
    str
        A JSON representation of the data.
    """

    return json.dumps(data, cls=JSONExtArrayEncoder)


def jsonext_loads(data: Union[str, bytes]) -> Any:
    r"""Deserializes a json representation of known objects into those objects.

    Parameters
    ----------
    data : str or bytes
        The byte-serialized JSON blob.

    Returns
    -------
    Any
        The deserialized Python objects.
    """

    return json.loads(data, object_hook=jsonext_decode)


## JSON


class JSONArrayEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        try:
            return pydantic_encoder(obj)
        except TypeError:
            pass

        if isinstance(obj, np.ndarray):
            if obj.shape:
                return obj.ravel().tolist()
            else:
                return obj.tolist()

        return json.JSONEncoder.default(self, obj)


def json_dumps(data: Any, **kwargs) -> str:
    r"""Safe serialization of a Python dictionary to JSON string representation using all known encoders.

    Parameters
    ----------
    data : Any
        A encodable python object.
    kwargs : Any
        Keyword arguments for json.dumps()

    Returns
    -------
    str
        A JSON representation of the data.
    """

    return json.dumps(data, cls=JSONArrayEncoder, **kwargs)


def json_loads(data: str) -> Any:
    r"""Deserializes a json representation of known objects into those objects.

    Parameters
    ----------
    data : str
        The serialized JSON blob.

    Returns
    -------
    Any
        The deserialized Python objects.
    """

    # Doesn't hurt anything to try to load JSONext as well
    return json.loads(data, object_hook=jsonext_decode)


## MSGPack


def msgpack_encode(obj: Any) -> Any:
    r"""
    Encodes an object using pydantic. Converts numpy arrays to plain python lists

    Parameters
    ----------
    obj : Any
        Any object that can be serialized with pydantic and NumPy encoding techniques.

    Returns
    -------
    Any
        A msgpack compatible form of the object.
    """

    try:
        return pydantic_encoder(obj)
    except TypeError:
        pass

    if isinstance(obj, np.ndarray):
        if obj.shape:
            return obj.ravel().tolist()
        else:
            return obj.tolist()

    return obj


def msgpack_dumps(data: Any) -> str:
    r"""Safe serialization of a Python object to msgpack binary representation using all known encoders.
    For NumPy, converts to lists.

    Parameters
    ----------
    data : Any
        A encodable python object.

    Returns
    -------
    str
        A msgpack representation of the data in bytes.
    """

    which_import("msgpack", raise_error=True, raise_msg=_msgpack_which_msg)

    return msgpack.dumps(data, default=msgpack_encode, use_bin_type=True)


def msgpack_loads(data: str) -> Any:
    r"""Deserializes a msgpack byte representation of known objects into those objects.

    Parameters
    ----------
    data : bytes
        The serialized msgpack byte array.

    Returns
    -------
    Any
        The deserialized Python objects.
    """

    which_import("msgpack", raise_error=True, raise_msg=_msgpack_which_msg)

    # Doesn't hurt anything to try to load msgpack-ext as well
    return msgpack.loads(data, object_hook=msgpackext_decode, raw=False)


## Helper functions


def serialize(data: Any, encoding: str, **kwargs) -> Union[str, bytes]:
    r"""Encoding Python objects using the provided encoder.

    Parameters
    ----------
    data : Any
        A encodable python object.
    encoding : str
        The type of encoding to perform: {'json', 'json-ext', 'msgpack-ext'}
    kwargs: Any
        For passing additional kwargs to serialization functions, such as indent

    Returns
    -------
    Union[str, bytes]
        A serialized representation of the data.

    """
    if encoding.lower() == "json":
        return json_dumps(data, **kwargs)
    elif encoding.lower() == "json-ext":
        return jsonext_dumps(data)
    elif encoding.lower() == "msgpack":
        return msgpack_dumps(data)
    elif encoding.lower() == "msgpack-ext":
        return msgpackext_dumps(data)
    else:
        raise KeyError(f"Encoding '{encoding}' not understood, valid options: 'json', 'json-ext', 'msgpack-ext'")


def deserialize(blob: Union[str, bytes], encoding: str) -> Any:
    r"""Encoding Python objects using .

    Parameters
    ----------
    blob : Union[str, bytes]
        The serialized data.
    encoding : str
        The type of encoding of the blob: {'json', 'json-ext', 'msgpack'}

    Returns
    -------
    Any
        The deserialized Python objects.
    """
    if encoding.lower() == "json":
        assert isinstance(blob, str)
        return json_loads(blob)
    elif encoding.lower() == "json-ext":
        assert isinstance(blob, (str, bytes))
        return jsonext_loads(blob)
    elif encoding.lower() in ["msgpack"]:
        assert isinstance(blob, bytes)
        return msgpack_loads(blob)
    elif encoding.lower() in ["msgpack-ext"]:
        assert isinstance(blob, bytes)
        return msgpackext_loads(blob)
    else:
        raise KeyError(
            f"Encoding '{encoding}' not understood, valid options: 'json', 'json-ext', 'msgpack', 'msgpack-ext'"
        )
