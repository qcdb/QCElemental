import json
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Union

import numpy as np
from pydantic import BaseModel, ConfigDict, model_serializer

from qcelemental.util import deserialize, serialize


def _repr(self) -> str:
    return f'{self.__repr_name__()}({self.__repr_str__(", ")})'


# Encoders, to be deprecated at some point
ndarray_encoder = {np.ndarray: lambda v: v.flatten().tolist()}


class ExtendedConfigDict(ConfigDict, total=False):
    serialize_default_excludes: Set
    """Add items to exclude from serialization"""

    serialize_skip_defaults: bool
    """When serializing, ignore default values (i.e. those not set by user)"""

    force_skip_defaults: bool
    """Manually force defaults to not be included in output dictionary"""

    canonical_repr: bool
    """Use canonical representation of the molecules"""

    repr_style: Union[List[str], Callable]
    """Representation styles"""


class ProtoModel(BaseModel):
    """QCSchema extension of pydantic.BaseModel."""

    model_config = ExtendedConfigDict(
        frozen=True,
        extra="forbid",
        populate_by_name=True,  # Allows using alias to populate
        serialize_default_excludes=set(),
        serialize_skip_defaults=False,
        force_skip_defaults=False,
    )

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        cls.__base_doc__ = ""  # remove when QCFractal merges `next`

        if "pydantic" in cls.__repr__.__module__:
            cls.__repr__ = _repr

        if "pydantic" in cls.__str__.__module__:
            cls.__str__ = _repr

    @classmethod
    def parse_raw(cls, data: Union[bytes, str], *, encoding: Optional[str] = None) -> "ProtoModel":  # type: ignore
        r"""
        Parses raw string or bytes into a Model object.

        This overwrites the deprecated parse_file of v2 Pydantic to eventually call parse_model or parse_model_json,
        but is kept to preserve our own API

        May also be deprecated from QCElemental in time

        Parameters
        ----------
        data
            A serialized data blob to be deserialized into a Model.
        encoding
            The type of the serialized array, available types are: {'json', 'json-ext', 'msgpack-ext', 'pickle'}

        Returns
        -------
        Model
            The requested model from a serialized format.
        """

        if encoding is None:
            if isinstance(data, str):
                encoding = "json"
            elif isinstance(data, bytes):
                encoding = "msgpack-ext"
            else:
                raise TypeError("Input is neither str nor bytes, please specify an encoding.")

        if encoding.endswith(("json", "javascript", "pickle")):
            # return super().parse_raw(data, content_type=encoding)
            return cls.model_validate_json(data)
        elif encoding in ["msgpack-ext", "json-ext", "msgpack"]:
            obj = deserialize(data, encoding)
        else:
            raise TypeError(f"Content type '{encoding}' not understood.")

        return cls.model_validate(obj)

    @classmethod
    def parse_file(cls, path: Union[str, Path], *, encoding: Optional[str] = None) -> "ProtoModel":  # type: ignore
        r"""Parses a file into a Model object.

        This overwrites the deprecated parse_file of v2 Pydantic to eventually call parse_model or parse_model_json,
        but is kept to preserve our own API

        May also be deprecated from QCElemental in time

        Parameters
        ----------
        path
            The path to the file.
        encoding
            The type of the files, available types are: {'json', 'msgpack', 'pickle'}. Attempts to
            automatically infer the file type from the file extension if None.

        Returns
        -------
        Model
            The requested model from a serialized format.

        """
        path = Path(path)
        if encoding is None:
            if path.suffix in [".json", ".js"]:
                encoding = "json"
            elif path.suffix in [".msgpack"]:
                encoding = "msgpack-ext"
            elif path.suffix in [".pickle"]:
                encoding = "pickle"
            else:
                raise TypeError("Could not infer `encoding`, please provide a `encoding` for this file.")

        return cls.parse_raw(path.read_bytes(), encoding=encoding)

    # UNCOMMENT IF NEEDED FOR UPGRADE
    #   defining this is maybe bad idea as dict(v2) does non-recursive dictionary, whereas model_dump does nested
    # def dict(self, **kwargs) -> Dict[str, Any]:
    #    warnings.warn("The `dict` method is deprecated; use `model_dump` instead.", DeprecationWarning)
    #    return self.model_dump(**kwargs)

    @model_serializer(mode="wrap")
    def _serialize_model(self, handler) -> Dict[str, Any]:
        """
        Customize the serialization output. Does duplicate with some code in model_dump, but handles the case of nested
        models and any model config options.

        Encoding is handled at the `model_dump` level and not here as that should happen only after EVERYTHING has been
        dumped/de-pydantic-ized.

        DEVELOPER WARNING: If default values for nested ProtoModels are not validated and are also not the expected
        model (e.g. Provenance fields are dicts by default), then this function will throw an error because the self
        field becomes the current value, not the model.
        """

        # Get the default return, let the model_dump handle kwarg
        default_result = handler(self)
        exclusion_set = self.model_config["serialize_default_excludes"]
        force_skip_default = self.model_config["force_skip_defaults"]
        output_dict = {}
        # Could handle this with a comprehension, easier this way
        for key, value in default_result.items():
            # Skip defaults on config level (skip default must be on and k has to be unset)
            # Also check against exclusion set on a model_config level
            if (force_skip_default and key not in self.model_fields_set) or key in exclusion_set:
                continue
            output_dict[key] = value
        return output_dict

    def model_dump(self, **kwargs) -> Dict[str, Any]:
        encoding = kwargs.pop("encoding", None)

        # kwargs["exclude"] = (
        #     kwargs.get("exclude", None) or set()
        # ) | self.model_config["serialize_default_excludes"]  # type: ignore
        # kwargs.setdefault("exclude_unset", self.model_config["serialize_skip_defaults"])  # type: ignore
        # if self.model_config["force_skip_defaults"]:  # type: ignore
        #     kwargs["exclude_unset"] = True

        # Model config defaults will be handled in the @model_serializer function
        # The @model_serializer function will be called AFTER this is called
        data = super().model_dump(**kwargs)

        if encoding is None:
            return data
        elif encoding == "json":
            return json.loads(serialize(data, encoding="json"))
        else:
            raise KeyError(f"Unknown encoding type '{encoding}', valid encoding types: 'json'.")

    def serialize(
        self,
        encoding: str,
        *,
        include: Optional[Set[str]] = None,
        exclude: Optional[Set[str]] = None,
        exclude_unset: Optional[bool] = None,
        exclude_defaults: Optional[bool] = None,
        exclude_none: Optional[bool] = None,
    ) -> Union[bytes, str]:
        r"""Generates a serialized representation of the model

        Parameters
        ----------
        encoding
            The serialization type, available types are: {'json', 'json-ext', 'msgpack-ext'}
        include
            Fields to be included in the serialization.
        exclude
            Fields to be excluded in the serialization.
        exclude_unset
            If True, skips fields that have default values provided.
        exclude_defaults
            If True, skips fields that have set or defaulted values equal to the default.
        exclude_none
            If True, skips fields that have value ``None``.

        Returns
        -------
        ~typing.Union[bytes, str]
            The serialized model.
        """

        kwargs = {}
        if include:
            kwargs["include"] = include
        if exclude:
            kwargs["exclude"] = exclude
        if exclude_unset:
            kwargs["exclude_unset"] = exclude_unset
        if exclude_defaults:
            kwargs["exclude_defaults"] = exclude_defaults
        if exclude_none:
            kwargs["exclude_none"] = exclude_none

        data = self.model_dump(**kwargs)

        return serialize(data, encoding=encoding)

    # UNCOMMENT IF NEEDED FOR UPGRADE REDO!!!
    def json(self, **kwargs):
        # Alias JSON here from BaseModel to reflect dict changes
        warnings.warn("The `json` method is deprecated; use `model_dump_json` instead.", DeprecationWarning)
        return self.model_dump_json(**kwargs)

    def model_dump_json(self, **kwargs):
        return self.serialize("json", **kwargs)

    def compare(self, other: Union["ProtoModel", BaseModel], **kwargs) -> bool:
        r"""Compares the current object to the provided object recursively.

        Parameters
        ----------
        other
            The model to compare to.
        **kwargs
            Additional kwargs to pass to :func:`~qcelemental.compare_recursive`.

        Returns
        -------
        bool
            True if the objects match.
        """
        from ...testing import compare_recursive

        return compare_recursive(self, other, **kwargs)

    @classmethod
    def _merge_config_with(cls, *args, **kwargs):
        """
        Helper function to merge protomodel's config with other args

        args: other ExtendedConfigDict instances or equivalent dicts
        kwargs: Keys to add into the dictionary raw
        """
        output_dict = {**cls.model_config}
        for arg in args:  # Handle other dicts first
            output_dict.update(arg)
        # Update any specific keywords
        output_dict.update(kwargs)
        # Finally, check against the Extended Config Dict
        return ExtendedConfigDict(**output_dict)


def check_convertible_version(ver: int, error: str):
    """Standardize the version/error handling for v2 QCSchema."""

    if ver == 1:
        return True
    elif ver == 2:
        return "self"
    else:
        raise ValueError(f"QCSchema {error} version={ver} does not exist for conversion.")


qcschema_draft = "http://json-schema.org/draft-04/schema#"
