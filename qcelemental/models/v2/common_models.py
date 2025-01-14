from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, Literal, Optional, Sequence, Tuple, Union

import numpy as np
from pydantic import Field, field_validator

from .basemodels import ProtoModel, qcschema_draft
from .basis_set import BasisSet

if TYPE_CHECKING:
    ReprArgs = Sequence[Tuple[Optional[str], Any]]


def provenance_json_schema_extra(schema, model):
    schema["$schema"] = qcschema_draft


class Provenance(ProtoModel):
    """Provenance information."""

    creator: str = Field(..., description="The name of the program, library, or person who created the object.")
    version: str = Field(
        "",
        description="The version of the creator, blank otherwise. "
        "This should be sortable by the very broad `PEP 440 <https://www.python.org/dev/peps/pep-0440/>`_.",
    )
    routine: str = Field("", description="The name of the routine or function within the creator, blank otherwise.")

    model_config = ProtoModel._merge_config_with(
        canonical_repr=True, json_schema_extra=provenance_json_schema_extra, extra="allow"
    )


class Model(ProtoModel):
    """The computational molecular sciences model to run."""

    method: str = Field(  # type: ignore
        ...,
        description="The quantum chemistry method to evaluate (e.g., B3LYP, PBE, ...). "
        "For MM, name of the force field.",
    )
    basis: Optional[Union[str, BasisSet]] = Field(  # type: ignore
        None,
        description="The quantum chemistry basis set to evaluate (e.g., 6-31g, cc-pVDZ, ...). Can be ``None`` for "
        "methods without basis sets. For molecular mechanics, name of the atom-typer.",
    )

    # basis_spec: BasisSpec = None  # This should be exclusive with basis, but for now will be omitted
    model_config = ProtoModel._merge_config_with(canonical_repr=True, extra="allow")


class DriverEnum(str, Enum):
    """Allowed computation driver values."""

    energy = "energy"
    gradient = "gradient"
    hessian = "hessian"
    properties = "properties"

    def derivative_int(self):
        egh = ["energy", "gradient", "hessian", "third", "fourth", "fifth"]
        if self == "properties":
            return 0
        else:
            return egh.index(self)
