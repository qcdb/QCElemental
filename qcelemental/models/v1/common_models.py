from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, Optional, Union

import numpy as np
from pydantic.v1 import Field

from .basemodels import ProtoModel, check_convertible_version, qcschema_draft
from .basis import BasisSet

if TYPE_CHECKING:
    from pydantic.v1.typing import ReprArgs

    import qcelemental


# Encoders, to be deprecated
ndarray_encoder = {np.ndarray: lambda v: v.flatten().tolist()}


class Provenance(ProtoModel):
    """Provenance information."""

    creator: str = Field(..., description="The name of the program, library, or person who created the object.")
    version: str = Field(
        "",
        description="The version of the creator, blank otherwise. This should be sortable by the very broad `PEP 440 <https://www.python.org/dev/peps/pep-0440/>`_.",
    )
    routine: str = Field("", description="The name of the routine or function within the creator, blank otherwise.")

    class Config(ProtoModel.Config):
        canonical_repr = True
        extra: str = "allow"

        def schema_extra(schema, model):
            schema["$schema"] = qcschema_draft


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

    class Config(ProtoModel.Config):
        canonical_repr = True
        extra: str = "allow"


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


class ComputeError(ProtoModel):
    """Complete description of the error from an unsuccessful program execution."""

    error_type: str = Field(  # type: ignore
        ...,  # Error enumeration not yet strict
        description="The type of error which was thrown. Restrict this field to short classifiers e.g. 'input_error'. Suggested classifiers: https://github.com/MolSSI/QCEngine/blob/master/qcengine/exceptions.py",
    )
    error_message: str = Field(  # type: ignore
        ...,
        description="Text associated with the thrown error. This is often the backtrace, but it can contain additional "
        "information as well.",
    )
    extras: Optional[Dict[str, Any]] = Field(  # type: ignore
        None,
        description="Additional information to bundle with the error.",
    )

    class Config:
        repr_style = ["error_type", "error_message"]

    def __repr_args__(self) -> "ReprArgs":
        return [("error_type", self.error_type), ("error_message", self.error_message)]


class FailedOperation(ProtoModel):
    """Record indicating that a given operation (program, procedure, etc.) has failed and containing the reason and input data which generated the failure."""

    id: str = Field(  # type: ignore
        None,
        description="A unique identifier which links this FailedOperation, often of the same Id of the operation "
        "should it have been successful. This will often be set programmatically by a database such as "
        "Fractal.",
    )
    input_data: Any = Field(  # type: ignore
        None,
        description="The input data which was passed in that generated this failure. This should be the complete "
        "input which when attempted to be run, caused the operation to fail.",
    )
    success: bool = Field(  # type: ignore
        False,
        description="A boolean indicator that the operation failed consistent with the model of successful operations. "
        "Should always be False. Allows programmatic assessment of all operations regardless of if they failed or "
        "succeeded",
    )
    error: ComputeError = Field(  # type: ignore
        ...,
        description="A container which has details of the error that failed this operation. See the "
        ":class:`ComputeError` for more details.",
    )
    extras: Optional[Dict[str, Any]] = Field(  # type: ignore
        {},
        description="Additional information to bundle with the failed operation. Details which pertain specifically "
        "to a thrown error should be contained in the `error` field. See :class:`ComputeError` for details.",
    )

    def __repr_args__(self) -> "ReprArgs":
        return [("error", self.error)]

    def convert_v(
        self, target_version: int, /
    ) -> Union["qcelemental.models.v1.FailedOperation", "qcelemental.models.v2.FailedOperation"]:
        """Convert to instance of particular QCSchema version."""
        import qcelemental as qcel

        if check_convertible_version(target_version, error="FailedOperation") == "self":
            return self

        dself = self.dict()
        if target_version == 2:
            # TODO if FailedOp gets a schema_version, add a validator
            self_vN = qcel.models.v2.FailedOperation(**dself)
        else:
            assert False, target_version

        return self_vN


qcschema_input_default = "qcschema_input"
qcschema_output_default = "qcschema_output"
qcschema_optimization_input_default = "qcschema_optimization_input"
qcschema_optimization_output_default = "qcschema_optimization_output"
qcschema_torsion_drive_input_default = "qcschema_torsion_drive_input"
qcschema_torsion_drive_output_default = "qcschema_torsion_drive_output"
qcschema_molecule_default = "qcschema_molecule"
