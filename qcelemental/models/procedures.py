from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

try:
    from typing import Literal
except ImportError:
    # remove when minimum py38
    from typing_extensions import Literal

try:
    from pydantic.v1 import Field, conlist, constr, validator
except ImportError:  # Will also trap ModuleNotFoundError
    from pydantic import Field, conlist, constr, validator

from ..util import provenance_stamp
from .basemodels import ProtoModel
from .common_models import (
    ComputeError,
    DriverEnum,
    Model,
    Provenance,
    qcschema_input_default,
    qcschema_optimization_input_default,
    qcschema_optimization_output_default,
    qcschema_torsion_drive_input_default,
    qcschema_torsion_drive_output_default,
)
from .molecule import Molecule
from .results import AtomicResult

if TYPE_CHECKING:
    try:
        from pydantic.v1.typing import ReprArgs
    except ImportError:  # Will also trap ModuleNotFoundError
        from pydantic.typing import ReprArgs


class TrajectoryProtocolEnum(str, Enum):
    """
    Which gradient evaluations to keep in an optimization trajectory.
    """

    all = "all"
    initial_and_final = "initial_and_final"
    final = "final"
    none = "none"


class OptimizationProtocols(ProtoModel):
    """
    Protocols regarding the manipulation of a Optimization output data.
    """

    trajectory: TrajectoryProtocolEnum = Field(
        TrajectoryProtocolEnum.all, description=str(TrajectoryProtocolEnum.__doc__)
    )

    class Config:
        force_skip_defaults = True


class QCInputSpecification(ProtoModel):
    """
    A compute description for energy, gradient, and Hessian computations used in a geometry optimization.
    """

    schema_name: Literal["qcschema_input"] = Field(qcschema_input_default)
    schema_version: int = 1

    driver: DriverEnum = Field(DriverEnum.gradient, description=str(DriverEnum.__doc__))
    model: Model = Field(..., description=str(Model.__doc__))
    keywords: Dict[str, Any] = Field({}, description="The program specific keywords to be used.")

    extras: Dict[str, Any] = Field(
        {},
        description="Additional information to bundle with the computation. Use for schema development and scratch space.",
    )


class OptimizationInput(ProtoModel):
    """QCSchema input directive for geometry optimization."""

    id: Optional[str] = None
    hash_index: Optional[str] = None
    schema_name: constr(  # type: ignore
        strip_whitespace=True, regex=qcschema_optimization_input_default
    ) = qcschema_optimization_input_default
    schema_version: int = 1

    keywords: Dict[str, Any] = Field({}, description="The optimization specific keywords to be used.")
    extras: Dict[str, Any] = Field({}, description="Extra fields that are not part of the schema.")
    protocols: OptimizationProtocols = Field(OptimizationProtocols(), description=str(OptimizationProtocols.__doc__))

    input_specification: QCInputSpecification = Field(..., description=str(QCInputSpecification.__doc__))
    initial_molecule: Molecule = Field(..., description="The starting molecule for the geometry optimization.")

    provenance: Provenance = Field(Provenance(**provenance_stamp(__name__)), description=str(Provenance.__doc__))

    def __repr_args__(self) -> "ReprArgs":
        return [
            ("model", self.input_specification.model.dict()),
            ("molecule_hash", self.initial_molecule.get_hash()[:7]),
        ]


class OptimizationResult(OptimizationInput):
    """QCSchema results model for geometry optimization."""

    schema_name: constr(  # type: ignore
        strip_whitespace=True, regex=qcschema_optimization_output_default
    ) = qcschema_optimization_output_default

    final_molecule: Optional[Molecule] = Field(..., description="The final molecule of the geometry optimization.")
    trajectory: List[AtomicResult] = Field(
        ..., description="A list of ordered Result objects for each step in the optimization."
    )
    energies: List[float] = Field(..., description="A list of ordered energies for each step in the optimization.")

    stdout: Optional[str] = Field(None, description="The standard output of the program.")
    stderr: Optional[str] = Field(None, description="The standard error of the program.")

    success: bool = Field(
        ..., description="The success of a given programs execution. If False, other fields may be blank."
    )
    error: Optional[ComputeError] = Field(None, description=str(ComputeError.__doc__))
    provenance: Provenance = Field(..., description=str(Provenance.__doc__))

    @validator("trajectory", each_item=False)
    def _trajectory_protocol(cls, v, values):
        # Do not propogate validation errors
        if "protocols" not in values:
            raise ValueError("Protocols was not properly formed.")

        keep_enum = values["protocols"].trajectory
        if keep_enum == "all":
            pass
        elif keep_enum == "initial_and_final":
            if len(v) != 2:
                v = [v[0], v[-1]]
        elif keep_enum == "final":
            if len(v) != 1:
                v = [v[-1]]
        elif keep_enum == "none":
            v = []
        else:
            raise ValueError(f"Protocol `trajectory:{keep_enum}` is not understood.")

        return v


class OptimizationSpecification(ProtoModel):
    """
    A specification for how a geometry optimization should be performed **inside** of
    another procedure.

    Notes
    -----
    * This class is still provisional and may be subject to removal and re-design.
    """

    schema_name: constr(strip_whitespace=True, regex="qcschema_optimization_specification") = "qcschema_optimization_specification"  # type: ignore
    schema_version: int = 1

    procedure: str = Field(..., description="Optimization procedure to run the optimization with.")
    keywords: Dict[str, Any] = Field({}, description="The optimization specific keywords to be used.")
    protocols: OptimizationProtocols = Field(OptimizationProtocols(), description=str(OptimizationProtocols.__doc__))

    @validator("procedure")
    def _check_procedure(cls, v):
        return v.lower()


class TDKeywords(ProtoModel):
    """
    TorsionDriveRecord options

    Notes
    -----
    * This class is still provisional and may be subject to removal and re-design.
    """

    dihedrals: List[Tuple[int, int, int, int]] = Field(
        ...,
        description="The list of dihedrals to select for the TorsionDrive operation. Each entry is a tuple of integers "
        "of for particle indices.",
    )
    grid_spacing: List[int] = Field(
        ...,
        description="List of grid spacing for dihedral scan in degrees. Multiple values will be mapped to each "
        "dihedral angle.",
    )
    dihedral_ranges: Optional[List[Tuple[int, int]]] = Field(
        None,
        description="A list of dihedral range limits as a pair (lower, upper). "
        "Each range corresponds to the dihedrals in input.",
    )
    energy_decrease_thresh: Optional[float] = Field(
        None,
        description="The threshold of the smallest energy decrease amount to trigger activating optimizations from "
        "grid point.",
    )
    energy_upper_limit: Optional[float] = Field(
        None,
        description="The threshold if the energy of a grid point that is higher than the current global minimum, to "
        "start new optimizations, in unit of a.u. I.e. if energy_upper_limit = 0.05, current global "
        "minimum energy is -9.9 , then a new task starting with energy -9.8 will be skipped.",
    )
    additional_keywords: Dict[str, Any] = Field(
        {}, description="Additional keywords to add to the torsiondrive's optimization runs."
    )


class TorsionDriveInput(ProtoModel):
    """Inputs for running a torsion drive.

    Notes
    -----
    * This class is still provisional and may be subject to removal and re-design.
    """

    schema_name: constr(strip_whitespace=True, regex=qcschema_torsion_drive_input_default) = qcschema_torsion_drive_input_default  # type: ignore
    schema_version: int = 1

    keywords: TDKeywords = Field(..., description="The torsion drive specific keywords to be used.")
    extras: Dict[str, Any] = Field({}, description="Extra fields that are not part of the schema.")

    input_specification: QCInputSpecification = Field(..., description=str(QCInputSpecification.__doc__))
    initial_molecule: conlist(item_type=Molecule, min_items=1) = Field(
        ..., description="The starting molecule(s) for the torsion drive."
    )

    optimization_spec: OptimizationSpecification = Field(
        ..., description="Settings to use for optimizations at each grid angle."
    )

    provenance: Provenance = Field(Provenance(**provenance_stamp(__name__)), description=str(Provenance.__doc__))

    @validator("input_specification")
    def _check_input_specification(cls, value):
        assert value.driver == DriverEnum.gradient, "driver must be set to gradient"
        return value


class TorsionDriveResult(TorsionDriveInput):
    """Results from running a torsion drive.

    Notes
    -----
    * This class is still provisional and may be subject to removal and re-design.
    """

    schema_name: constr(strip_whitespace=True, regex=qcschema_torsion_drive_output_default) = qcschema_torsion_drive_output_default  # type: ignore
    schema_version: int = 1

    final_energies: Dict[str, float] = Field(
        ..., description="The final energy at each angle of the TorsionDrive scan."
    )
    final_molecules: Dict[str, Molecule] = Field(
        ..., description="The final molecule at each angle of the TorsionDrive scan."
    )

    optimization_history: Dict[str, List[OptimizationResult]] = Field(
        ...,
        description="The map of each angle of the TorsionDrive scan to each optimization computations.",
    )

    stdout: Optional[str] = Field(None, description="The standard output of the program.")
    stderr: Optional[str] = Field(None, description="The standard error of the program.")

    success: bool = Field(
        ..., description="The success of a given programs execution. If False, other fields may be blank."
    )
    error: Optional[ComputeError] = Field(None, description=str(ComputeError.__doc__))
    provenance: Provenance = Field(..., description=str(Provenance.__doc__))


def Optimization(*args, **kwargs):
    """QC Optimization Results Schema.

    .. deprecated:: 0.12
       Use :py:func:`qcelemental.models.OptimizationResult` instead.

    """
    from warnings import warn

    warn(
        "Optimization has been renamed to OptimizationResult and will be removed as soon as v0.13.0", DeprecationWarning
    )
    return OptimizationResult(*args, **kwargs)
