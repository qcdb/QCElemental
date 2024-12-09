from enum import Enum
from typing import TYPE_CHECKING, Annotated, Any, Dict, List, Literal, Optional, Tuple, Union

from pydantic import Field, conlist, constr, field_validator

from ...util import provenance_stamp
from .basemodels import ExtendedConfigDict, ProtoModel
from .common_models import (
    ComputeError,
    DriverEnum,
    Model,
    Provenance,
    check_convertible_version,
    qcschema_torsion_drive_input_default,
    qcschema_torsion_drive_output_default,
)
from .molecule import Molecule
from .results import AtomicResult, AtomicResultProperties, AtomicSpecification
from .results import AtomicSpecification as QCInputSpecification  # TODO Undo after TD
from .types import Array

if TYPE_CHECKING:
    from .common_models import ReprArgs


# ====  Protocols  ==============================================================


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

    schema_name: Literal["qcschema_optimization_protocols"] = "qcschema_optimization_protocols"
    trajectory: TrajectoryProtocolEnum = Field(
        TrajectoryProtocolEnum.all, description=str(TrajectoryProtocolEnum.__doc__)
    )

    model_config = ExtendedConfigDict(force_skip_defaults=True)


# ====  Inputs (Kw/Spec/In)  ====================================================

OptSubSpecs = Annotated[
    Union[AtomicSpecification],  # , ManyBodySpecification],
    Field(
        discriminator="schema_name",
        description="A directive to compute a gradient. Either an ordinary atomic/single-point or a many-body spec.",
    ),
]

OptSubProps = Annotated[
    Union[AtomicResultProperties],  # , ManyBodyProperties],
    Field(
        discriminator="schema_name",
        description="An abridged single-geometry property set. Either an ordinary atomic/single-point or a many-body properties.",
    ),
]

OptSubRes = Annotated[
    Union[AtomicResult],  # ManyBodyResult],
    Field(
        discriminator="schema_name",
        description="A single-geometry result. Either an ordinary atomic/single-point or a many-body result.",
    ),
]


class OptimizationSpecification(ProtoModel):
    """Specification for how to run a geometry optimization."""

    schema_name: Literal["qcschema_optimization_specification"] = "qcschema_optimization_specification"
    # schema_version: Literal[2] = Field(
    #     2,
    #     description="The version number of ``schema_name`` to which this model conforms.",
    # )

    # right default for program?
    program: str = Field(
        "", description="Optimizer CMS code / QCEngine procedure to run the geometry optimization with."
    )
    keywords: Dict[str, Any] = Field({}, description="The optimization specific keywords to be used.")
    protocols: OptimizationProtocols = Field(OptimizationProtocols(), description=str(OptimizationProtocols.__doc__))
    extras: Dict[str, Any] = Field(
        {},
        description="Additional information to bundle with the computation. Use for schema development and scratch space.",
    )
    specification: AtomicSpecification = Field(
        ..., description="The specification for how to run gradients for the optimization."
    )

    @field_validator("program")
    @classmethod
    def _check_procedure(cls, v):
        return v.lower()

    def convert_v(
        self, target_version: int, /
    ) -> Union["qcelemental.models.v1.OptimizationSpecification", "qcelemental.models.v2.OptimizationSpecification"]:
        """Convert to instance of particular QCSchema version."""
        import qcelemental as qcel

        if check_convertible_version(target_version, error="OptimizationSpecification") == "self":
            return self

        loss_store = {}
        dself = self.model_dump()
        if target_version == 1:
            dself["procedure"] = dself.pop("program")
            dself["keywords"]["program"] = dself["specification"].pop("program")

            loss_store["extras"] = dself.pop("extras")
            loss_store["specification"] = dself.pop("specification")

            # if loss_store:
            #     dself["extras"]["_qcsk_conversion_loss"] = loss_store

            self_vN = qcel.models.v1.OptimizationSpecification(**dself)
        else:
            assert False, target_version

        return self_vN


class OptimizationInput(ProtoModel):
    """QCSchema input directive for geometry optimization."""

    id: Optional[str] = None
    schema_name: Literal["qcschema_optimization_input"] = "qcschema_optimization_input"
    schema_version: Literal[2] = Field(
        2,
        description="The version number of ``schema_name`` to which this model conforms.",
    )

    specification: OptimizationSpecification = Field(..., description=str(OptimizationSpecification.__doc__))
    initial_molecule: Molecule = Field(..., description="The starting molecule for the geometry optimization.")

    provenance: Provenance = Field(Provenance(**provenance_stamp(__name__)), description=str(Provenance.__doc__))

    def __repr_args__(self) -> "ReprArgs":
        return [
            ("model", self.specification.specification.model.model_dump()),
            ("molecule_hash", self.initial_molecule.get_hash()[:7]),
        ]

    @field_validator("schema_version", mode="before")
    def _version_stamp(cls, v):
        return 2

    def convert_v(
        self, target_version: int, /
    ) -> Union["qcelemental.models.v1.OptimizationInput", "qcelemental.models.v2.OptimizationInput"]:
        """Convert to instance of particular QCSchema version."""
        import qcelemental as qcel

        if check_convertible_version(target_version, error="OptimizationInput") == "self":
            return self

        dself = self.model_dump()
        if target_version == 1:
            dself["extras"] = dself["specification"].pop("extras")
            dself["protocols"] = dself["specification"].pop("protocols")
            dself["keywords"] = dself["specification"].pop("keywords")

            dself["input_specification"] = self.specification.specification.convert_v(target_version)
            dself["keywords"]["program"] = dself["specification"]["specification"].pop("program")
            dself["specification"].pop("specification")
            dself["specification"].pop("schema_name")

            opt_program = dself["specification"].pop("program")
            assert not dself["specification"], dself["specification"]
            dself.pop("specification")  # now empty

            self_vN = qcel.models.v1.OptimizationInput(**dself)
        else:
            assert False, target_version

        return self_vN


# ====  Properties  =============================================================


class OptimizationProperties(ProtoModel):
    r"""
    Named properties of geometry optimization computations following the MolSSI QCSchema.
    """

    schema_name: Literal["qcschema_optimization_properties"] = Field(
        "qcschema_optimization_properties",
        description=f"The QCSchema specification to which this model conforms.",
    )
    # schema_version: Literal[2] = Field(
    #     2,
    #     description="The version number of :attr:`~qcelemental.models.OptimizationProperties.schema_name` to which this model conforms.",
    # )

    # ========  Calcinfo  =======================================================
    # ========  Canonical  ======================================================

    nuclear_repulsion_energy: Optional[float] = Field(None, description="The nuclear repulsion energy.")

    return_energy: Optional[float] = Field(
        None,
        description=f"The energy of the final optimized molecule. Always available. Identical to the final :attr:`~qcelemental.models.OptimizationResult.trajectory_properties.return_energy`.",
        json_schema_extra={"units": "E_h"},
    )

    return_gradient: Optional[Array[float]] = Field(
        None,
        description=f"The gradient of the final optimized molecule. Always available. Identical to :attr:`~qcelemental.models.OptimizationResult.trajectory_properties.return_gradient`.",
        json_schema_extra={"units": "E_h/a0", "shape": ["nat", 3]},
    )

    optimization_iterations: Optional[int] = Field(
        None, description="The number of geometry iterations taken before convergence."
    )

    model_config = ProtoModel._merge_config_with(force_skip_defaults=True)


# ====  Results  ================================================================


class OptimizationResult(ProtoModel):
    """QCSchema results model for geometry optimization."""

    schema_name: Literal["qcschema_optimization_output"] = "qcschema_optimization_output"  # TODO _result?
    schema_version: Literal[2] = Field(
        2,
        description="The version number of ``schema_name`` to which this model conforms.",
    )
    id: Optional[str] = Field(None, description="The optional ID for the computation.")
    input_data: OptimizationInput = Field(..., description=str(OptimizationInput.__doc__))

    final_molecule: Optional[Molecule] = Field(..., description="The final molecule of the geometry optimization.")
    trajectory_results: List[AtomicResult] = Field(
        ..., description="A list of ordered Result objects for each step in the optimization."
    )
    trajectory_properties: List[AtomicResultProperties] = Field(
        ..., description="A list of ordered energies and other properties for each step in the optimization."
    )

    stdout: Optional[str] = Field(None, description="The standard output of the program.")
    stderr: Optional[str] = Field(None, description="The standard error of the program.")

    success: Literal[True] = Field(
        True, description="The success of a given programs execution. If False, other fields may be blank."
    )
    provenance: Provenance = Field(..., description=str(Provenance.__doc__))

    # TODO add native_files if any opt programs supply extra files or need an input file?
    # native_files: Dict[str, Any] = Field({}, description="DSL files.")

    properties: OptimizationProperties = Field(..., description=str(OptimizationProperties.__doc__))

    extras: Dict[str, Any] = Field(
        {},
        description="Additional information to bundle with the computation. Use for schema development and scratch space.",
    )

    @field_validator("trajectory_results")
    @classmethod
    def _trajectory_protocol(cls, v, info):
        # Do not propogate validation errors
        if "input_data" not in info.data:
            raise ValueError("Input_data was not properly formed.")

        keep_enum = info.data["input_data"].specification.protocols.trajectory
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

    @field_validator("schema_version", mode="before")
    def _version_stamp(cls, v):
        return 2

    def convert_v(
        self, target_version: int, /
    ) -> Union["qcelemental.models.v1.OptimizationResult", "qcelemental.models.v2.OptimizationResult"]:
        """Convert to instance of particular QCSchema version."""
        import qcelemental as qcel

        if check_convertible_version(target_version, error="OptimizationResult") == "self":
            return self

        dself = self.model_dump()
        if target_version == 1:
            trajectory_class = self.trajectory_results[0].__class__

            # for input_data, work from model, not dict, to use convert_v
            dself.pop("input_data")
            input_data = self.input_data.convert_v(1).model_dump()  # exclude_unset=True, exclude_none=True

            dself.pop("properties")  # new in v2
            dself["trajectory"] = [
                trajectory_class(**atres).convert_v(target_version) for atres in dself["trajectory_results"]
            ]
            dself.pop("trajectory_results")
            dself["energies"] = [atprop.pop("return_energy", None) for atprop in dself["trajectory_properties"]]
            dself.pop("trajectory_properties")

            dself["extras"] = {**input_data.pop("extras", {}), **dself.pop("extras", {})}  # merge
            dself = {**input_data, **dself}

            self_vN = qcel.models.v1.OptimizationResult(**dself)
        else:
            assert False, target_version

        return self_vN


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


class TorsionDriveInput(ProtoModel):
    """Inputs for running a torsion drive.

    Notes
    -----
    * This class is still provisional and may be subject to removal and re-design.
    """

    schema_name: constr(
        strip_whitespace=True, pattern=qcschema_torsion_drive_input_default
    ) = qcschema_torsion_drive_input_default  # type: ignore
    schema_version: Literal[2] = 2

    keywords: TDKeywords = Field(..., description="The torsion drive specific keywords to be used.")
    extras: Dict[str, Any] = Field({}, description="Extra fields that are not part of the schema.")

    input_specification: QCInputSpecification = Field(..., description=str(QCInputSpecification.__doc__))
    initial_molecule: conlist(item_type=Molecule, min_length=1) = Field(
        ..., description="The starting molecule(s) for the torsion drive."
    )

    optimization_spec: OptimizationSpecification = Field(
        ..., description="Settings to use for optimizations at each grid angle."
    )

    provenance: Provenance = Field(Provenance(**provenance_stamp(__name__)), description=str(Provenance.__doc__))

    @field_validator("input_specification")
    @classmethod
    def _check_input_specification(cls, value):
        assert value.driver == DriverEnum.gradient, "driver must be set to gradient"
        return value

    @field_validator("schema_version", mode="before")
    def _version_stamp(cls, v):
        return 2

    def convert_v(
        self, target_version: int, /
    ) -> Union["qcelemental.models.v1.TorsionDriveInput", "qcelemental.models.v2.TorsionDriveInput"]:
        """Convert to instance of particular QCSchema version."""
        import qcelemental as qcel

        if check_convertible_version(target_version, error="TorsionDriveInput") == "self":
            return self

        dself = self.model_dump()
        if target_version == 1:
            if dself["optimization_spec"].pop("extras", None):
                pass

            self_vN = qcel.models.v1.TorsionDriveInput(**dself)
        else:
            assert False, target_version

        return self_vN


class TorsionDriveResult(TorsionDriveInput):
    """Results from running a torsion drive.

    Notes
    -----
    * This class is still provisional and may be subject to removal and re-design.
    """

    schema_name: constr(
        strip_whitespace=True, pattern=qcschema_torsion_drive_output_default
    ) = qcschema_torsion_drive_output_default  # type: ignore
    schema_version: Literal[2] = 2

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

    success: Literal[True] = Field(
        True, description="The success of a given programs execution. If False, other fields may be blank."
    )
    provenance: Provenance = Field(..., description=str(Provenance.__doc__))

    @field_validator("schema_version", mode="before")
    def _version_stamp(cls, v):
        return 2

    def convert_v(
        self, target_version: int, /
    ) -> Union["qcelemental.models.v1.TorsionDriveResult", "qcelemental.models.v2.TorsionDriveResult"]:
        """Convert to instance of particular QCSchema version."""
        import qcelemental as qcel

        if check_convertible_version(target_version, error="TorsionDriveResult") == "self":
            return self

        dself = self.model_dump()
        if target_version == 1:
            opthist_class = next(iter(self.optimization_history.values()))[0].__class__

            if dself["optimization_spec"].pop("extras", None):
                pass

            dself["optimization_history"] = {
                k: [opthist_class(**res).convert_v(target_version) for res in lst]
                for k, lst in dself["optimization_history"].items()
            }

            self_vN = qcel.models.v1.TorsionDriveResult(**dself)
        else:
            assert False, target_version

        return self_vN
