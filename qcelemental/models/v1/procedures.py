from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Tuple, Union

from pydantic.v1 import Field, conlist, constr, validator

from ...util import provenance_stamp
from .basemodels import ProtoModel, check_convertible_version
from .basis import BasisSet
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
    from pydantic.v1.typing import ReprArgs

    import qcelemental


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

    def convert_v(
        self, target_version: int, /
    ) -> Union["qcelemental.models.v1.OptimizationProtocols", "qcelemental.models.v2.OptimizationProtocols"]:
        """Convert to instance of particular QCSchema version."""
        import qcelemental as qcel

        if check_convertible_version(target_version, error="OptimizationProtocols") == "self":
            return self

        dself = self.dict()
        if target_version == 2:
            # serialization is compact, so use model to assure value
            dself.pop("trajectory", None)
            dself["trajectory_results"] = self.trajectory.value

            self_vN = qcel.models.v2.OptimizationProtocols(**dself)
        else:
            assert False, target_version

        return self_vN


class QCInputSpecification(ProtoModel):
    """
    A compute description for energy, gradient, and Hessian computations used in a geometry optimization.
    """

    schema_name: Literal["qcschema_input"] = Field(qcschema_input_default)
    schema_version: Literal[1] = 1

    driver: DriverEnum = Field(DriverEnum.gradient, description=str(DriverEnum.__doc__))
    model: Model = Field(..., description=str(Model.__doc__))
    keywords: Dict[str, Any] = Field({}, description="The program specific keywords to be used.")

    extras: Dict[str, Any] = Field(
        {},
        description="Additional information to bundle with the computation. Use for schema development and scratch space.",
    )

    def convert_v(
        self, target_version: int, /
    ) -> Union["qcelemental.models.v1.QCInputSpecification", "qcelemental.models.v2.AtomicSpecification"]:
        """Convert to instance of particular QCSchema version."""
        import qcelemental as qcel

        if check_convertible_version(target_version, error="QCInputSpecification") == "self":
            return self

        dself = self.dict()
        if target_version == 2:
            dself.pop("schema_name")
            dself.pop("schema_version")

            # TODO consider Model.convert_v
            model = dself.pop("model")
            if isinstance(self.model.basis, BasisSet):
                model["basis"] = self.model.basis.convert_v(target_version)
            dself["model"] = model

            self_vN = qcel.models.v2.AtomicSpecification(**dself)
        else:
            assert False, target_version

        return self_vN


class OptimizationInput(ProtoModel):
    """QCSchema input directive for geometry optimization."""

    id: Optional[str] = None
    hash_index: Optional[str] = None
    schema_name: constr(  # type: ignore
        strip_whitespace=True, regex=qcschema_optimization_input_default
    ) = qcschema_optimization_input_default
    schema_version: Literal[1] = 1

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

    def convert_v(
        self, target_version: int, /
    ) -> Union["qcelemental.models.v1.OptimizationInput", "qcelemental.models.v2.OptimizationInput"]:
        """Convert to instance of particular QCSchema version."""
        import qcelemental as qcel

        if check_convertible_version(target_version, error="OptimizationInput") == "self":
            return self

        dself = self.dict()
        if target_version == 2:
            dself.pop("schema_version")  # changed in v2
            dself.pop("hash_index", None)  # no longer used, so dropped in v2

            dself["initial_molecule"] = self.initial_molecule.convert_v(target_version)

            spec = {}
            spec["extras"] = dself.pop("extras")
            dself.pop("protocols")
            spec["protocols"] = self.protocols.convert_v(target_version).model_dump()
            spec["specification"] = self.input_specification.convert_v(target_version).model_dump()
            dself.pop("input_specification")
            spec["specification"]["program"] = dself["keywords"].pop(
                "program", ""
            )  # "" is when there's an implcit program, like nwchemopt
            spec["keywords"] = dself.pop("keywords")
            dself["specification"] = spec

            self_vN = qcel.models.v2.OptimizationInput(**dself)
        else:
            assert False, target_version

        return self_vN


class OptimizationResult(OptimizationInput):
    """QCSchema results model for geometry optimization."""

    schema_name: constr(  # type: ignore
        strip_whitespace=True, regex=qcschema_optimization_output_default
    ) = qcschema_optimization_output_default
    # Note no schema_version: Literal[1] = Field(1) b/c inherited from OptimizationInput

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

    def convert_v(
        self,
        target_version: int,
        /,
        *,
        external_input_data: Optional[Union[Dict[str, Any], "OptimizationInput"]] = None,
    ) -> Union["qcelemental.models.v1.OptimizationResult", "qcelemental.models.v2.OptimizationResult"]:
        """Convert to instance of particular QCSchema version.

        Parameters
        ----------
        target_version
            The version to convert to.
        external_input_data
            Since self contains data merged from input, this allows passing in the original input, particularly for `extras` fields.
            Can be model or dictionary and should be *already* converted to target_version.
            Replaces ``input_data`` field entirely (not merges with extracts from self) and w/o consistency checking.

        Returns
        -------
        OptimizationResult
            Returns self (not a copy) if ``target_version`` already satisfied.
            Returns a new OptimizationResult of ``target_version`` otherwise.

        """
        import qcelemental as qcel

        if check_convertible_version(target_version, error="OptimizationResult") == "self":
            return self

        try:
            trajectory_class = self.trajectory[0].__class__
        except IndexError:
            trajectory_class = None

        dself = self.dict()
        if target_version == 2:
            # remove harmless empty error field that v2 won't accept. if populated, pydantic will catch it.
            if not dself.get("error", True):
                dself.pop("error")

            dself.pop("hash_index", None)  # no longer used, so dropped in v2
            dself.pop("schema_name")  # changed in v2
            dself.pop("schema_version")  # changed in v2

            v1_input_data = {
                k: dself.pop(k)
                for k in list(dself.keys())
                if k in ["initial_molecule", "protocols", "keywords", "input_specification"]
            }
            # sep any merged extras known to belong to input
            v1_input_data["extras"] = {k: dself["extras"].pop(k) for k in list(dself["extras"].keys()) if k in []}
            v2_input_data = qcel.models.v1.OptimizationInput(**v1_input_data).convert_v(target_version)

            # any input provenance has been overwritten
            # if dself["id"]:
            #     input_data["id"] = dself["id"]  # in/out should likely match

            if external_input_data:
                # Note: overwriting with external, not updating. reconsider?
                if isinstance(external_input_data, dict):
                    if isinstance(external_input_data["specification"], dict):
                        in_extras = external_input_data["specification"].get("extras", {})
                    else:
                        in_extras = external_input_data["specification"].extras
                else:
                    in_extras = external_input_data.specification.extras
                    optsubptcl = external_input_data.specification.specification.protocols
                dself["extras"] = {k: v for k, v in dself["extras"].items() if (k, v) not in in_extras.items()}
                dself["input_data"] = external_input_data
            else:
                dself["input_data"] = v2_input_data
                optsubptcl = None

            dself["final_molecule"] = self.final_molecule.convert_v(target_version)
            dself["properties"] = {
                "nuclear_repulsion_energy": self.final_molecule.nuclear_repulsion_energy(),
                "return_energy": dself["energies"][-1],
                "optimization_iterations": len(dself["energies"]),
            }
            if dself.get("trajectory", []):
                if (
                    last_grad := dself["trajectory"][-1].get("properties", {}).get("return_gradient", None)
                ) is not None:
                    dself["properties"]["return_gradient"] = last_grad
            if len(dself.get("trajectory", [])) == len(dself["energies"]):
                dself["trajectory_properties"] = [
                    res["properties"] for res in dself["trajectory"]
                ]  # TODO filter to key keys
            dself["trajectory_properties"] = [{"return_energy": ene} for ene in dself["energies"]]
            dself.pop("energies")

            dself["trajectory_results"] = [
                trajectory_class(**atres).convert_v(target_version, external_protocols=optsubptcl)
                for atres in dself["trajectory"]
            ]
            dself.pop("trajectory")

            self_vN = qcel.models.v2.OptimizationResult(**dself)
        else:
            assert False, target_version

        return self_vN


class OptimizationSpecification(ProtoModel):
    """
    A specification for how a geometry optimization should be performed **inside** of
    another procedure.

    Notes
    -----
    * This class is still provisional and may be subject to removal and re-design.
    """

    schema_name: constr(strip_whitespace=True, regex="qcschema_optimization_specification") = "qcschema_optimization_specification"  # type: ignore
    schema_version: Literal[1] = 1

    procedure: str = Field(..., description="Optimization procedure to run the optimization with.")
    keywords: Dict[str, Any] = Field({}, description="The optimization specific keywords to be used.")
    protocols: OptimizationProtocols = Field(OptimizationProtocols(), description=str(OptimizationProtocols.__doc__))

    @validator("procedure")
    def _check_procedure(cls, v):
        return v.lower()

    # NOTE: def convert_v() is missing deliberately. Because the v1 schema has a different role only for
    #   TorsionDrive, it doesn't have nearly enough info to create a v2 schema.


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

    schema_name: constr(strip_whitespace=True, regex=qcschema_torsion_drive_input_default) = qcschema_torsion_drive_input_default  # type: ignore
    schema_version: Literal[1] = 1

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

    def convert_v(
        self, target_version: int, /
    ) -> Union["qcelemental.models.v1.TorsionDriveInput", "qcelemental.models.v2.TorsionDriveInput"]:
        """Convert to instance of particular QCSchema version."""
        import qcelemental as qcel

        if check_convertible_version(target_version, error="TorsionDriveInput") == "self":
            return self

        dself = self.dict()
        # dself = self.model_dump(exclude_unset=True, exclude_none=True)
        if target_version == 2:
            gradspec = self.input_specification.convert_v(target_version).model_dump()
            gradspec["program"] = dself["optimization_spec"]["keywords"].pop("program", "")
            dself.pop("input_specification")

            optspec = {}
            optspec["program"] = dself["optimization_spec"].pop("procedure")
            dself["optimization_spec"].pop("protocols")
            optspec["protocols"] = self.optimization_spec.protocols.convert_v(target_version).model_dump()
            optspec["keywords"] = dself["optimization_spec"].pop("keywords")
            optspec["specification"] = gradspec
            dself["optimization_spec"].pop("schema_name")
            dself["optimization_spec"].pop("schema_version")
            assert not dself["optimization_spec"], dself["optimization_spec"]
            dself.pop("optimization_spec")

            tdspec = {}
            tdspec["program"] = "torsiondrive"
            tdspec["extras"] = dself.pop("extras")
            tdspec["keywords"] = dself.pop("keywords")
            tdspec["protocols"] = {"scan_results": "all"}
            tdspec["specification"] = optspec

            dtop = {}
            dtop["provenance"] = dself.pop("provenance")
            dself.pop("initial_molecule")
            dtop["initial_molecule"] = [mol.convert_v(target_version) for mol in self.initial_molecule]
            dtop["specification"] = tdspec
            dself.pop("schema_name")
            dself.pop("schema_version")
            assert not dself, dself

            self_vN = qcel.models.v2.TorsionDriveInput(**dtop)
        else:
            assert False, target_version

        return self_vN


class TorsionDriveResult(TorsionDriveInput):
    """Results from running a torsion drive.

    Notes
    -----
    * This class is still provisional and may be subject to removal and re-design.
    """

    schema_name: constr(strip_whitespace=True, regex=qcschema_torsion_drive_output_default) = qcschema_torsion_drive_output_default  # type: ignore
    # Note no schema_version: Literal[1] = Field(1) b/c inherited from TorsionDriveInput

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

    def convert_v(
        self, target_version: int, /, *, external_input_data: "TorsionDriveInput" = None
    ) -> Union["qcelemental.models.v1.TorsionDriveResult", "qcelemental.models.v2.TorsionDriveResult"]:
        """Convert to instance of particular QCSchema version."""
        import qcelemental as qcel

        if check_convertible_version(target_version, error="TorsionDriveResult") == "self":
            return self

        dself = self.dict()
        if target_version == 2:
            opthist_class = next(iter(self.optimization_history.values()))[0].__class__
            dtop = {}

            # remove harmless empty error field that v2 won't accept. if populated, pydantic will catch it.
            if not dself.get("error", True):
                dself.pop("error")

            v1_input_data = {
                k: dself.pop(k)
                for k in list(dself.keys())
                if k in ["initial_molecule", "keywords", "optimization_spec", "input_specification"]  # protocols
            }
            # any input provenance has been overwritten
            # sep any merged extras known to belong to input
            v1_input_data["extras"] = {k: dself["extras"].pop(k) for k in list(dself["extras"].keys()) if k in []}
            v2_input_data = qcel.models.v1.TorsionDriveInput(**v1_input_data).convert_v(target_version)

            # if dself["id"]:
            #     input_data["id"] = dself["id"]  # in/out should likely match

            if external_input_data:
                # Note: overwriting with external, not updating. reconsider?
                if isinstance(external_input_data, dict):
                    if isinstance(external_input_data["specification"], dict):
                        in_extras = external_input_data["specification"].get("extras", {})
                    else:
                        in_extras = external_input_data["specification"].extras
                else:
                    in_extras = external_input_data.specification.extras
                dtop["extras"] = {k: v for k, v in dself["extras"].items() if (k, v) not in in_extras.items()}
                dtop["input_data"] = external_input_data
            else:
                dtop["input_data"] = v2_input_data
                dtop["extras"] = dself.pop("extras")

            dtop["provenance"] = dself.pop("provenance")
            dtop["stdout"] = dself.pop("stdout")
            dtop["stderr"] = dself.pop("stderr")
            dtop["success"] = dself.pop("success")
            dtop["final_energies"] = dself.pop("final_energies")
            dself.pop("final_molecules")
            dtop["final_molecules"] = {k: m.convert_v(target_version) for k, m in self.final_molecules.items()}
            dtop["scan_results"] = {
                k: [opthist_class(**res).convert_v(target_version) for res in lst]
                for k, lst in dself["optimization_history"].items()
            }
            dself.pop("optimization_history")
            dself.pop("schema_name")
            dself.pop("schema_version")
            if "error" in dself:
                dtop["error"] = dself.pop("error")  # guaranteed to be fatal
            assert not dself, dself

            self_vN = qcel.models.v2.TorsionDriveResult(**dtop)
        else:
            assert False, target_version

        return self_vN


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
