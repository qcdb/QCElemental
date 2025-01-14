from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Tuple, Union

from pydantic import Field, conlist, field_validator

from ...util import provenance_stamp
from .basemodels import ExtendedConfigDict, ProtoModel, check_convertible_version
from .common_models import DriverEnum, Provenance
from .molecule import Molecule
from .optimization import OptimizationResult, OptimizationSpecification
from .types import Array

if TYPE_CHECKING:
    import qcelemental

    from .common_models import ReprArgs


# ====  Protocols  ==============================================================


class ScanResultsProtocolEnum(str, Enum):
    """
    Which gradient evaluations to keep in an optimization trajectory.
    """

    all = "all"  # use this if instance might be converted to v1
    lowest = "lowest"  # discard any optimizations at each scan point that did not find the lowest energy
    none = "none"


class TorsionDriveProtocols(ProtoModel):
    """
    Protocols regarding the manipulation of a Torsion Drive subcalculation history.
    """

    schema_name: Literal["qcschema_torsion_drive_protocols"] = "qcschema_torsion_drive_protocols"
    scan_results: ScanResultsProtocolEnum = Field(
        ScanResultsProtocolEnum.none, description=str(ScanResultsProtocolEnum.__doc__)
    )

    model_config = ExtendedConfigDict(force_skip_defaults=True)


# ====  Inputs (Kw/Spec/In)  ====================================================


class TorsionDriveKeywords(ProtoModel):
    """
    TorsionDriveRecord options

    Notes
    -----
    * This class is still provisional and may be subject to removal and re-design.
    """

    schema_name: Literal["qcschema_torsion_drive_keywords"] = Field(
        "qcschema_torsion_drive_keywords",
        description=f"The QCSchema specification to which this model conforms.",
    )

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


class TorsionDriveSpecification(ProtoModel):
    """Specification for how to run a torsion drive scan."""

    schema_name: Literal["qcschema_torsion_drive_specification"] = "qcschema_torsion_drive_specification"

    program: str = Field(
        "", description="Torsion Drive CMS code / QCEngine procedure with which to run the torsion scan."
    )
    keywords: TorsionDriveKeywords = Field(..., description="The torsion drive specific keywords to be used.")
    protocols: TorsionDriveProtocols = Field(TorsionDriveProtocols(), description=str(TorsionDriveProtocols.__doc__))
    extras: Dict[str, Any] = Field(
        {},
        description="Additional information to bundle with the computation. Use for schema development and scratch space.",
    )
    specification: OptimizationSpecification = Field(
        ...,
        description="The specification for how to run optimizations for the torsion scan (within this is spec for gradients for the optimization.",
    )

    @field_validator("program")
    @classmethod
    def _check_procedure(cls, v):
        return v.lower()

    # Note: no convert_v() method as TDSpec doesn't have a v1 equivalent


class TorsionDriveInput(ProtoModel):
    """Inputs for running a torsion drive."""

    schema_name: Literal["qcschema_torsion_drive_input"] = "qcschema_torsion_drive_input"
    schema_version: Literal[2] = Field(
        2,
        description="The version number of ``schema_name`` to which this model conforms.",
    )

    id: Optional[str] = None
    initial_molecule: conlist(item_type=Molecule, min_length=1) = Field(
        ..., description="The starting molecule(s) for the torsion drive."
    )

    specification: TorsionDriveSpecification = Field(..., description=str(TorsionDriveSpecification.__doc__))

    provenance: Provenance = Field(Provenance(**provenance_stamp(__name__)), description=str(Provenance.__doc__))

    @field_validator("specification")
    @classmethod
    def _check_input_specification(cls, value, info):
        driver = value.specification.specification.driver

        assert driver == DriverEnum.gradient, "driver must be set to gradient"
        return value

    def convert_v(
        self, target_version: int, /
    ) -> Union["qcelemental.models.v1.TorsionDriveInput", "qcelemental.models.v2.TorsionDriveInput"]:
        """Convert to instance of particular QCSchema version."""
        import qcelemental as qcel

        if check_convertible_version(target_version, error="TorsionDriveInput") == "self":
            return self

        dself = self.model_dump()
        if target_version == 1:
            dself.pop("schema_version")  # changed in v1
            dself.pop("id")  # unused in v1
            dself["extras"] = dself["specification"].pop("extras")
            dself["initial_molecule"] = [m.convert_v(target_version) for m in self.initial_molecule]
            dself["keywords"] = dself["specification"].pop("keywords")
            dself["keywords"].pop("schema_name")  # unused in v1

            dself["optimization_spec"] = self.specification.specification.convert_v(target_version)
            dself["input_specification"] = self.specification.specification.specification.convert_v(target_version)
            dself["specification"].pop("specification")
            dself["specification"].pop("schema_name")

            td_program = dself["specification"].pop("program")
            dself["specification"].pop("protocols")  # lost
            assert not dself["specification"], dself["specification"]
            dself.pop("specification")  # now empty

            self_vN = qcel.models.v1.TorsionDriveInput(**dself)
        else:
            assert False, target_version

        return self_vN


# ====  Properties  =============================================================
# ========  Calcinfo  =======================================================
# ========  Canonical  ======================================================


# ====  Results  ================================================================


class TorsionDriveResult(ProtoModel):
    """Results from running a torsion drive."""

    schema_name: Literal["qcschema_torsion_drive_result"] = "qcschema_torsion_drive_result"
    schema_version: Literal[2] = Field(
        2,
        description="The version number of ``schema_name`` to which this model conforms.",
    )
    id: Optional[str] = Field(None, description="The optional ID for the computation.")
    input_data: TorsionDriveInput = Field(..., description=str(TorsionDriveInput.__doc__))

    # final_energies, final_molecules, optimization_history I'm hoping to refactor into scan_properties and scan_results but need to talk to OpenFF folks
    final_energies: Dict[str, float] = Field(
        ..., description="The final energy at each angle of the TorsionDrive scan."
    )
    final_molecules: Dict[str, Molecule] = Field(
        ..., description="The final molecule at each angle of the TorsionDrive scan."
    )
    scan_results: Dict[str, List[OptimizationResult]] = Field(
        ...,
        description="The map of each angle of the TorsionDrive scan to each optimization computations.",
    )

    stdout: Optional[str] = Field(None, description="The standard output of the program.")
    stderr: Optional[str] = Field(None, description="The standard error of the program.")

    # native_files placeholder for when any td programs supply extra files or need an input file. no protocol at present
    native_files: Dict[str, Any] = Field({}, description="DSL files.")

    # TODO add properties if a set can be collected
    # properties: TorsionDriveProperties = Field(..., description=str(TorsionDriveProperties.__doc__))

    extras: Dict[str, Any] = Field(
        {},
        description="Additional information to bundle with the computation. Use for schema development and scratch space.",
    )

    success: Literal[True] = Field(
        True, description="The success of a given programs execution. If False, other fields may be blank."
    )
    provenance: Provenance = Field(..., description=str(Provenance.__doc__))

    @field_validator("scan_results")
    @classmethod
    def _scan_protocol(cls, v, info):
        # Do not propogate validation errors
        if "input_data" not in info.data:
            raise ValueError("Input_data was not properly formed.")

        keep_enum = info.data["input_data"].specification.protocols.scan_results
        if keep_enum == "all":
            pass
        elif keep_enum == "lowest":
            if not all(len(vv) == 1 for vv in v.values()):
                v_trunc = {}
                for scan_pt, optres_list in v.items():
                    final_energies = [optres.properties.return_energy for optres in optres_list]
                    lowest_energy_idx = final_energies.index(min(final_energies))
                    v_trunc[scan_pt] = [optres_list[lowest_energy_idx]]
                v = v_trunc
        elif keep_enum == "none":
            v = {}
        else:
            raise ValueError(f"Protocol `scan_results:{keep_enum}` is not understood.")

        return v

    def convert_v(
        self, target_version: int, /
    ) -> Union["qcelemental.models.v1.TorsionDriveResult", "qcelemental.models.v2.TorsionDriveResult"]:
        """Convert to instance of particular QCSchema version.

        Notes
        -----
        * Use TorsionDriveProtocols.scan_results=all for full conversion to v1.

        """
        import qcelemental as qcel

        if check_convertible_version(target_version, error="TorsionDriveResult") == "self":
            return self

        dself = self.model_dump()
        if target_version == 1:
            try:
                opthist_class = next(iter(self.scan_results.values()))[0].__class__
            except StopIteration:
                opthist_class = None
            dtop = {}

            # for input_data, work from model, not dict, to use convert_v
            dself.pop("input_data")
            input_data = self.input_data.convert_v(target_version).model_dump()
            input_data.pop("schema_name")  # prevent inheriting

            dtop["final_energies"] = dself.pop("final_energies")
            dself.pop("final_molecules")
            dtop["final_molecules"] = {k: m.convert_v(target_version) for k, m in self.final_molecules.items()}
            dtop["optimization_history"] = {
                k: [opthist_class(**res).convert_v(target_version) for res in lst]
                for k, lst in dself["scan_results"].items()
            }
            dself.pop("scan_results")

            dself.pop("id")  # unused in v1
            dself.pop("native_files")  # new in v2
            dtop["provenance"] = dself.pop("provenance")
            dtop["stdout"] = dself.pop("stdout")
            dtop["stderr"] = dself.pop("stderr")
            dtop["success"] = dself.pop("success")
            dtop["extras"] = {**input_data.pop("extras", {}), **dself.pop("extras", {})}  # merge
            dself.pop("schema_name")  # otherwise merge below uses TDIn schema_name
            dself.pop("schema_version")
            assert not dself, dself

            dtop = {**input_data, **dtop}

            self_vN = qcel.models.v1.TorsionDriveResult(**dtop)
        else:
            assert False, target_version

        return self_vN
