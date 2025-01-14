from enum import Enum
from functools import partial
from typing import TYPE_CHECKING, Any, Dict, Literal, Optional, Set, Union

import numpy as np
from pydantic import Field, field_validator

from ...util import provenance_stamp
from .basemodels import ExtendedConfigDict, ProtoModel, check_convertible_version, qcschema_draft
from .basis_set import BasisSet
from .common_models import DriverEnum, Model, Provenance
from .molecule import Molecule
from .types import Array

if TYPE_CHECKING:
    import qcelemental

    from .common_models import ReprArgs


# ====  Properties  =============================================================


class AtomicProperties(ProtoModel):
    r"""
    Named properties of quantum chemistry computations following the MolSSI QCSchema.

    All arrays are stored flat but must be reshapable into the dimensions in attribute ``shape``, with abbreviations as follows:

    * nao: number of atomic orbitals = :attr:`~qcelemental.models.AtomicProperties.calcinfo_nbasis`
    * nmo: number of molecular orbitals = :attr:`~qcelemental.models.AtomicProperties.calcinfo_nmo`
    """

    schema_name: Literal["qcschema_atomic_properties"] = Field(
        "qcschema_atomic_properties", description=(f"The QCSchema specification to which this model conforms.")
    )

    # ========  Calcinfo  =======================================================

    calcinfo_nbasis: Optional[int] = Field(None, description="The number of basis functions for the computation.")
    calcinfo_nmo: Optional[int] = Field(None, description="The number of molecular orbitals for the computation.")
    calcinfo_nalpha: Optional[int] = Field(None, description="The number of alpha electrons in the computation.")
    calcinfo_nbeta: Optional[int] = Field(None, description="The number of beta electrons in the computation.")
    calcinfo_natom: Optional[int] = Field(None, description="The number of atoms in the computation.")

    # ========  Canonical  ======================================================

    nuclear_repulsion_energy: Optional[float] = Field(None, description="The nuclear repulsion energy.")
    return_energy: Optional[float] = Field(
        None,
        description=f"The energy of the requested method, identical to :attr:`~qcelemental.models.AtomicResult.return_result` for :attr:`~qcelemental.models.AtomicInput.driver`\\ =\\ :attr:`~qcelemental.models.DriverEnum.energy` computations.",
    )
    return_gradient: Optional[Array[float]] = Field(
        None,
        description=f"The gradient of the requested method, identical to :attr:`~qcelemental.models.AtomicResult.return_result` for :attr:`~qcelemental.models.AtomicInput.driver`\\ =\\ :attr:`~qcelemental.models.DriverEnum.gradient` computations.",
        json_schema_extra={"units": "E_h/a0"},
    )
    return_hessian: Optional[Array[float]] = Field(
        None,
        description=f"The Hessian of the requested method, identical to :attr:`~qcelemental.models.AtomicResult.return_result` for :attr:`~qcelemental.models.AtomicInput.driver`\\ =\\ :attr:`~qcelemental.models.DriverEnum.hessian` computations.",
        json_schema_extra={"units": "E_h/a0^2"},
    )

    # ========  Method data  ====================================================

    # SCF Keywords
    scf_one_electron_energy: Optional[float] = Field(
        None,
        description="The one-electron (core Hamiltonian) energy contribution to the total SCF energy.",
        json_schema_extra={"units": "E_h"},
    )
    scf_two_electron_energy: Optional[float] = Field(
        None,
        description="The two-electron energy contribution to the total SCF energy.",
        json_schema_extra={"units": "E_h"},
    )
    scf_vv10_energy: Optional[float] = Field(
        None,
        description="The VV10 functional energy contribution to the total SCF energy.",
        json_schema_extra={"units": "E_h"},
    )
    scf_xc_energy: Optional[float] = Field(
        None,
        description="The functional (XC) energy contribution to the total SCF energy.",
        json_schema_extra={"units": "E_h"},
    )
    scf_dispersion_correction_energy: Optional[float] = Field(
        None,
        description="The dispersion correction appended to an underlying functional when a DFT-D method is requested.",
        json_schema_extra={"units": "E_h"},
    )
    scf_dipole_moment: Optional[Array[float]] = Field(
        None,
        description="The SCF X, Y, and Z dipole components",
        json_schema_extra={"units": "e a0"},
    )
    scf_quadrupole_moment: Optional[Array[float]] = Field(
        None,
        description="The quadrupole components (redundant; 6 unique).",
        json_schema_extra={"units": "e a0^2", "shape": [3, 3]},
    )
    scf_total_energy: Optional[float] = Field(
        None,
        description="The total electronic energy of the SCF stage of the calculation.",
        json_schema_extra={"units": "E_h"},
    )
    scf_total_gradient: Optional[Array[float]] = Field(
        None,
        description="The total electronic gradient of the SCF stage of the calculation.",
        json_schema_extra={"units": "E_h/a0"},
    )
    scf_total_hessian: Optional[Array[float]] = Field(
        None,
        description="The total electronic Hessian of the SCF stage of the calculation.",
        json_schema_extra={"units": "E_h/a0^2"},
    )
    scf_iterations: Optional[int] = Field(None, description="The number of SCF iterations taken before convergence.")

    # MP2 Keywords
    mp2_same_spin_correlation_energy: Optional[float] = Field(
        None,
        description="The portion of MP2 doubles correlation energy from same-spin (i.e. triplet) correlations, without any user scaling.",
        json_schema_extra={"units": "E_h"},
    )
    mp2_opposite_spin_correlation_energy: Optional[float] = Field(
        None,
        description="The portion of MP2 doubles correlation energy from opposite-spin (i.e. singlet) correlations, without any user scaling.",
        json_schema_extra={"units": "E_h"},
    )
    mp2_singles_energy: Optional[float] = Field(
        None,
        description="The singles portion of the MP2 correlation energy. Zero except in ROHF.",
        json_schema_extra={"units": "E_h"},
    )
    mp2_doubles_energy: Optional[float] = Field(
        None,
        description="The doubles portion of the MP2 correlation energy including same-spin and opposite-spin correlations.",
        json_schema_extra={"units": "E_h"},
    )
    mp2_correlation_energy: Optional[float] = Field(
        None,
        description="The MP2 correlation energy.",
        json_schema_extra={"units": "E_h"},
    )
    mp2_total_energy: Optional[float] = Field(
        None,
        description="The total MP2 energy (MP2 correlation energy + HF energy).",
        json_schema_extra={"units": "E_h"},
    )
    mp2_dipole_moment: Optional[Array[float]] = Field(
        None,
        description="The MP2 X, Y, and Z dipole components.",
        json_schema_extra={"shape": [3], "units": "e a0"},
    )

    # CCSD Keywords
    ccsd_same_spin_correlation_energy: Optional[float] = Field(
        None,
        description="The portion of CCSD doubles correlation energy from same-spin (i.e. triplet) correlations, without any user scaling.",
        json_schema_extra={"units": "E_h"},
    )
    ccsd_opposite_spin_correlation_energy: Optional[float] = Field(
        None,
        description="The portion of CCSD doubles correlation energy from opposite-spin (i.e. singlet) correlations, without any user scaling.",
        json_schema_extra={"units": "E_h"},
    )
    ccsd_singles_energy: Optional[float] = Field(
        None,
        description="The singles portion of the CCSD correlation energy. Zero except in ROHF.",
        json_schema_extra={"units": "E_h"},
    )
    ccsd_doubles_energy: Optional[float] = Field(
        None,
        description="The doubles portion of the CCSD correlation energy including same-spin and opposite-spin correlations.",
        json_schema_extra={"units": "E_h"},
    )
    ccsd_correlation_energy: Optional[float] = Field(
        None,
        description="The CCSD correlation energy.",
        json_schema_extra={"units": "E_h"},
    )
    ccsd_total_energy: Optional[float] = Field(
        None,
        description="The total CCSD energy (CCSD correlation energy + HF energy).",
        json_schema_extra={"units": "E_h"},
    )
    ccsd_dipole_moment: Optional[Array[float]] = Field(
        None,
        description="The CCSD X, Y, and Z dipole components.",
        json_schema_extra={"shape": [3], "units": "e a0"},
    )
    ccsd_iterations: Optional[int] = Field(None, description="The number of CCSD iterations taken before convergence.")

    # CCSD(T) keywords
    ccsd_prt_pr_correlation_energy: Optional[float] = Field(
        None,
        description="The CCSD(T) correlation energy.",
        json_schema_extra={"units": "E_h"},
    )
    ccsd_prt_pr_total_energy: Optional[float] = Field(
        None,
        description="The total CCSD(T) energy (CCSD(T) correlation energy + HF energy).",
        json_schema_extra={"units": "E_h"},
    )
    ccsd_prt_pr_dipole_moment: Optional[Array[float]] = Field(
        None,
        description="The CCSD(T) X, Y, and Z dipole components.",
        json_schema_extra={"shape": [3], "units": "e a0"},
    )

    # CCSDT keywords
    ccsdt_correlation_energy: Optional[float] = Field(
        None,
        description="The CCSDT correlation energy.",
        json_schema_extra={"units": "E_h"},
    )
    ccsdt_total_energy: Optional[float] = Field(
        None,
        description="The total CCSDT energy (CCSDT correlation energy + HF energy).",
        json_schema_extra={"units": "E_h"},
    )
    ccsdt_dipole_moment: Optional[Array[float]] = Field(
        None,
        description="The CCSDT X, Y, and Z dipole components.",
        json_schema_extra={"shape": [3], "units": "e a0"},
    )
    ccsdt_iterations: Optional[int] = Field(
        None, description="The number of CCSDT iterations taken before convergence."
    )

    # CCSDTQ keywords
    ccsdtq_correlation_energy: Optional[float] = Field(
        None,
        description="The CCSDTQ correlation energy.",
        json_schema_extra={"units": "E_h"},
    )
    ccsdtq_total_energy: Optional[float] = Field(
        None,
        description="The total CCSDTQ energy (CCSDTQ correlation energy + HF energy).",
        json_schema_extra={"units": "E_h"},
    )
    ccsdtq_dipole_moment: Optional[Array[float]] = Field(
        None,
        description="The CCSDTQ X, Y, and Z dipole components.",
        json_schema_extra={"shape": [3], "units": "e a0"},
    )
    ccsdtq_iterations: Optional[int] = Field(
        None, description="The number of CCSDTQ iterations taken before convergence."
    )

    model_config = ProtoModel._merge_config_with(force_skip_defaults=True)

    def __repr_args__(self) -> "ReprArgs":
        return [(k, v) for k, v in self.dict().items()]

    @field_validator(
        "scf_dipole_moment",
        "mp2_dipole_moment",
        "ccsd_dipole_moment",
        "ccsd_prt_pr_dipole_moment",
        "scf_quadrupole_moment",
    )
    @classmethod
    def _validate_poles(cls, v, info):
        if v is None:
            return v

        if info.field_name.endswith("_dipole_moment"):
            order = 1
        elif info.field_name.endswith("_quadrupole_moment"):
            order = 2

        shape = tuple([3] * order)
        return np.asarray(v).reshape(shape)

    @field_validator(
        "return_gradient",
        "return_hessian",
        "scf_total_gradient",
        "scf_total_hessian",
    )
    @classmethod
    def _validate_derivs(cls, v, info):
        if v is None:
            return v

        nat = info.data.get("calcinfo_natom", None)
        if nat is None:
            raise ValueError(f"Please also set ``calcinfo_natom``!")

        if info.field_name.endswith("_gradient"):
            shape = (nat, 3)
        elif info.field_name.endswith("_hessian"):
            shape = (3 * nat, 3 * nat)

        try:
            v = np.asarray(v).reshape(shape)
        except (ValueError, AttributeError):
            raise ValueError(f"Derivative must be castable to shape {shape}!")
        return v

    def dict(self, *args, **kwargs):
        # pure-json dict repr for QCFractal compliance, see https://github.com/MolSSI/QCFractal/issues/579
        # Sep 2021: commenting below for now to allow recomposing AtomicResult.properties for qcdb.
        #   This will break QCFractal tests for now, but future qcf will be ok with it.
        # kwargs["encoding"] = "json"
        return super().model_dump(*args, **kwargs)


class WavefunctionProperties(ProtoModel):
    r"""Wavefunction properties resulting from a computation.
    Matrix quantities are stored in column-major order. Presence and contents configurable by protocol."""

    # Class properties
    _return_results_names: Set[str] = {
        "orbitals_a",
        "orbitals_b",
        "density_a",
        "density_b",
        "fock_a",
        "fock_b",
        "eigenvalues_a",
        "eigenvalues_b",
        "occupations_a",
        "occupations_b",
    }

    schema_name: Literal["qcschema_wavefunction_properties"] = Field(
        "qcschema_wavefunction_properties", description=f"The QCSchema specification to which this model conforms."
    )

    # The full basis set description of the quantities
    basis: BasisSet = Field(..., description=str(BasisSet.__doc__))
    restricted: bool = Field(
        ...,
        description=str(
            "If the computation was restricted or not (alpha == beta). If True, all beta quantities are skipped."
        ),
    )

    # Core Hamiltonian
    h_core_a: Optional[Array[float]] = Field(
        None,
        description="Alpha-spin core (one-electron) Hamiltonian in the AO basis.",
        json_schema_extra={
            "shape": ["nao", "nao"],
        },
    )
    h_core_b: Optional[Array[float]] = Field(
        None,
        description="Beta-spin core (one-electron) Hamiltonian in the AO basis.",
        json_schema_extra={
            "shape": ["nao", "nao"],
        },
    )
    h_effective_a: Optional[Array[float]] = Field(
        None,
        description="Alpha-spin effective core (one-electron) Hamiltonian in the AO basis.",
        json_schema_extra={
            "shape": ["nao", "nao"],
        },
    )
    h_effective_b: Optional[Array[float]] = Field(
        None,
        description="Beta-spin effective core (one-electron) Hamiltonian in the AO basis",
        json_schema_extra={
            "shape": ["nao", "nao"],
        },
    )

    # SCF Results
    scf_orbitals_a: Optional[Array[float]] = Field(
        None,
        description="SCF alpha-spin orbitals in the AO basis.",
        json_schema_extra={
            "shape": ["nao", "nmo"],
        },
    )
    scf_orbitals_b: Optional[Array[float]] = Field(
        None,
        description="SCF beta-spin orbitals in the AO basis.",
        json_schema_extra={
            "shape": ["nao", "nmo"],
        },
    )
    scf_density_a: Optional[Array[float]] = Field(
        None,
        description="SCF alpha-spin density matrix in the AO basis.",
        json_schema_extra={
            "shape": ["nao", "nao"],
        },
    )
    scf_density_b: Optional[Array[float]] = Field(
        None,
        description="SCF beta-spin density matrix in the AO basis.",
        json_schema_extra={
            "shape": ["nao", "nao"],
        },
    )
    scf_fock_a: Optional[Array[float]] = Field(
        None,
        description="SCF alpha-spin Fock matrix in the AO basis.",
        json_schema_extra={
            "shape": ["nao", "nao"],
        },
    )
    scf_fock_b: Optional[Array[float]] = Field(
        None,
        description="SCF beta-spin Fock matrix in the AO basis.",
        json_schema_extra={
            "shape": ["nao", "nao"],
        },
    )
    scf_eigenvalues_a: Optional[Array[float]] = Field(
        None,
        description="SCF alpha-spin orbital eigenvalues.",
        json_schema_extra={
            "shape": ["nmo"],
        },
    )
    scf_eigenvalues_b: Optional[Array[float]] = Field(
        None,
        description="SCF beta-spin orbital eigenvalues.",
        json_schema_extra={
            "shape": ["nmo"],
        },
    )
    scf_occupations_a: Optional[Array[float]] = Field(
        None,
        description="SCF alpha-spin orbital occupations.",
        json_schema_extra={
            "shape": ["nmo"],
        },
    )
    scf_occupations_b: Optional[Array[float]] = Field(
        None,
        description="SCF beta-spin orbital occupations.",
        json_schema_extra={
            "shape": ["nmo"],
        },
    )

    # BELOW from qcsk
    scf_coulomb_a: Optional[Array[float]] = Field(
        None,
        description="SCF alpha-spin Coulomb matrix in the AO basis.",
        json_schema_extra={
            "shape": ["nao", "nao"],
        },
    )
    scf_coulomb_b: Optional[Array[float]] = Field(
        None,
        description="SCF beta-spin Coulomb matrix in the AO basis.",
        json_schema_extra={
            "shape": ["nao", "nao"],
        },
    )
    scf_exchange_a: Optional[Array[float]] = Field(
        None,
        description="SCF alpha-spin exchange matrix in the AO basis.",
        json_schema_extra={
            "shape": ["nao", "nao"],
        },
    )
    scf_exchange_b: Optional[Array[float]] = Field(
        None,
        description="SCF beta-spin exchange matrix in the AO basis.",
        json_schema_extra={
            "shape": ["nao", "nao"],
        },
    )

    # Localized-orbital SCF wavefunction quantities
    localized_orbitals_a: Optional[Array[float]] = Field(
        None,
        description="Localized alpha-spin orbitals in the AO basis. All nmo orbitals are included, even if only a subset were localized.",
        json_schema_extra={"shape": ["nao", "nmo"]},
    )
    localized_orbitals_b: Optional[Array[float]] = Field(
        None,
        description="Localized beta-spin orbitals in the AO basis. All nmo orbitals are included, even if only a subset were localized.",
        json_schema_extra={"shape": ["nao", "nmo"]},
    )
    localized_fock_a: Optional[Array[float]] = Field(
        None,
        description="Alpha-spin Fock matrix in the localized molecular orbital basis. All nmo orbitals are included, even if only a subset were localized.",
        json_schema_extra={"shape": ["nmo", "nmo"]},
    )
    localized_fock_b: Optional[Array[float]] = Field(
        None,
        description="Beta-spin Fock matrix in the localized molecular orbital basis. All nmo orbitals are included, even if only a subset were localized.",
        json_schema_extra={"shape": ["nmo", "nmo"]},
    )
    # ABOVE from qcsk

    # Return results, must be defined last
    orbitals_a: Optional[str] = Field(None, description="Index to the alpha-spin orbitals of the primary return.")
    orbitals_b: Optional[str] = Field(None, description="Index to the beta-spin orbitals of the primary return.")
    density_a: Optional[str] = Field(None, description="Index to the alpha-spin density of the primary return.")
    density_b: Optional[str] = Field(None, description="Index to the beta-spin density of the primary return.")
    fock_a: Optional[str] = Field(None, description="Index to the alpha-spin Fock matrix of the primary return.")
    fock_b: Optional[str] = Field(None, description="Index to the beta-spin Fock matrix of the primary return.")
    eigenvalues_a: Optional[str] = Field(
        None, description="Index to the alpha-spin orbital eigenvalues of the primary return."
    )
    eigenvalues_b: Optional[str] = Field(
        None, description="Index to the beta-spin orbital eigenvalues of the primary return."
    )
    occupations_a: Optional[str] = Field(
        None, description="Index to the alpha-spin orbital occupations of the primary return."
    )
    occupations_b: Optional[str] = Field(
        None, description="Index to the beta-spin orbital occupations of the primary return."
    )

    # Note that serializing WfnProp skips unset fields (and indeed the validator will error upon None values)
    #   while including all fields for the submodel BasisSet. This is the right behavior, imo, but note that
    #   v1 skips unset fields in BasisSet as well as the top-level model.
    model_config = ProtoModel._merge_config_with(force_skip_defaults=True)

    @field_validator("scf_eigenvalues_a", "scf_eigenvalues_b", "scf_occupations_a", "scf_occupations_b")
    @classmethod
    def _assert1d(cls, v):
        try:
            v = v.reshape(-1)
        except (ValueError, AttributeError):
            raise ValueError("Vector must be castable to shape (-1, )!")
        return v

    @field_validator("scf_orbitals_a", "scf_orbitals_b")
    @classmethod
    def _assert2d_nao_x(cls, v, info):
        bas = info.data.get("basis", None)

        # Do not raise multiple errors
        if bas is None:
            return v

        try:
            v = v.reshape(bas.nbf, -1)
        except (ValueError, AttributeError):
            raise ValueError("Matrix must be castable to shape (nbf, -1)!")
        return v

    @field_validator(
        "h_core_a",
        "h_core_b",
        "h_effective_a",
        "h_effective_b",
        # SCF
        "scf_density_a",
        "scf_density_b",
        "scf_fock_a",
        "scf_fock_b",
    )
    @classmethod
    def _assert2d(cls, v, info):
        bas = info.data.get("basis", None)

        # Do not raise multiple errors
        if bas is None:
            return v

        try:
            v = v.reshape(bas.nbf, bas.nbf)
        except (ValueError, AttributeError):
            raise ValueError("Matrix must be castable to shape (nbf, nbf)!")
        return v

    @field_validator(
        "orbitals_a",
        "orbitals_b",
        "density_a",
        "density_b",
        "fock_a",
        "fock_b",
        "eigenvalues_a",
        "eigenvalues_b",
        "occupations_a",
        "occupations_b",
    )
    @classmethod
    def _assert_exists(cls, v, info):
        if info.data.get(v, None) is None:
            raise ValueError(f"Return quantity {v} does not exist in the values.")
        return v

    def convert_v(
        self, target_version: int, /
    ) -> Union["qcelemental.models.v1.WavefunctionProperties", "qcelemental.models.v2.WavefunctionProperties"]:
        """Convert to instance of particular QCSchema version."""
        import qcelemental as qcel

        if check_convertible_version(target_version, error="WavefunctionProperties") == "self":
            return self

        dself = self.model_dump()
        if target_version == 1:
            dself["basis"] = self.basis.convert_v(target_version).dict()

            self_vN = qcel.models.v1.WavefunctionProperties(**dself)
        else:
            assert False, target_version

        return self_vN


# ====  Protocols  ==============================================================


class WavefunctionProtocolEnum(str, Enum):
    r"""Wavefunction to keep from a computation."""

    all = "all"
    orbitals_and_eigenvalues = "orbitals_and_eigenvalues"
    occupations_and_eigenvalues = "occupations_and_eigenvalues"
    return_results = "return_results"
    none = "none"


class ErrorCorrectionProtocol(ProtoModel):
    r"""Configuration for how QCEngine handles error correction

    WARNING: These protocols are currently experimental and only supported by NWChem tasks
    """

    default_policy: bool = Field(
        True, description="Whether to allow error corrections to be used " "if not directly specified in `policies`"
    )
    # TODO (wardlt): Consider support for common policies (e.g., 'only increase iterations') as strings (see #182)
    policies: Optional[Dict[str, bool]] = Field(
        None,
        description="Settings that define whether specific error corrections are allowed. "
        "Keys are the name of a known error and values are whether it is allowed to be used.",
    )

    def allows(self, policy: str):
        if self.policies is None:
            return self.default_policy
        return self.policies.get(policy, self.default_policy)


class NativeFilesProtocolEnum(str, Enum):
    r"""CMS program files to keep from a computation."""

    all = "all"
    input = "input"
    none = "none"


class AtomicProtocols(ProtoModel):
    r"""Protocols regarding the manipulation of computational result data."""

    schema_name: Literal["qcschema_atomic_protocols"] = "qcschema_atomic_protocols"

    wavefunction: WavefunctionProtocolEnum = Field(
        WavefunctionProtocolEnum.none, description=str(WavefunctionProtocolEnum.__doc__)
    )
    stdout: bool = Field(True, description="Primary output file to keep from the computation")
    error_correction: ErrorCorrectionProtocol = Field(
        default_factory=ErrorCorrectionProtocol, description="Policies for error correction"
    )
    native_files: NativeFilesProtocolEnum = Field(
        NativeFilesProtocolEnum.none,
        description="Policies for keeping processed files from the computation",
    )

    model_config = ExtendedConfigDict(force_skip_defaults=True)


# ====  Inputs (Kw/Spec/In)  ====================================================


class AtomicSpecification(ProtoModel):
    """Specification for a single point QC calculation"""

    schema_name: Literal["qcschema_atomic_specification"] = "qcschema_atomic_specification"

    keywords: Dict[str, Any] = Field({}, description="The program specific keywords to be used.")
    program: str = Field(
        "", description="The program for which the Specification is intended."
    )  # TODO interaction with cmdline
    driver: DriverEnum = Field(..., description=DriverEnum.__doc__)
    model: Model = Field(..., description=Model.__doc__)
    protocols: AtomicProtocols = Field(
        AtomicProtocols(),
        description=AtomicProtocols.__doc__,
    )
    extras: Dict[str, Any] = Field(
        {},
        description="Additional information to bundle with the computation. Use for schema development and scratch space.",
    )

    def convert_v(
        self, target_version: int, /
    ) -> Union["qcelemental.models.v1.QCInputSpecification", "qcelemental.models.v2.AtomicSpecification"]:
        """Convert to instance of particular QCSchema version."""
        import qcelemental as qcel

        if check_convertible_version(target_version, error="AtomicSpecification") == "self":
            return self

        loss_store = {}
        dself = self.model_dump()
        if target_version == 1:
            dself.pop("schema_name")

            loss_store["protocols"] = dself.pop("protocols")
            loss_store["program"] = dself.pop("program")

            if loss_store:
                pass
                # TODO dself["extras"]["_qcsk_conversion_loss"] = loss_store

            self_vN = qcel.models.v1.QCInputSpecification(**dself)
        else:
            assert False, target_version

        return self_vN


def atomic_input_json_schema_extra(schema, model):
    schema["$schema"] = qcschema_draft


class AtomicInput(ProtoModel):
    r"""The MolSSI Quantum Chemistry Schema"""

    id: Optional[str] = Field(None, description="The optional ID for the computation.")
    schema_name: Literal["qcschema_atomic_input"] = Field(
        "qcschema_atomic_input", description=(f"The QCSchema specification to which this model conforms.")
    )
    schema_version: Literal[2] = Field(
        2,
        description="The version number of ``schema_name`` to which this model conforms.",
    )

    molecule: Molecule = Field(..., description="The molecule to use in the computation.")

    specification: AtomicSpecification = Field(
        ..., description="Additional fields specifying how to run the single-point computation."
    )

    provenance: Provenance = Field(
        default_factory=partial(provenance_stamp, __name__),
        description=str(Provenance.__doc__),
        validate_default=True,  # Cast inputs to
    )

    model_config = ProtoModel._merge_config_with(json_schema_extra=atomic_input_json_schema_extra)

    def __repr_args__(self) -> "ReprArgs":
        return [
            ("driver", self.specification.driver.value),
            ("model", self.specification.model.model_dump()),
            ("molecule_hash", self.molecule.get_hash()[:7]),
        ]

    def convert_v(
        self, target_version: int, /
    ) -> Union["qcelemental.models.v1.AtomicInput", "qcelemental.models.v2.AtomicInput"]:
        """Convert to instance of particular QCSchema version."""
        import qcelemental as qcel

        if check_convertible_version(target_version, error="AtomicInput") == "self":
            return self

        dself = self.model_dump()
        if target_version == 1:
            dself.pop("schema_name")
            dself.pop("schema_version")

            # TODO consider Model.convert_v
            model = dself["specification"].pop("model")
            if isinstance(self.specification.model.basis, BasisSet):
                model["basis"] = self.specification.model.basis.convert_v(target_version)

            dself["molecule"] = self.molecule.convert_v(target_version).model_dump()
            dself["driver"] = dself["specification"].pop("driver")
            dself["model"] = model
            dself["keywords"] = dself["specification"].pop("keywords", None)
            dself["protocols"] = dself["specification"].pop("protocols", None)
            dself["extras"] = dself["specification"].pop("extras", {})
            dself["specification"].pop("program", None)  # TODO store?
            dself["specification"].pop("schema_name", None)
            assert not dself["specification"], dself["specification"]
            dself.pop("specification")  # now empty

            self_vN = qcel.models.v1.AtomicInput(**dself)
        else:
            assert False, target_version

        return self_vN


# ====  Results  ================================================================


class AtomicResult(ProtoModel):
    r"""Results from a CMS program execution."""

    schema_name: Literal["qcschema_atomic_result"] = Field(
        "qcschema_atomic_result", description=(f"The QCSchema specification to which this model conforms.")
    )
    schema_version: Literal[2] = Field(
        2,
        description="The version number of :attr:`~qcelemental.models.AtomicResult.schema_name` to which this model conforms.",
    )
    id: Optional[str] = Field(None, description="The optional ID for the computation.")
    input_data: AtomicInput = Field(..., description=str(AtomicInput.__doc__))
    molecule: Molecule = Field(..., description="The molecule with frame and orientation of the results.")
    properties: AtomicProperties = Field(..., description=str(AtomicProperties.__doc__))
    wavefunction: Optional[WavefunctionProperties] = Field(None, description=str(WavefunctionProperties.__doc__))

    return_result: Union[float, Array[float], Dict[str, Any]] = Field(
        ...,
        description="The primary return specified by the :attr:`~qcelemental.models.AtomicInput.driver` field. Scalar if energy; array if gradient or hessian; dictionary with property keys if properties.",
    )  # type: ignore

    stdout: Optional[str] = Field(
        None,
        description="The primary logging output of the program, whether natively standard output or a file. Presence vs. absence (or null-ness?) configurable by protocol.",
    )
    stderr: Optional[str] = Field(None, description="The standard error of the program execution.")
    native_files: Dict[str, Any] = Field({}, description="DSL files.")

    success: Literal[True] = Field(
        True, description="The success of program execution. If False, other fields may be blank."
    )
    provenance: Provenance = Field(..., description=str(Provenance.__doc__))
    extras: Dict[str, Any] = Field(
        {},
        description="Additional information to bundle with the computation. Use for schema development and scratch space.",
    )

    @field_validator("return_result")
    @classmethod
    def _validate_return_result(cls, v, info):
        # Do not propagate validation errors
        if "input_data" not in info.data:
            raise ValueError("Input_data was not properly formed.")
        driver = info.data["input_data"].specification.driver
        if driver == "energy":
            if isinstance(v, np.ndarray) and v.size == 1:
                v = v.item(0)
        elif driver == "gradient":
            v = np.asarray(v).reshape(-1, 3)
        elif driver == "hessian":
            v = np.asarray(v)
            nsq = int(v.size**0.5)
            v.shape = (nsq, nsq)

        return v

    @field_validator("wavefunction", mode="before")
    @classmethod
    def _wavefunction_protocol(cls, value, info):
        # We are pre, gotta do extra checks
        if value is None:
            return value
        elif isinstance(value, dict):
            wfn = value.copy()
        elif isinstance(value, WavefunctionProperties):
            wfn = value.model_dump()
        else:
            raise ValueError("wavefunction must be None, a dict, or a WavefunctionProperties object.")

        # Do not propagate validation errors
        if "input_data" not in info.data:
            raise ValueError("Input_data was not properly formed.")

        # Handle restricted
        restricted = wfn.get("restricted", None)
        if restricted is None:
            raise ValueError("`restricted` is required.")

        if restricted:
            for k in list(wfn.keys()):
                if k.endswith("_b"):
                    wfn.pop(k)

        # Handle protocols
        wfnp = info.data["input_data"].specification.protocols.wavefunction
        return_keep = None
        if wfnp == "all":
            pass
        elif wfnp == "none":
            wfn = None
        elif wfnp == "return_results":
            return_keep = [
                "orbitals_a",
                "orbitals_b",
                "density_a",
                "density_b",
                "fock_a",
                "fock_b",
                "eigenvalues_a",
                "eigenvalues_b",
                "occupations_a",
                "occupations_b",
            ]
        elif wfnp == "orbitals_and_eigenvalues":
            return_keep = ["orbitals_a", "orbitals_b", "eigenvalues_a", "eigenvalues_b"]
        elif wfnp == "occupations_and_eigenvalues":
            return_keep = ["occupations_a", "occupations_b", "eigenvalues_a", "eigenvalues_b"]
        else:
            raise ValueError(f"Protocol `wavefunction:{wfnp}` is not understood.")

        if return_keep is not None:
            ret_wfn = {"restricted": restricted}
            if "basis" in wfn:
                ret_wfn["basis"] = wfn["basis"]

            for rk in return_keep:
                key = wfn.get(rk, None)
                if key is None:
                    continue

                ret_wfn[rk] = key
                ret_wfn[key] = wfn[key]

            return ret_wfn
        else:
            return wfn

    @field_validator("stdout")
    @classmethod
    def _stdout_protocol(cls, value, info):
        # Do not propagate validation errors
        if "input_data" not in info.data:
            raise ValueError("Input_data was not properly formed.")

        outp = info.data["input_data"].specification.protocols.stdout
        if outp is True:
            return value
        elif outp is False:
            return None
        else:
            raise ValueError(f"Protocol `stdout:{outp}` is not understood")

    @field_validator("native_files")
    @classmethod
    def _native_file_protocol(cls, value, info):
        # Do not propagate validation errors
        if "input_data" not in info.data:
            raise ValueError("Input_data was not properly formed.")

        ancp = info.data["input_data"].specification.protocols.native_files
        if ancp == "all":
            return value
        elif ancp == "none":
            return {}
        elif ancp == "input":
            return_keep = ["input"]
            if value is None:
                files = {}
            else:
                files = value.copy()
        else:
            raise ValueError(f"Protocol `native_files:{ancp}` is not understood")

        ret = {}
        for rk in return_keep:
            ret[rk] = files.get(rk, None)
        return ret

    def convert_v(
        self, target_version: int, /
    ) -> Union["qcelemental.models.v1.AtomicResult", "qcelemental.models.v2.AtomicResult"]:
        """Convert to instance of particular QCSchema version."""
        import qcelemental as qcel

        if check_convertible_version(target_version, error="AtomicResult") == "self":
            return self

        dself = self.model_dump()
        if target_version == 1:
            dself.pop("schema_name")
            dself.pop("schema_version")

            # for input_data, work from model, not dict, to use convert_v
            dself.pop("input_data")
            input_data = self.input_data.convert_v(target_version).model_dump()  # exclude_unset=True, exclude_none=True
            input_data.pop("molecule", None)  # discard
            input_data.pop("provenance", None)  # discard
            if self.wavefunction is not None:
                dself["wavefunction"] = self.wavefunction.convert_v(target_version).model_dump()
            dself["molecule"] = self.molecule.convert_v(target_version)
            dself["extras"] = {**input_data.pop("extras", {}), **dself.pop("extras", {})}  # merge
            dself = {**input_data, **dself}

            self_vN = qcel.models.v1.AtomicResult(**dself)
        else:
            assert False, target_version

        return self_vN
