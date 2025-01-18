from enum import Enum
from functools import partial
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Set, Union

try:
    from typing import Literal
except ImportError:
    # remove when minimum py38
    from typing_extensions import Literal

import numpy as np

try:
    from pydantic.v1 import Field, constr, validator
except ImportError:  # Will also trap ModuleNotFoundError
    from pydantic import Field, constr, validator

from ..util import provenance_stamp
from .basemodels import ProtoModel, qcschema_draft
from .basis import BasisSet
from .common_models import ComputeError, DriverEnum, Model, Provenance, qcschema_input_default, qcschema_output_default
from .molecule import Molecule
from .types import Array

if TYPE_CHECKING:
    try:
        from pydantic.v1.typing import ReprArgs
    except ImportError:  # Will also trap ModuleNotFoundError
        from pydantic.typing import ReprArgs


class MolecularDescriptors(ProtoModel):
    r"""
    Molecular descriptors for cheminformatics and AI/ML applications following the MolSSI QCSchema.
    """

    canonical_smiles: Optional[str] = Field(
        None, description="Canonical (non-kekulized, implicit hydrogens) SMILES string representing the molecule."
    )
    inchi_key: Optional[str] = Field(None, description="Internet searchable InChI hash.")
    inchi: Optional[str] = Field(None, description="InChI string for molecule.")
    num_val_e: Optional[int] = Field(None, description="Valence electron count")
    num_rad_e: Optional[int] = Field(None, description="Radical electron count.")
    num_hetero: Optional[int] = Field(None, description="Number of heteroatoms (not C/H).")
    num_no: Optional[int] = Field(None, description="Number of N/O atoms.")
    num_nhoh: Optional[int] = Field(None, description="Number of NH/OH groups.")
    num_h_acceptors: Optional[int] = Field(None, description="Number of hydrogen-bond acceptors.")
    num_h_donors: Optional[int] = Field(None, description="Number of hydrogen-bond donors.")
    num_rot_bonds: Optional[int] = Field(None, description="Number of (relatively) freely rotating bonds.")
    num_spiro_atoms: Optional[int] = Field(None, description="Number of spiro atoms.")
    num_bridgehead_atoms: Optional[int] = Field(None, description="Number of bridgehead atoms")
    atomic_charges: Optional[Dict[str, List[float]]] = Field(
        None,
        description="Calculated atomic charges where 'gasteiger' is following method in (Gasteiger, J.; Marseli, M. Tetrahedron, 1980), and 'eem' is Electronegativity Equalization Method following (Mortier, W.J.; Van Genechten, K.; Gasteiger, J. JACS, 1985 and Mortier, W.J.; Ghosh, S.K.; Shankar, S. JACS, 1986).",
    )
    stereochemistry: Optional[List[Union[Tuple[int, str], Tuple[int, int, str]]]] = Field(
        None, description="CIP (R/S, E/Z) stereochemistry by atom indices and stereo type."
    )
    aromaticity: Optional[Dict[str, List[Union[int, Tuple[int, int]]]]] = Field(
        None, description="Dictionary of bond indices and atom indices of aromatic bonds."
    )
    ring_info: Optional[Dict[str, Union[int, Dict[int, Dict[str, Union[int, bool, List[int]]]]]]] = Field(
        None,
        description="Ring counts by type info for each ring where the key is the ring index and contains the size, whether it's aromatic(bool) and/or a heterocycle(bool), and atom indices.",
    )
    logp: Optional[float] = Field(
        None,
        description="Octanol-water partition coefficient approximation from Wildman, S.A.; Crippen, G.M. J. Chem. Inf. Comput. Sci., 1999.",
    )
    mol_refract: Optional[float] = Field(
        None,
        description="Molar refractivity in m^3/mol from Wildman, S.A.; Crippen, G.M. J. Chem. Inf. Comput. Sci., 1999.",
    )
    surface_area: Optional[Dict[str, float]] = Field(
        None,
        description="Surface area approximations by different methods, containing Topological Polar Surface Area(TPSA) (Ertl P. et al, J. Med. Chem., 2000), the CCG approximations Labute ASA, PEOE VSA, SMR VSA, SlogP VSA and EState VSA (Labute, P. J. Mol. Graph. Model., 2000), an EState VSA variant developed by RDKit, and an average of the VSA methods.",
    )
    plane_best_fit: Optional[float] = Field(
        None,
        description="Average distance in Angstrom from the plane of best fit of a molecule's atoms (Firth, N.C.; Brown, N.; Blagg, J. J. Chem. Inf. Model., 2012).",
    )
    mo_inertia: Optional[Dict[str, List[float]]] = Field(
        None,
        description="Principal and normalized moments of inertia as described in (Sauer, W.H.B.; Schwarz, M.K. J. Chem. Inf. Comput. Sci., 2003).",
    )
    rad_gyration: Optional[float] = Field(
        None,
        description="Radius of gyration from Arteca, G.A. 'Molecular Shape Descriptors' in Reviews in Computational Chemistry vol. 9.",
    )
    inertial_shape_factor: Optional[float] = Field(
        None,
        description="From Todeschini, R.; Consonni, V. 'Descriptors from Molecular Geometry' in Handbook of Chemoinformatics.",
    )
    eccentricity: Optional[float] = Field(
        None,
        description="Elliptical eccentricities derived from principal inertia moments, from Arteca, G.A. 'Molecular Shape Descriptors' in Reviews in Computational Chemistry vol. 9.",
    )
    asphericity: Optional[float] = Field(
        None,
        description="Molecular asphericity derived from principal moments of inertia, from Baumgartner, A. J. Chem. Phys. 1993.",
    )
    spherocity_idx: Optional[float] = Field(
        None,
        description="A function of the eigenvalues of the covariance matrix of atomic coordinates, from Todeschini, R.; Consonni, V. 'Descriptors from Molecular Geometry' in Handbook of Chemoinformatics.",
    )
    balaban_idx: Optional[float] = Field(
        None, description="2D Topological index from Balaban, J. Chem. Phys. Lett. 1982."
    )
    bertz_ct: Optional[float] = Field(None, description="2D topological index from Bertz, S.H. JACS, 1981.")
    ipc_idx: Optional[float] = Field(
        None, description="2D topological index from Bonchev, D.; Trinajstic, N. J. Chem. Phys. 1977."
    )
    hall_kier_parameters: Optional[Dict[str, float]] = Field(
        None,
        description="A set of 2D topological indexes from Hall, L.H.; Kier, L.B. 'The Molecular Connectivity Chi Indexes and ...' in Reviews in Computational Chemistry Vol. 2, also including Phi index from Kier, L.B. Mol. Inform., 1989 (journal previously called Quant. Struct.-Act. Relat.)",
    )
    bcut2D: Optional[Dict[str, float]] = Field(
        None,
        description="High and low eigenvalues of topologically weighted matrices where an atomic property (charge, polarizability, etc) fills diagonal as a non-fingerprint based measure of similarity, from Pearlman, R.S.; Smith,K.M. 'Novel Software Tools for Chemical Diversity' in 3D QSAR in Drug Design Vol. 2.",
    )
    autocorr2D: Optional[List[float]] = Field(
        None,
        description="2D topological based autocorrelation of atomic physicochemical properties from Todeschini, R.; Consonni, V. 'Descriptors from Molecular Geometry' in Handbook of Chemoinformatics.",
    )
    autocorr3D: Optional[List[float]] = Field(
        None,
        description="3D geometry based autocorrelation of atomic physicochemical properties from Todeschini, R.; Consonni, V. 'Descriptors from Molecular Geometry' in Handbook of Chemoinformatics.",
    )
    morse: Optional[List[float]] = Field(
        None,
        description="3D Molecule Representation of Structures based on Electron diffraction (MoRSE) from Todeschini, R.; Consonni, V. 'Descriptors from Molecular Geometry' in Handbook of Chemoinformatics.",
    )
    rdf: Optional[List[float]] = Field(
        None,
        description="Radial Distance Function (RDF) descriptors from Todeschini, R.; Consonni, V. 'Descriptors from Molecular Geometry' in Handbook of Chemoinformatics.",
    )
    whim: Optional[List[float]] = Field(
        None,
        description="Weighted Holistic Invariant Molecular (WHIM) descriptors, based on projections of atoms along the principal axis, from Todeschini, R.; Consonni, V. 'Descriptors from Molecular Geometry' in Handbook of Chemoinformatics.",
    )
    getaway: Optional[List[float]] = Field(
        None,
        description="GEometry, Topology, and Atom-Weights AssemblY (GETAWAY) 3D descriptors, based on statistical analysis of Molecular Influence Matrix, from Todeschini, R.; Consonni, V. 'Descriptors from Molecular Geometry' in Handbook of Chemoinformatics.",
    )


class AtomicResultProperties(ProtoModel):
    r"""
    Named properties of quantum chemistry computations following the MolSSI QCSchema.

    All arrays are stored flat but must be reshapable into the dimensions in attribute ``shape``, with abbreviations as follows:

    * nao: number of atomic orbitals = :attr:`~qcelemental.models.AtomicResultProperties.calcinfo_nbasis`
    * nmo: number of molecular orbitals = :attr:`~qcelemental.models.AtomicResultProperties.calcinfo_nmo`
    """

    # Calcinfo
    calcinfo_nbasis: Optional[int] = Field(None, description="The number of basis functions for the computation.")
    calcinfo_nmo: Optional[int] = Field(None, description="The number of molecular orbitals for the computation.")
    calcinfo_nalpha: Optional[int] = Field(None, description="The number of alpha electrons in the computation.")
    calcinfo_nbeta: Optional[int] = Field(None, description="The number of beta electrons in the computation.")
    calcinfo_natom: Optional[int] = Field(
        None, description="The number of atoms in the computation."
    )  # perhaps this shouldn't be optional

    # Canonical
    nuclear_repulsion_energy: Optional[float] = Field(None, description="The nuclear repulsion energy.")
    return_energy: Optional[float] = Field(
        None,
        description=f"The energy of the requested method, identical to :attr:`~qcelemental.models.AtomicResult.return_result` for :attr:`~qcelemental.models.AtomicInput.driver`\\ =\\ :attr:`~qcelemental.models.DriverEnum.energy` computations.",
    )
    return_gradient: Optional[Array[float]] = Field(
        None,
        description=f"The gradient of the requested method, identical to :attr:`~qcelemental.models.AtomicResult.return_result` for :attr:`~qcelemental.models.AtomicInput.driver`\\ =\\ :attr:`~qcelemental.models.DriverEnum.gradient` computations.",
        units="E_h/a0",
    )
    return_hessian: Optional[Array[float]] = Field(
        None,
        description=f"The Hessian of the requested method, identical to :attr:`~qcelemental.models.AtomicResult.return_result` for :attr:`~qcelemental.models.AtomicInput.driver`\\ =\\ :attr:`~qcelemental.models.DriverEnum.hessian` computations.",
        units="E_h/a0^2",
    )

    # Cheminformatics (currently just rdkit) Keywords
    descriptors: Optional[MolecularDescriptors] = Field(None, description=MolecularDescriptors.__doc__)

    # SCF Keywords
    scf_one_electron_energy: Optional[float] = Field(
        None,
        description="The one-electron (core Hamiltonian) energy contribution to the total SCF energy.",
        units="E_h",
    )
    scf_two_electron_energy: Optional[float] = Field(
        None,
        description="The two-electron energy contribution to the total SCF energy.",
        units="E_h",
    )
    scf_vv10_energy: Optional[float] = Field(
        None,
        description="The VV10 functional energy contribution to the total SCF energy.",
        units="E_h",
    )
    scf_xc_energy: Optional[float] = Field(
        None,
        description="The functional (XC) energy contribution to the total SCF energy.",
        units="E_h",
    )
    scf_dispersion_correction_energy: Optional[float] = Field(
        None,
        description="The dispersion correction appended to an underlying functional when a DFT-D method is requested.",
        units="E_h",
    )
    scf_dipole_moment: Optional[Array[float]] = Field(
        None,
        description="The SCF X, Y, and Z dipole components",
        units="e a0",
    )
    scf_quadrupole_moment: Optional[Array[float]] = Field(
        None,
        description="The quadrupole components (redundant; 6 unique).",
        shape=[3, 3],
        units="e a0^2",
    )
    scf_total_energy: Optional[float] = Field(
        None,
        description="The total electronic energy of the SCF stage of the calculation.",
        units="E_h",
    )
    scf_total_gradient: Optional[Array[float]] = Field(
        None,
        description="The total electronic gradient of the SCF stage of the calculation.",
        units="E_h/a0",
    )
    scf_total_hessian: Optional[Array[float]] = Field(
        None,
        description="The total electronic Hessian of the SCF stage of the calculation.",
        units="E_h/a0^2",
    )
    scf_iterations: Optional[int] = Field(None, description="The number of SCF iterations taken before convergence.")

    # MP2 Keywords
    mp2_same_spin_correlation_energy: Optional[float] = Field(
        None,
        description="The portion of MP2 doubles correlation energy from same-spin (i.e. triplet) correlations, without any user scaling.",
        units="E_h",
    )
    mp2_opposite_spin_correlation_energy: Optional[float] = Field(
        None,
        description="The portion of MP2 doubles correlation energy from opposite-spin (i.e. singlet) correlations, without any user scaling.",
        units="E_h",
    )
    mp2_singles_energy: Optional[float] = Field(
        None,
        description="The singles portion of the MP2 correlation energy. Zero except in ROHF.",
        units="E_h",
    )
    mp2_doubles_energy: Optional[float] = Field(
        None,
        description="The doubles portion of the MP2 correlation energy including same-spin and opposite-spin correlations.",
        units="E_h",
    )
    mp2_correlation_energy: Optional[float] = Field(
        None,
        description="The MP2 correlation energy.",
        units="E_h",
    )
    mp2_total_energy: Optional[float] = Field(
        None,
        description="The total MP2 energy (MP2 correlation energy + HF energy).",
        units="E_h",
    )
    mp2_dipole_moment: Optional[Array[float]] = Field(
        None,
        description="The MP2 X, Y, and Z dipole components.",
        shape=[3],
        units="e a0",
    )

    # CCSD Keywords
    ccsd_same_spin_correlation_energy: Optional[float] = Field(
        None,
        description="The portion of CCSD doubles correlation energy from same-spin (i.e. triplet) correlations, without any user scaling.",
        units="E_h",
    )
    ccsd_opposite_spin_correlation_energy: Optional[float] = Field(
        None,
        description="The portion of CCSD doubles correlation energy from opposite-spin (i.e. singlet) correlations, without any user scaling.",
        units="E_h",
    )
    ccsd_singles_energy: Optional[float] = Field(
        None,
        description="The singles portion of the CCSD correlation energy. Zero except in ROHF.",
        units="E_h",
    )
    ccsd_doubles_energy: Optional[float] = Field(
        None,
        description="The doubles portion of the CCSD correlation energy including same-spin and opposite-spin correlations.",
        units="E_h",
    )
    ccsd_correlation_energy: Optional[float] = Field(
        None,
        description="The CCSD correlation energy.",
        units="E_h",
    )
    ccsd_total_energy: Optional[float] = Field(
        None,
        description="The total CCSD energy (CCSD correlation energy + HF energy).",
        units="E_h",
    )
    ccsd_dipole_moment: Optional[Array[float]] = Field(
        None,
        description="The CCSD X, Y, and Z dipole components.",
        shape=[3],
        units="e a0",
    )
    ccsd_iterations: Optional[int] = Field(None, description="The number of CCSD iterations taken before convergence.")

    # CCSD(T) keywords
    ccsd_prt_pr_correlation_energy: Optional[float] = Field(
        None,
        description="The CCSD(T) correlation energy.",
        units="E_h",
    )
    ccsd_prt_pr_total_energy: Optional[float] = Field(
        None,
        description="The total CCSD(T) energy (CCSD(T) correlation energy + HF energy).",
        units="E_h",
    )
    ccsd_prt_pr_dipole_moment: Optional[Array[float]] = Field(
        None,
        description="The CCSD(T) X, Y, and Z dipole components.",
        shape=[3],
        units="e a0",
    )

    # CCSDT keywords
    ccsdt_correlation_energy: Optional[float] = Field(
        None,
        description="The CCSDT correlation energy.",
        units="E_h",
    )
    ccsdt_total_energy: Optional[float] = Field(
        None,
        description="The total CCSDT energy (CCSDT correlation energy + HF energy).",
        units="E_h",
    )
    ccsdt_dipole_moment: Optional[Array[float]] = Field(
        None,
        description="The CCSDT X, Y, and Z dipole components.",
        shape=[3],
        units="e a0",
    )
    ccsdt_iterations: Optional[int] = Field(
        None, description="The number of CCSDT iterations taken before convergence."
    )

    # CCSDTQ keywords
    ccsdtq_correlation_energy: Optional[float] = Field(
        None,
        description="The CCSDTQ correlation energy.",
        units="E_h",
    )
    ccsdtq_total_energy: Optional[float] = Field(
        None,
        description="The total CCSDTQ energy (CCSDTQ correlation energy + HF energy).",
        units="E_h",
    )
    ccsdtq_dipole_moment: Optional[Array[float]] = Field(
        None,
        description="The CCSDTQ X, Y, and Z dipole components.",
        shape=[3],
        units="e a0",
    )
    ccsdtq_iterations: Optional[int] = Field(
        None, description="The number of CCSDTQ iterations taken before convergence."
    )

    class Config(ProtoModel.Config):
        force_skip_defaults = True

    def __repr_args__(self) -> "ReprArgs":
        return [(k, v) for k, v in self.dict().items()]

    @validator(
        "scf_dipole_moment",
        "mp2_dipole_moment",
        "ccsd_dipole_moment",
        "ccsd_prt_pr_dipole_moment",
        "scf_quadrupole_moment",
    )
    def _validate_poles(cls, v, values, field):
        if v is None:
            return v

        if field.name.endswith("_dipole_moment"):
            order = 1
        elif field.name.endswith("_quadrupole_moment"):
            order = 2

        shape = tuple([3] * order)
        return np.asarray(v).reshape(shape)

    @validator(
        "return_gradient",
        "return_hessian",
        "scf_total_gradient",
        "scf_total_hessian",
    )
    def _validate_derivs(cls, v, values, field):
        if v is None:
            return v

        nat = values.get("calcinfo_natom", None)
        if nat is None:
            raise ValueError(f"Please also set ``calcinfo_natom``!")

        if field.name.endswith("_gradient"):
            shape = (nat, 3)
        elif field.name.endswith("_hessian"):
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
        return super().dict(*args, **kwargs)


class WavefunctionProperties(ProtoModel):
    r"""Wavefunction properties resulting from a computation. Matrix quantities are stored in column-major order. Presence and contents configurable by protocol."""

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
        None, description="Alpha-spin core (one-electron) Hamiltonian in the AO basis.", shape=["nao", "nao"]
    )
    h_core_b: Optional[Array[float]] = Field(
        None, description="Beta-spin core (one-electron) Hamiltonian in the AO basis.", shape=["nao", "nao"]
    )
    h_effective_a: Optional[Array[float]] = Field(
        None, description="Alpha-spin effective core (one-electron) Hamiltonian in the AO basis.", shape=["nao", "nao"]
    )
    h_effective_b: Optional[Array[float]] = Field(
        None, description="Beta-spin effective core (one-electron) Hamiltonian in the AO basis", shape=["nao", "nao"]
    )

    # SCF Results
    scf_orbitals_a: Optional[Array[float]] = Field(
        None, description="SCF alpha-spin orbitals in the AO basis.", shape=["nao", "nmo"]
    )
    scf_orbitals_b: Optional[Array[float]] = Field(
        None, description="SCF beta-spin orbitals in the AO basis.", shape=["nao", "nmo"]
    )
    scf_density_a: Optional[Array[float]] = Field(
        None, description="SCF alpha-spin density matrix in the AO basis.", shape=["nao", "nao"]
    )
    scf_density_b: Optional[Array[float]] = Field(
        None, description="SCF beta-spin density matrix in the AO basis.", shape=["nao", "nao"]
    )
    scf_fock_a: Optional[Array[float]] = Field(
        None, description="SCF alpha-spin Fock matrix in the AO basis.", shape=["nao", "nao"]
    )
    scf_fock_b: Optional[Array[float]] = Field(
        None, description="SCF beta-spin Fock matrix in the AO basis.", shape=["nao", "nao"]
    )
    scf_eigenvalues_a: Optional[Array[float]] = Field(
        None, description="SCF alpha-spin orbital eigenvalues.", shape=["nmo"]
    )
    scf_eigenvalues_b: Optional[Array[float]] = Field(
        None, description="SCF beta-spin orbital eigenvalues.", shape=["nmo"]
    )
    scf_occupations_a: Optional[Array[float]] = Field(
        None, description="SCF alpha-spin orbital occupations.", shape=["nmo"]
    )
    scf_occupations_b: Optional[Array[float]] = Field(
        None, description="SCF beta-spin orbital occupations.", shape=["nmo"]
    )

    # BELOW from qcsk
    scf_coulomb_a: Optional[Array[float]] = Field(
        None, description="SCF alpha-spin Coulomb matrix in the AO basis.", shape=["nao", "nao"]
    )
    scf_coulomb_b: Optional[Array[float]] = Field(
        None, description="SCF beta-spin Coulomb matrix in the AO basis.", shape=["nao", "nao"]
    )
    scf_exchange_a: Optional[Array[float]] = Field(
        None, description="SCF alpha-spin exchange matrix in the AO basis.", shape=["nao", "nao"]
    )
    scf_exchange_b: Optional[Array[float]] = Field(
        None, description="SCF beta-spin exchange matrix in the AO basis.", shape=["nao", "nao"]
    )

    # Localized-orbital SCF wavefunction quantities
    localized_orbitals_a: Optional[Array[float]] = Field(
        None,
        description="Localized alpha-spin orbitals in the AO basis. All nmo orbitals are included, even if only a subset were localized.",
        shape=["nao", "nmo"],
    )
    localized_orbitals_b: Optional[Array[float]] = Field(
        None,
        description="Localized beta-spin orbitals in the AO basis. All nmo orbitals are included, even if only a subset were localized.",
        shape=["nao", "nmo"],
    )
    localized_fock_a: Optional[Array[float]] = Field(
        None,
        description="Alpha-spin Fock matrix in the localized molecular orbital basis. All nmo orbitals are included, even if only a subset were localized.",
        shape=["nmo", "nmo"],
    )
    localized_fock_b: Optional[Array[float]] = Field(
        None,
        description="Beta-spin Fock matrix in the localized molecular orbital basis. All nmo orbitals are included, even if only a subset were localized.",
        shape=["nmo", "nmo"],
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

    class Config(ProtoModel.Config):
        force_skip_defaults = True

    @validator("scf_eigenvalues_a", "scf_eigenvalues_b", "scf_occupations_a", "scf_occupations_b")
    def _assert1d(cls, v, values):
        try:
            v = v.reshape(-1)
        except (ValueError, AttributeError):
            raise ValueError("Vector must be castable to shape (-1, )!")
        return v

    @validator("scf_orbitals_a", "scf_orbitals_b")
    def _assert2d_nao_x(cls, v, values):
        bas = values.get("basis", None)

        # Do not raise multiple errors
        if bas is None:
            return v

        try:
            v = v.reshape(bas.nbf, -1)
        except (ValueError, AttributeError):
            raise ValueError("Matrix must be castable to shape (nbf, -1)!")
        return v

    @validator(
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
    def _assert2d(cls, v, values):
        bas = values.get("basis", None)

        # Do not raise multiple errors
        if bas is None:
            return v

        try:
            v = v.reshape(bas.nbf, bas.nbf)
        except (ValueError, AttributeError):
            raise ValueError("Matrix must be castable to shape (nbf, nbf)!")
        return v

    @validator(
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
    def _assert_exists(cls, v, values):
        if values.get(v, None) is None:
            raise ValueError(f"Return quantity {v} does not exist in the values.")
        return v


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


class AtomicResultProtocols(ProtoModel):
    r"""Protocols regarding the manipulation of computational result data."""

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

    class Config:
        force_skip_defaults = True


### Primary models


class AtomicInput(ProtoModel):
    r"""The MolSSI Quantum Chemistry Schema"""

    id: Optional[str] = Field(None, description="The optional ID for the computation.")
    schema_name: constr(strip_whitespace=True, regex=r"^(qc\_?schema_input)$") = Field(  # type: ignore
        qcschema_input_default,
        description=(
            f"The QCSchema specification this model conforms to. Explicitly fixed as {qcschema_input_default}."
        ),
    )
    schema_version: int = Field(
        1,
        description="The version number of :attr:`~qcelemental.models.AtomicInput.schema_name` to which this model conforms.",
    )

    molecule: Molecule = Field(..., description="The molecule to use in the computation.")
    driver: DriverEnum = Field(..., description=str(DriverEnum.__doc__))
    model: Model = Field(..., description=str(Model.__doc__))
    keywords: Dict[str, Any] = Field({}, description="The program-specific keywords to be used.")
    protocols: AtomicResultProtocols = Field(AtomicResultProtocols(), description=str(AtomicResultProtocols.__doc__))

    extras: Dict[str, Any] = Field(
        {},
        description="Additional information to bundle with the computation. Use for schema development and scratch space.",
    )

    provenance: Provenance = Field(
        default_factory=partial(provenance_stamp, __name__), description=str(Provenance.__doc__)
    )

    class Config(ProtoModel.Config):
        def schema_extra(schema, model):
            schema["$schema"] = qcschema_draft

    def __repr_args__(self) -> "ReprArgs":
        return [
            ("driver", self.driver.value),
            ("model", self.model.dict()),
            ("molecule_hash", self.molecule.get_hash()[:7]),
        ]


class AtomicResult(AtomicInput):
    r"""Results from a CMS program execution."""

    schema_name: Literal["qcschema_output"] = Field(
        qcschema_output_default,
        description=(
            f"The QCSchema specification this model conforms to. Explicitly fixed as {qcschema_output_default}."
        ),
    )
    properties: AtomicResultProperties = Field(..., description=str(AtomicResultProperties.__doc__))
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

    success: bool = Field(..., description="The success of program execution. If False, other fields may be blank.")
    error: Optional[ComputeError] = Field(None, description=str(ComputeError.__doc__))
    provenance: Provenance = Field(..., description=str(Provenance.__doc__))

    @validator("schema_name", pre=True)
    def _input_to_output(cls, v):
        r"""If qcschema_input is passed in, cast it to output, otherwise no"""
        if v.lower().strip() in [qcschema_input_default, qcschema_output_default]:
            return qcschema_output_default
        raise ValueError(
            "Only {0} or {1} is allowed for schema_name, "
            "which will be converted to {0}".format(qcschema_output_default, qcschema_input_default)
        )

    @validator("return_result")
    def _validate_return_result(cls, v, values):
        if values["driver"] == "gradient":
            v = np.asarray(v).reshape(-1, 3)
        elif values["driver"] == "hessian":
            v = np.asarray(v)
            nsq = int(v.size**0.5)
            v.shape = (nsq, nsq)

        return v

    @validator("wavefunction", pre=True)
    def _wavefunction_protocol(cls, value, values):
        # We are pre, gotta do extra checks
        if value is None:
            return value
        elif isinstance(value, dict):
            wfn = value.copy()
        elif isinstance(value, WavefunctionProperties):
            wfn = value.dict()
        else:
            raise ValueError("wavefunction must be None, a dict, or a WavefunctionProperties object.")

        # Do not propagate validation errors
        if "protocols" not in values:
            raise ValueError("Protocols was not properly formed.")

        # Handle restricted
        restricted = wfn.get("restricted", None)
        if restricted is None:
            raise ValueError("`restricted` is required.")

        if restricted:
            for k in list(wfn.keys()):
                if k.endswith("_b"):
                    wfn.pop(k)

        # Handle protocols
        wfnp = values["protocols"].wavefunction
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

    @validator("stdout")
    def _stdout_protocol(cls, value, values):
        # Do not propagate validation errors
        if "protocols" not in values:
            raise ValueError("Protocols was not properly formed.")

        outp = values["protocols"].stdout
        if outp is True:
            return value
        elif outp is False:
            return None
        else:
            raise ValueError(f"Protocol `stdout:{outp}` is not understood")

    @validator("native_files")
    def _native_file_protocol(cls, value, values):
        ancp = values["protocols"].native_files
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


class ResultProperties(AtomicResultProperties):
    """QC Result Properties Schema.

    .. deprecated:: 0.12
       Use :py:func:`qcelemental.models.AtomicResultProperties` instead.

    """

    def __init__(self, *args, **kwargs):
        from warnings import warn

        warn(
            "ResultProperties has been renamed to AtomicResultProperties and will be removed as soon as v0.13.0",
            DeprecationWarning,
        )
        super().__init__(*args, **kwargs)


class ResultProtocols(AtomicResultProtocols):
    """QC Result Protocols Schema.

    .. deprecated:: 0.12
       Use :py:func:`qcelemental.models.AtomicResultProtocols` instead.

    """

    def __init__(self, *args, **kwargs):
        from warnings import warn

        warn(
            "ResultProtocols has been renamed to AtomicResultProtocols and will be removed as soon as v0.13.0",
            DeprecationWarning,
        )
        super().__init__(*args, **kwargs)


class ResultInput(AtomicInput):
    """QC Input Schema.

    .. deprecated:: 0.12
       Use :py:func:`qcelemental.models.AtomicInput` instead.

    """

    def __init__(self, *args, **kwargs):
        from warnings import warn

        warn("ResultInput has been renamed to AtomicInput and will be removed as soon as v0.13.0", DeprecationWarning)
        super().__init__(*args, **kwargs)


class Result(AtomicResult):
    """QC Result Schema.

    .. deprecated:: 0.12
       Use :py:func:`qcelemental.models.AtomicResult` instead.

    """

    def __init__(self, *args, **kwargs):
        from warnings import warn

        warn("Result has been renamed to AtomicResult and will be removed as soon as v0.13.0", DeprecationWarning)
        super().__init__(*args, **kwargs)
