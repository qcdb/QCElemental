"""
Molecule Object Model
"""

import collections
import hashlib
import json
import warnings
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Literal, Optional, Tuple, Union, cast

import numpy as np
from pydantic import Field, constr, field_validator, model_serializer
from typing_extensions import Annotated

try:
    import nglview
except ModuleNotFoundError:
    # import is purely for forward reference for docs-build. import is not required except for Molecule.show()
    pass

# molparse imports separated b/c https://github.com/python/mypy/issues/7203
from ...molparse.from_arrays import from_arrays
from ...molparse.from_schema import from_schema
from ...molparse.from_string import from_string
from ...molparse.to_schema import to_schema
from ...molparse.to_string import to_string
from ...periodic_table import periodictable
from ...physical_constants import constants
from ...testing import compare, compare_values
from ...util import deserialize, measure_coordinates, msgpackext_loads, provenance_stamp, which_import
from .basemodels import ProtoModel, check_convertible_version, qcschema_draft
from .common_models import Provenance
from .types import Array

if TYPE_CHECKING:
    import qcelemental

    from .common_models import ReprArgs

# Rounding quantities for hashing
GEOMETRY_NOISE = 8
MASS_NOISE = 6
CHARGE_NOISE = 4

_extension_map = {
    ".npy": "numpy",
    ".json": "json",
    ".xyz": "xyz",
    ".psimol": "psi4",
    ".psi4": "psi4",
    ".msgpack": "msgpack-ext",
}


def float_prep(array, around):
    r"""
    Rounds floats to a common value and build positive zeros to prevent hash conflicts.
    """
    if isinstance(array, (list, np.ndarray)):
        # Round array
        array = np.around(array, around)
        # Flip zeros
        array[np.abs(array) < 5 ** (-(around + 1))] = 0

    elif isinstance(array, (float, int)):
        array = round(array, around)
        if array == -0.0:
            array = 0.0
    else:
        raise TypeError("Type '{}' not recognized".format(type(array).__name__))

    return array


NonnegativeInt = Annotated[int, Field(ge=0)]
BondOrderFloat = Annotated[float, Field(ge=0, le=5)]


class Identifiers(ProtoModel):
    r"""Canonical chemical identifiers"""

    molecule_hash: Optional[str] = None
    molecular_formula: Optional[str] = None
    smiles: Optional[str] = None
    inchi: Optional[str] = None
    inchikey: Optional[str] = None
    canonical_explicit_hydrogen_smiles: Optional[str] = None
    canonical_isomeric_explicit_hydrogen_mapped_smiles: Optional[str] = None
    canonical_isomeric_explicit_hydrogen_smiles: Optional[str] = None
    canonical_isomeric_smiles: Optional[str] = None
    canonical_smiles: Optional[str] = None
    pubchem_cid: Optional[str] = Field(None, description="PubChem Compound ID")
    pubchem_sid: Optional[str] = Field(None, description="PubChem Substance ID")
    pubchem_conformerid: Optional[str] = Field(None, description="PubChem Conformer ID")

    model_config = ProtoModel._merge_config_with(serialize_skip_defaults=True)


def molecule_json_schema_extras(schema, model):
    # below addresses the draft-04 issue until https://github.com/samuelcolvin/pydantic/issues/1478 .
    schema["$schema"] = qcschema_draft


class Molecule(ProtoModel):
    r"""
    The physical Cartesian representation of the molecular system.

    A QCSchema representation of a Molecule. This model contains
    data for symbols, geometry, connectivity, charges, fragmentation, etc while also supporting a wide array of I/O and manipulation capabilities.

    Molecule objects geometry, masses, and charges are truncated to 8, 6, and 4 decimal places respectively to assist with duplicate detection.

    Notes
    -----
    All arrays are stored flat but must be reshapable into the dimensions in attribute ``shape``, with abbreviations as follows:

      * nat: number of atomic = calcinfo_natom
      * nfr: number of fragments
      * <varies>: irregular dimension not systematically reshapable

    """

    schema_name: Literal["qcschema_molecule"] = Field(
        "qcschema_molecule", description=(f"The QCSchema specification to which this model conforms.")
    )
    schema_version: Literal[3] = Field(
        3,
        description="The version number of :attr:`~qcelemental.models.Molecule.schema_name` to which this model conforms.",
    )
    validated: bool = Field(  # type: ignore
        False,
        description="A boolean indicator (for speed purposes) that the input Molecule data has been previously checked "
        "for schema (data layout and type) and physics (e.g., non-overlapping atoms, feasible "
        "multiplicity) compliance. This should be False in most cases. A ``True`` setting "
        "should only ever be set by the constructor for this class itself or other trusted sources such as "
        "a Fractal Server or previously serialized Molecules.",
    )

    # Required data
    symbols: Array[str] = Field(  # type: ignore
        ...,
        description="The ordered array of atomic elemental symbols in title case. This field's index "
        "sets atomic order for all other per-atom fields like :attr:`~qcelemental.models.Molecule.real` and the first dimension of "
        ":attr:`~qcelemental.models.Molecule.geometry`. Ghost/virtual atoms must have an entry here in :attr:`~qcelemental.models.Molecule.symbols`; ghostedness is "
        "indicated through the :attr:`~qcelemental.models.Molecule.real` field.",
        json_schema_extra={"shape": ["nat"]},
    )
    geometry: Array[float] = Field(  # type: ignore
        ...,
        description="The ordered array for Cartesian XYZ atomic coordinates [a0]. "
        "Atom ordering is fixed; that is, a consumer who shuffles atoms must not reattach the input "
        "(pre-shuffling) molecule schema instance to any output (post-shuffling) per-atom results "
        "(e.g., gradient). Index of the first dimension matches the 0-indexed indices of all other "
        "per-atom settings like :attr:`~qcelemental.models.Molecule.symbols` and :attr:`~qcelemental.models.Molecule.real`."
        "\n"
        "Serialized storage is always flat, (3*nat,), but QCSchema implementations will want to reshape it. "
        "QCElemental can also accept array-likes which can be mapped to (nat,3) such as a 1-D list of length 3*nat, "
        "or the serialized version of the array in (3*nat,) shape; all forms will be reshaped to "
        "(nat,3) for this attribute.",
        json_schema_extra={"shape": ["nat", 3], "units": "a0"},
    )

    # Molecule data
    name: Optional[str] = Field(  # type: ignore
        None,
        description="Common or human-readable name to assign to this molecule. "
        "This field can be arbitrary; see :attr:`~qcelemental.models.Molecule.identifiers` "
        "for well-defined labels.",
    )
    identifiers: Optional[Identifiers] = Field(  # type: ignore
        None,
        description="An optional dictionary of additional identifiers by which this molecule can be referenced, "
        "such as INCHI, canonical SMILES, etc. See the :class:`~qcelemental.models.results.Identifiers` "
        "model for more details.",
    )
    comment: Optional[str] = Field(  # type: ignore
        None,
        description="Additional comments for this molecule. Intended for pure human/user consumption and clarity.",
    )
    molecular_charge: float = Field(0.0, description="The net electrostatic charge of the molecule.")  # type: ignore
    molecular_multiplicity: float = Field(1, description="The total multiplicity of the molecule.")  # type: ignore

    # Atom data
    masses_: Optional[Array[float]] = Field(  # type: ignore
        None,
        description="The ordered array of atomic masses. Index order matches the 0-indexed indices of all other "
        "per-atom fields like :attr:`~qcelemental.models.Molecule.symbols` "
        "and :attr:`~qcelemental.models.Molecule.real`. "
        "If this is not provided, the mass of each atom is inferred from its most common isotope. "
        "If this is provided, it must be the same length as :attr:`~qcelemental.models.Molecule.symbols` "
        "but can accept ``None`` entries for standard masses to infer from the same index in the "
        ":attr:`~qcelemental.models.Molecule.symbols` field.",
        alias="masses",
        json_schema_extra={"shape": ["nat"], "units": "u"},
    )
    real_: Optional[Array[bool]] = Field(  # type: ignore
        None,
        description="The ordered array indicating if each atom is real (``True``) or ghost/virtual (``False``). "
        "Index matches the 0-indexed indices of all other per-atom settings like "
        ":attr:`~qcelemental.models.Molecule.symbols` and the first dimension of "
        ":attr:`~qcelemental.models.Molecule.geometry`. "
        "If this is not provided, all atoms are assumed to be real (``True``). "
        "If this is provided, the reality or ghostedness of every atom must be specified.",
        alias="real",
        json_schema_extra={
            "shape": ["nat"],
        },
    )
    atom_labels_: Optional[Array[str]] = Field(  # type: ignore
        None,
        description="Additional per-atom labels as an array of strings. Typical use is in model conversions, "
        "such as Elemental <-> Molpro and not typically something which should be user assigned. "
        "See the :attr:`~qcelemental.models.Molecule.comment` field for general human-consumable "
        "text to affix to the molecule.",
        alias="atom_labels",
        json_schema_extra={
            "shape": ["nat"],
        },
    )
    atomic_numbers_: Optional[Array[np.int16]] = Field(  # type: ignore
        None,
        description="An optional ordered 1-D array-like object of atomic numbers of shape (nat,). Index matches the "
        "0-indexed indices of all other per-atom settings like :attr:`~qcelemental.models.Molecule.symbols`"
        " and :attr:`~qcelemental.models.Molecule.real`. Values are inferred from the "
        ":attr:`~qcelemental.models.Molecule.symbols` list if not explicitly set. Ghostedness should be "
        "indicated through :attr:`~qcelemental.models.Molecule.real` field, not zeros here.",
        alias="atomic_numbers",
        json_schema_extra={
            "shape": ["nat"],
        },
    )
    mass_numbers_: Optional[Array[np.int16]] = Field(  # type: ignore
        None,
        description="An optional ordered 1-D array-like object of atomic *mass* numbers of shape (nat). "
        "Index matches the 0-indexed indices of all other per-atom settings like "
        ":attr:`~qcelemental.models.Molecule.symbols` and :attr:`~qcelemental.models.Molecule.real`. "
        "Values are inferred from the most common isotopes of the "
        ":attr:`~qcelemental.models.Molecule.symbols` list if not explicitly set. "
        "If single isotope not (yet) known for an atom, -1 is placeholder.",
        alias="mass_numbers",
        json_schema_extra={
            "shape": ["nat"],
        },
    )

    # Fragment and connection data
    connectivity_: Optional[List[Tuple[NonnegativeInt, NonnegativeInt, BondOrderFloat]]] = Field(  # type: ignore
        None,
        description="A list of bonds within the molecule. "
        "Each entry is a tuple of ``(atom_index_A, atom_index_B, bond_order)`` where the ``atom_index`` "
        "matches the 0-indexed indices of all other per-atom settings like "
        ":attr:`~qcelemental.models.Molecule.symbols` and :attr:`~qcelemental.models.Molecule.real`. "
        "Bonds may be freely reordered and inverted.",
        alias="connectivity",
        min_length=1,
    )
    fragments_: Optional[List[Array[np.int32]]] = Field(  # type: ignore
        None,
        description="List of indices grouping atoms (0-indexed) into molecular fragments within the molecule. "
        "Each entry in the outer list is a new fragment; index matches the ordering in "
        ":attr:`~qcelemental.models.Molecule.fragment_charges` and "
        ":attr:`~qcelemental.models.Molecule.fragment_multiplicities`. "
        "Inner lists are 0-indexed atoms which compose the fragment; every atom must be in exactly one "
        "inner list. Noncontiguous fragments are allowed, though no QM program is known to support them. "
        "Fragment ordering is fixed; that is, a consumer who shuffles fragments must not reattach the "
        "input (pre-shuffling) molecule schema instance to any output (post-shuffling) per-fragment "
        "results (e.g., n-body energy arrays).",
        alias="fragments",
        json_schema_extra={
            "shape": ["nfr", "<varies>"],
        },
    )
    fragment_charges_: Optional[List[float]] = Field(  # type: ignore
        None,
        description="The total charge of each fragment in the :attr:`~qcelemental.models.Molecule.fragments` list. "
        "The index of this list matches the 0-index indices of "
        ":attr:`~qcelemental.models.Molecule.fragments` list. Will be filled in based on a set of rules if "
        "not provided (and :attr:`~qcelemental.models.Molecule.fragments` are specified).",
        alias="fragment_charges",
        json_schema_extra={
            "shape": ["nfr"],
        },
    )
    fragment_multiplicities_: Optional[List[float]] = Field(  # type: ignore
        None,
        description="The multiplicity of each fragment in the :attr:`~qcelemental.models.Molecule.fragments` list. "
        "The index of this list matches the 0-index indices of "
        ":attr:`~qcelemental.models.Molecule.fragments` list. Will be filled in based on a set of rules if "
        "not provided (and :attr:`~qcelemental.models.Molecule.fragments` are specified).",
        alias="fragment_multiplicities",
        json_schema_extra={
            "shape": ["nfr"],
        },
    )

    # Orientation
    fix_com: bool = Field(  # type: ignore
        False,
        description="Whether translation of geometry is allowed (fix F) or disallowed (fix T)."
        "When False, QCElemental will pre-process the Molecule object to translate the center of mass "
        "to (0,0,0) in Euclidean coordinate space, resulting in a different :attr:`~qcelemental.models.Molecule.geometry` than the "
        "one provided. 'Fix' is used in the sense of 'specify': that is, `fix_com=True` signals that "
        "the origin in `geometry` is a deliberate part of the Molecule spec, whereas `fix_com=False` "
        "(default) allows that the origin is happenstance and may be adjusted. "
        "guidance: A consumer who translates the geometry must not reattach the input (pre-translation) molecule schema instance to any output (post-translation) origin-sensitive results (e.g., an ordinary energy when EFP present).",
    )
    fix_orientation: bool = Field(  # type: ignore
        False,
        description="Whether rotation of geometry is allowed (fix F) or disallowed (fix T). "
        "When False, QCElemental will pre-process the Molecule object to orient via the intertial tensor, "
        "resulting in a different :attr:`~qcelemental.models.Molecule.geometry` than the one provided. "
        "'Fix' is used in the sense of 'specify': that is, `fix_orientation=True` signals that "
        "the frame orientation in `geometry` is a deliberate part of the Molecule spec, whereas "
        "`fix_orientation=False` (default) allows that the frame is happenstance and may be adjusted. "
        "guidance: A consumer who rotates the geometry must not reattach the input (pre-rotation) molecule schema instance to any output (post-rotation) frame-sensitive results (e.g., molecular vibrations).",
    )
    fix_symmetry: Optional[str] = Field(  # type: ignore
        None,
        description="Maximal point group symmetry which :attr:`~qcelemental.models.Molecule.geometry` should be treated. Lowercase.",
    )
    # Extra
    provenance: Provenance = Field(
        default_factory=partial(provenance_stamp, __name__),
        description="The provenance information about how this Molecule (and its attributes) were generated, "
        "provided, and manipulated.",
        validate_default=True,  # Force casting provenance to dict
    )
    id: Optional[Any] = Field(  # type: ignore
        None,
        description="A unique identifier for this Molecule object. This field exists primarily for Databases "
        "(e.g. Fractal's Server) to track and lookup this specific object and should virtually "
        "never need to be manually set.",
    )
    extras: Dict[str, Any] = Field(  # type: ignore
        {},
        description="Additional information to bundle with the molecule. Use for schema development and scratch space.",
    )

    model_config = ProtoModel._merge_config_with(
        serialize_skip_defaults=True,
        json_schema_extra=molecule_json_schema_extras,
        repr_style=lambda self: [
            ("name", self.name),
            ("formula", self.get_molecular_formula(chgmult=True)),
            ("hash", self.get_hash()[:7]),
        ],
    )
    # Alias fields are handled with the Field objects above

    def __init__(self, orient: bool = False, validate: Optional[bool] = None, **kwargs: Any) -> None:
        r"""Initializes the molecule object from dictionary-like values.

        Parameters
        ----------
        orient
            If True, orientates the Molecule to a common reference frame.
        validate
            If ``None`` validation is always applied unless the ``validated`` flag is set. Otherwise uses the boolean to decide to validate the Molecule or not.
        **kwargs
            The values of the Molecule object attributes.
        """
        if validate is None:
            validate = not kwargs.get("validated", False)

        geometry_prep = kwargs.pop("_geometry_prep", False)
        geometry_noise = kwargs.pop("geometry_noise", GEOMETRY_NOISE)

        if validate:
            kwargs["schema_name"] = kwargs.pop("schema_name", "qcschema_molecule")
            kwargs["schema_version"] = kwargs.pop("schema_version", 3)
            # original_keys = set(kwargs.keys())  # revive when ready to revisit sparsity

            nonphysical = kwargs.pop("nonphysical", False)
            schema = to_schema(
                from_schema(kwargs, nonphysical=nonphysical), dtype=kwargs["schema_version"], copy=False, np_out=True
            )
            schema = _filter_defaults(schema)

            kwargs["validated"] = True
            kwargs = {**kwargs, **schema}  # Allow any extra fields
            validate = True

        if "extras" not in kwargs or kwargs["extras"] is None:  # latter re-defaults to empty dict
            kwargs["extras"] = {}
        super().__init__(**kwargs)

        # We are pulling out the values *explicitly* so that the pydantic skip_defaults works as expected
        # All attributes set below are equivalent to the default set.
        values = self.__dict__

        if validate:
            # Title case for consistency
            if np.lib.NumpyVersion(np.__version__) >= "2.0.0b1":
                values["symbols"] = np.char.chararray.title(self.symbols)
            else:
                values["symbols"] = np.core.defchararray.title(self.symbols)

        if orient:
            values["geometry"] = float_prep(self._orient_molecule_internal(), geometry_noise)
        elif validate or geometry_prep:
            values["geometry"] = float_prep(values["geometry"], geometry_noise)

    @field_validator("geometry")
    @classmethod
    def _must_be_3n(cls, v, info):
        n = len(info.data["symbols"])
        try:
            v = v.reshape(n, 3)
        except (ValueError, AttributeError):
            raise ValueError("Geometry must be castable to shape (N,3)!")
        return v

    @field_validator("masses_", "real_")
    @classmethod
    def _must_be_n(cls, v, info):
        n = len(info.data["symbols"])
        if len(v) != n:
            raise ValueError("Masses and Real must be same number of entries as Symbols")
        return v

    @field_validator("real_")
    @classmethod
    def _populate_real(cls, v, info):
        # Can't use geometry here since its already been validated and not in values
        n = len(info.data["symbols"])
        if len(v) == 0:
            v = np.array([True for _ in range(n)])
        return v

    @field_validator("fragment_charges_")
    @classmethod
    def _must_be_n_frag(cls, v, info):
        if "fragments_" in info.data and info.data["fragments_"] is not None:
            n = len(info.data["fragments_"])
            if len(v) != n:
                raise ValueError("Fragment Charges must be same number of entries as Fragments")
        return v

    @field_validator("fragment_multiplicities_")
    def _must_be_n_frag_mult(cls, v, info):
        if "fragments_" in info.data and info.data["fragments_"] is not None:
            n = len(info.data["fragments_"])
            if len(v) != n:
                raise ValueError("Fragment Multiplicities must be same number of entries as Fragments")
        v = [(int(m) if m.is_integer() else m) for m in v]
        if any([m < 1.0 for m in v]):
            raise ValueError(f"Fragment Multiplicity must be positive: {v}")
        return v

    @field_validator("molecular_multiplicity")
    def _int_if_possible(cls, v, info):
        if v.is_integer():
            # preserve existing hashes
            v = int(v)
        if v < 1.0:
            raise ValueError("Molecular Multiplicity must be positive")
        return v

    @property
    def hash_fields(self):
        return [
            "symbols",
            "masses",
            "molecular_charge",
            "molecular_multiplicity",
            "real",
            "geometry",
            "fragments",
            "fragment_charges",
            "fragment_multiplicities",
            "connectivity",
        ]

    @property
    def masses(self) -> Array[float]:
        masses = self.__dict__.get("masses_")
        if masses is None:
            masses = np.array([periodictable.to_mass(x) for x in self.symbols])
        return masses

    @property
    def real(self) -> Array[bool]:
        real = self.__dict__.get("real_")
        if real is None:
            real = np.array([True for x in self.symbols])
        return real

    @property
    def atom_labels(self) -> Array[str]:
        atom_labels = self.__dict__.get("atom_labels_")
        if atom_labels is None:
            atom_labels = np.array(["" for x in self.symbols])
        return atom_labels

    @property
    def atomic_numbers(self) -> Array[np.int16]:
        atomic_numbers = self.__dict__.get("atomic_numbers_")
        if atomic_numbers is None:
            atomic_numbers = np.array([periodictable.to_Z(x) for x in self.symbols])
        return atomic_numbers

    @property
    def mass_numbers(self) -> Array[np.int16]:
        mass_numbers = self.__dict__.get("mass_numbers_")
        if mass_numbers is None:
            mass_numbers = np.array([periodictable.to_A(x) for x in self.symbols])
        return mass_numbers

    @property
    def connectivity(self) -> List[Tuple[int, int, float]]:
        connectivity = self.__dict__.get("connectivity_")
        # default is None, not []
        return connectivity

    @property
    def fragments(self) -> List[Array[np.int32]]:
        fragments = self.__dict__.get("fragments_")
        if fragments is None:
            fragments = [np.arange(len(self.symbols), dtype=np.int32)]
        return fragments

    @property
    def fragment_charges(self) -> List[float]:
        fragment_charges = self.__dict__.get("fragment_charges_")
        if fragment_charges is None:
            fragment_charges = [self.molecular_charge]
        return fragment_charges

    @property
    def fragment_multiplicities(self) -> List[float]:
        fragment_multiplicities = self.__dict__.get("fragment_multiplicities_")
        if fragment_multiplicities is None:
            fragment_multiplicities = [self.molecular_multiplicity]
        return fragment_multiplicities

    ### Non-Pydantic API functions

    def show(self, ngl_kwargs: Optional[Dict[str, Any]] = None) -> "nglview.NGLWidget":  # type: ignore
        r"""Creates a 3D representation of a molecule that can be manipulated in Jupyter Notebooks and exported as
        images (`.png`).

        Parameters
        ----------
        ngl_kwargs
            Addition nglview NGLWidget kwargs

        Returns
        -------
        nglview.NGLWidget
            A nglview view of the molecule

        """
        if not which_import("nglview", return_bool=True):
            raise ModuleNotFoundError(
                f"Python module nglwview not found. Solve by installing it: `conda install -c conda-forge nglview`"
            )  # pragma: no cover

        import nglview as nv  # type: ignore

        if ngl_kwargs is None:
            ngl_kwargs = {}

        structure = nv.TextStructure(self.to_string("nglview-sdf"), ext="sdf")
        widget = nv.NGLWidget(structure, **ngl_kwargs)
        return widget

    def measure(
        self, measurements: Union[List[int], List[List[int]]], *, degrees: bool = True
    ) -> Union[float, List[float]]:
        r"""
        Takes a measurement of the moleucle from the indicies provided.

        Parameters
        ----------
        measurements
            Either a single list of indices or multiple. Return a distance, angle, or dihedral depending if
            2, 3, or 4 indices is provided, respectively. Values are returned in Bohr (distance) or degree.
        degrees
            Returns degrees by default, radians otherwise.

        Returns
        -------
        Union[float, List[float]]
            Either a value or list of the measured values.
        """

        return measure_coordinates(self.geometry, measurements, degrees=degrees)

    def orient_molecule(self):
        r"""
        Centers the molecule and orients via inertia tensor before returning a new Molecule
        """
        return Molecule(orient=True, **self.model_dump())

    def compare(self, other):
        warnings.warn(
            "Molecule.compare is deprecated and will be removed in v0.13.0. Use == instead.", DeprecationWarning
        )
        return self == other

    def __eq__(self, other):
        r"""
        Checks if two molecules are identical. This is a molecular identity defined
        by scientific terms, and not programing terms, so it's less rigorous than
        a programmatic equality or a memory equivalent `is`.
        """
        import qcelemental

        if isinstance(other, dict):
            other = Molecule(orient=False, **other)
        elif isinstance(other, (Molecule, qcelemental.models.v1.Molecule)):
            # allow v2 on grounds of "scientific, not programming terms"
            pass
        else:
            raise TypeError("Comparison molecule not understood of type '{}'.".format(type(other)))

        return self.get_hash() == other.get_hash()

    # UNCOMMENT IF NEEDED FOR UPGRADE REDO??
    def dict(self, **kwargs):
        warnings.warn("The `dict` method is deprecated; use `model_dump` instead.", DeprecationWarning)
        return self.model_dump(**kwargs)
        # TODO maybe bad idea as dict(v2) does non-recursive dictionary, whereas model_dump does nested

    @model_serializer(mode="wrap")
    def _serialize_molecule(self, handler) -> Dict[str, Any]:
        default_result = handler(self)
        output_dict = {}
        for key, value in default_result.items():
            # Could do this as a single comprehension dict, but this is easier to read
            # Handle exclude unset is always true
            if key not in self.model_fields_set:
                continue
            # Handle "by_alias" is always true
            alias = self.model_fields[key].alias
            output_key = alias if alias is not None else key
            output_dict[output_key] = value
        return output_dict

    def pretty_print(self):
        r"""Print the molecule in Angstroms. Same as :py:func:`print_out` only always in Angstroms.
        (method name in libmints is print_in_angstrom)

        """
        text = ""

        text += """    Geometry (in {0:s}), charge = {1:.1f}, multiplicity = {2:d}:\n\n""".format(
            "Angstrom", self.molecular_charge, self.molecular_multiplicity
        )
        text += """       Center              X                  Y                   Z       \n"""
        text += """    ------------   -----------------  -----------------  -----------------\n"""

        for i in range(len(self.geometry)):
            text += """    {0:8s}{1:4s} """.format(self.symbols[i], "" if self.real[i] else "(Gh)")
            for j in range(3):
                text += """  {0:17.12f}""".format(
                    self.geometry[i][j] * constants.conversion_factor("bohr", "angstroms")
                )
            text += "\n"
        # text += "\n"

        return text

    def get_fragment(
        self,
        real: Union[int, List],
        ghost: Optional[Union[int, List]] = None,
        orient: bool = False,
        group_fragments: bool = True,
    ) -> "Molecule":
        r"""Get new Molecule with fragments preserved, dropped, or ghosted.

        Parameters
        ----------
        real
            Fragment index or list of indices (0-indexed) to be real atoms in new Molecule.
        ghost
            Fragment index or list of indices (0-indexed) to be ghost atoms (basis fns only) in new Molecule.
        orient
            Whether or not to align (inertial frame) and phase geometry upon new Molecule instantiation
            (according to _orient_molecule_internal)?
        group_fragments
            Whether or not to group real fragments at the start of the atom list and ghost fragments toward the back.
            Previous to ``v0.5``, this was always effectively True. True is handy for finding duplicate
            (atom-order-independent) molecules by hash. False preserves fragment order (though collapsing gaps for
            absent fragments) like Psi4's ``extract_subsets``. False is handy for gradients where atom order of
            returned values matters.

        Returns
        -------
        Molecule
            New qcelemental.models.Molecule with ``self``\'s fragments present, ghosted, or absent.

        """
        if isinstance(real, int):
            real = [real]

        if isinstance(ghost, int):
            ghost = [ghost]
        elif ghost is None:
            ghost = []

        constructor_dict: Dict = {}

        ret_name = (self.name if self.name is not None else "") + " (" + str(real) + "," + str(ghost) + ")"
        constructor_dict["name"] = ret_name
        # ret = Molecule(None, name=ret_name)

        if len(set(real) & set(ghost)):
            raise TypeError(
                "Molecule:get_fragment: real and ghost sets are overlapping! ({0}, {1}).".format(str(real), str(ghost))
            )

        geom_blocks = []
        symbols = []
        masses = []
        real_atoms = []
        fragments = []
        fragment_charges = []
        fragment_multiplicities = []
        atom_size = 0

        if group_fragments:
            # Loop through the real blocks
            frag_start = 0
            for frag in real:
                frag_size = len(self.fragments[frag])
                geom_blocks.append(self.geometry[self.fragments[frag]])

                for idx in self.fragments[frag]:
                    symbols.append(self.symbols[idx])
                    real_atoms.append(True)
                    masses.append(self.masses[idx])

                fragments.append(list(range(frag_start, frag_start + frag_size)))
                frag_start += frag_size

                fragment_charges.append(float(self.fragment_charges[frag]))
                fragment_multiplicities.append(self.fragment_multiplicities[frag])

            # Set charge and multiplicity
            constructor_dict["molecular_charge"] = sum(fragment_charges)
            constructor_dict["molecular_multiplicity"] = sum(x - 1 for x in fragment_multiplicities) + 1

            # Loop through the ghost blocks
            for frag in ghost:
                frag_size = len(self.fragments[frag])
                geom_blocks.append(self.geometry[self.fragments[frag]])

                for idx in self.fragments[frag]:
                    symbols.append(self.symbols[idx])
                    real_atoms.append(False)
                    masses.append(self.masses[idx])

                fragments.append(list(range(frag_start, frag_start + frag_size)))
                frag_start += frag_size

                fragment_charges.append(0)
                fragment_multiplicities.append(1)

        else:
            # List[Array[np.int32]]
            at2fr: List[Union[int, None]] = [None] * len(self.symbols)
            for ifr, fr in enumerate(self.fragments):
                for iat in fr:
                    at2fr[iat] = ifr

            at2at: List[Union[int, None]] = [None] * len(self.symbols)
            for iat in range(len(self.symbols)):
                ifr = at2fr[iat]

                if ifr in real or ifr in ghost:
                    geom_blocks.append(self.geometry[iat])
                    symbols.append(self.symbols[iat])
                    real_atoms.append(ifr in real)
                    masses.append(self.masses[iat])

                    at2at[iat] = atom_size
                    atom_size += 1

                else:
                    at2at[iat] = None

            for ifr, fr in enumerate(self.fragments):
                if ifr in real or ifr in ghost:
                    fragments.append([at2at[iat] for iat in fr])

                if ifr in real:
                    fragment_charges.append(self.fragment_charges[ifr])
                    fragment_multiplicities.append(self.fragment_multiplicities[ifr])

                elif ifr in ghost:
                    fragment_charges.append(0)
                    fragment_multiplicities.append(1)

            assert None not in fragments

        constructor_dict["fragments"] = fragments
        constructor_dict["fragment_charges"] = fragment_charges
        constructor_dict["fragment_multiplicities"] = fragment_multiplicities
        constructor_dict["symbols"] = symbols
        constructor_dict["geometry"] = np.vstack(geom_blocks)
        constructor_dict["real"] = real_atoms
        constructor_dict["masses"] = masses

        return Molecule(orient=orient, **constructor_dict)

    def to_string(  # type: ignore
        self,
        dtype: str,
        units: str = None,
        *,
        atom_format: str = None,
        ghost_format: str = None,
        width: int = 17,
        prec: int = 12,
        return_data: bool = False,
    ):
        r"""Returns a string that can be used by a variety of programs.

        Unclear if this will be removed or renamed to "to_psi4_string" in the future

        Suggest psi4 --> psi4frag and psi4 route to to_string
        """
        molrec = from_schema(self.model_dump(), nonphysical=True)
        return to_string(
            molrec,
            dtype=dtype,
            units=units,
            atom_format=atom_format,
            ghost_format=ghost_format,
            width=width,
            prec=prec,
            return_data=return_data,
        )

    def get_hash(self):
        r"""
        Returns the hash of the molecule.
        """

        m = hashlib.sha1()
        concat = ""

        for field in self.hash_fields:
            data = getattr(self, field)
            if field == "geometry":
                data = float_prep(data, GEOMETRY_NOISE)
            elif field in ["fragment_charges", "molecular_charge", "fragment_multiplicities", "molecular_multiplicity"]:
                data = float_prep(data, CHARGE_NOISE)
            elif field == "masses":
                data = float_prep(data, MASS_NOISE)

            concat += json.dumps(data, default=lambda x: x.ravel().tolist())

        m.update(concat.encode("utf-8"))
        return m.hexdigest()

    def get_molecular_formula(self, order: str = "alphabetical", chgmult: bool = False) -> str:
        r"""
        Returns the molecular formula for a molecule.

        Parameters
        ----------
        order: str, optional
            Sorting order of the formula. Valid choices are "alphabetical" and "hill".
        chgmult
            If not neutral singlet, return formula as {mult}^{formula}{chg}.

        Returns
        -------
        str
            The molecular formula.

        Examples
        --------

        >>> methane = qcelemental.models.Molecule('''
        ... H      0.5288      0.1610      0.9359
        ... C      0.0000      0.0000      0.0000
        ... H      0.2051      0.8240     -0.6786
        ... H      0.3345     -0.9314     -0.4496
        ... H     -1.0685     -0.0537      0.1921
        ... ''')
        >>> methane.get_molecular_formula()
        CH4

        >>> hcl = qcelemental.models.Molecule('''
        ... H      0.0000      0.0000      0.0000
        ... Cl     0.0000      0.0000      1.2000
        ... ''')
        >>> hcl.get_molecular_formula()
        ClH

        >>> two_pentanol_radcat = qcelemental.models.Molecule('''
        ... 1 2
        ... C         -4.43914        1.67538       -0.14135
        ... C         -2.91385        1.70652       -0.10603
        ... H         -4.82523        2.67391       -0.43607
        ... H         -4.84330        1.41950        0.86129
        ... H         -4.79340        0.92520       -0.88015
        ... H         -2.59305        2.48187        0.62264
        ... H         -2.53750        1.98573       -1.11429
        ... C         -2.34173        0.34025        0.29616
        ... H         -2.72306        0.06156        1.30365
        ... C         -0.80326        0.34498        0.31454
        ... H         -2.68994       -0.42103       -0.43686
        ... O         -0.32958        1.26295        1.26740
        ... H         -0.42012        0.59993       -0.70288
        ... C         -0.26341       -1.04173        0.66218
        ... H         -0.61130       -1.35318        1.67053
        ... H          0.84725       -1.02539        0.65807
        ... H         -0.60666       -1.78872       -0.08521
        ... H         -0.13614        2.11102        0.78881
        ... ''')
        >>> two_pentanol_radcat.get_molecular_formula(chgmult=True)
        2^C5H12O+

        Notes
        -----
        This includes all atoms in the molecule, including ghost atoms. See :py:meth:`element_composition` to exclude.

        """

        from ...molutil import molecular_formula_from_symbols

        formula = molecular_formula_from_symbols(symbols=self.symbols, order=order)

        c, m = self.molecular_charge, self.molecular_multiplicity
        if not chgmult or (c == 0.0 and m == 1):
            return formula

        if m > 1:
            formula = f"{m}^{formula}"
        if c < 0.0:
            formula += abs(int(c)) * "-"
        elif c > 0.0:
            formula += int(c) * "+"
        return formula

    ### Constructors

    @classmethod
    def from_data(
        cls,
        data: Union[str, Dict[str, Any], np.ndarray, bytes],
        dtype: Optional[str] = None,
        *,
        orient: bool = False,
        validate: bool = None,
        **kwargs: Dict[str, Any],
    ) -> "Molecule":
        r"""
        Constructs a molecule object from a data structure.

        Parameters
        ----------
        data
            Data to construct Molecule from
        dtype
            How to interpret the data, if not passed attempts to discover this based on input type.
        orient
            Orientates the molecule to a standard frame or not.
        validate
            Validates the molecule or not.
        **kwargs
            Additional kwargs to pass to the constructors. kwargs take precedence over data.

        Returns
        -------
        Molecule
            A constructed molecule class.

        """
        if dtype is None:
            if isinstance(data, str):
                dtype = "string"
            elif isinstance(data, np.ndarray):
                dtype = "numpy"
            elif isinstance(data, dict):
                dtype = "dict"
            elif isinstance(dtype, bytes):
                dtype = "msgpack"
            else:
                raise TypeError("Input type not understood, please supply the 'dtype' kwarg.")

        if dtype in ["string", "psi4", "xyz", "xyz+"]:
            mol_dict = from_string(data, dtype if dtype != "string" else None)
            assert isinstance(mol_dict, dict)
            input_dict = to_schema(mol_dict["qm"], dtype=3, np_out=True)
            input_dict = _filter_defaults(input_dict)
            input_dict["validated"] = True
            input_dict["_geometry_prep"] = True
        elif dtype == "numpy":
            data = np.asarray(data)
            data = {
                "geom": data[:, 1:],
                "elez": data[:, 0],
                "units": kwargs.pop("units", "Angstrom"),
                "fragment_separators": kwargs.pop("frags", []),
            }
            input_dict = to_schema(from_arrays(**data), dtype=3, np_out=True)
            input_dict = _filter_defaults(input_dict)
            input_dict["validated"] = True
            input_dict["_geometry_prep"] = True
        elif dtype == "msgpack":
            assert isinstance(data, bytes)
            input_dict = msgpackext_loads(data)
        elif dtype == "json":
            assert isinstance(data, str)
            input_dict = json.loads(data)
        elif dtype == "dict":
            assert isinstance(data, dict)
            input_dict = data
        else:
            raise KeyError("Dtype not understood '{}'.".format(dtype))

        input_dict.update(kwargs)

        # if charge/spin options are given, invalidate charge and spin options that are missing
        charge_spin_opts = {"molecular_charge", "fragment_charges", "molecular_multiplicity", "fragment_multiplicities"}
        kwarg_keys = set(kwargs.keys())
        if len(charge_spin_opts & kwarg_keys) > 0:
            for key in charge_spin_opts - kwarg_keys:
                input_dict.pop(key, None)
            input_dict.pop("validated", None)

        return cls(orient=orient, validate=validate, **input_dict)

    @classmethod
    def from_file(cls, filename: str, dtype: Optional[str] = None, *, orient: bool = False, **kwargs):
        r"""
        Constructs a molecule object from a file.

        Parameters
        ----------
        filename
            The filename to build
        dtype
            The type of file to interpret.
        orient
            Orientates the molecule to a standard frame or not.
        **kwargs
            Any additional keywords to pass to the constructor

        Returns
        -------
        Molecule
            A constructed molecule class.

        """

        ext = Path(filename).suffix

        if dtype is None:
            if ext in _extension_map:
                dtype = _extension_map[ext]
            else:
                # Let `from_string` try to sort it
                dtype = "string"

        # Raw string type, read and pass through
        if dtype in ["string", "xyz", "xyz+", "psi4"]:
            with open(filename, "r") as infile:
                data = infile.read()
        elif dtype == "numpy":
            data = np.load(filename)
        elif dtype in ["json", "json-ext"]:
            with open(filename, "r") as infile:
                data = deserialize(infile.read(), encoding="json-ext")
            dtype = "dict"
        elif dtype in ["msgpack", "msgpack-ext"]:
            with open(filename, "rb") as infile_bytes:
                data = deserialize(infile_bytes.read(), encoding="msgpack-ext")
            dtype = "dict"
        else:
            raise KeyError("Dtype not understood '{}'.".format(dtype))

        return cls.from_data(data, dtype, orient=orient, **kwargs)

    def to_file(self, filename: str, dtype: Optional[str] = None) -> None:
        r"""Writes the Molecule to a file.

        Parameters
        ----------
        filename
            The filename to write to
        dtype
            The type of file to write, attempts to infer dtype from the filename if not provided.

        """
        ext = Path(filename).suffix

        if dtype is None:
            if ext in _extension_map:
                dtype = _extension_map[ext]
            else:
                raise KeyError(f"Could not infer dtype from filename: `{filename}`")

        if dtype in ["xyz", "xyz+", "psi4"]:
            stringified = self.to_string(dtype)
        elif dtype in ["json", "json-ext", "msgpack", "msgpack-ext"]:
            stringified = self.serialize(dtype)
        elif dtype in ["numpy"]:
            elements = np.array(self.atomic_numbers).reshape(-1, 1)
            npmol = np.hstack((elements, self.geometry * constants.conversion_factor("bohr", "angstroms")))
            np.save(filename, npmol)
            return
        else:
            raise KeyError(f"Dtype `{dtype}` is not valid")

        flags = "wb" if dtype.startswith("msgpack") else "w"

        with open(filename, flags) as handle:
            handle.write(stringified)

    ### Non-Pydantic internal functions

    def _orient_molecule_internal(self):
        r"""
        Centers the molecule and orients via inertia tensor before returning a new set of the
        molecule geometry
        """

        new_geometry = self.geometry.copy()  # Ensure we get a copy
        # Get the mass as an array
        # Masses are needed for orientation
        np_mass = np.array(self.masses)

        # Center on Mass
        new_geometry -= np.average(new_geometry, axis=0, weights=np_mass)

        # Rotate into inertial frame
        tensor = self._inertial_tensor(new_geometry, weight=np_mass)
        _, evecs = np.linalg.eigh(tensor)

        new_geometry = np.dot(new_geometry, evecs)

        # Phases? Lets do the simplest thing and ensure the first atom in each column
        # that is not on a plane is positve

        phase_check = [False, False, False]

        geom_noise = 10 ** (-GEOMETRY_NOISE)
        for num in range(new_geometry.shape[0]):
            for x in range(3):
                if phase_check[x]:
                    continue

                val = new_geometry[num, x]

                if abs(val) < geom_noise:
                    continue

                phase_check[x] = True

                if val < 0:
                    new_geometry[:, x] *= -1

            if sum(phase_check) == 3:
                break
        return new_geometry

    def __repr_args__(self) -> "ReprArgs":
        return [
            ("name", self.name),
            ("formula", self.get_molecular_formula(chgmult=True)),
            ("hash", self.get_hash()[:7]),
        ]

    def _ipython_display_(self, **kwargs) -> None:
        try:
            self.show()._ipython_display_(**kwargs)
        except ModuleNotFoundError:
            from IPython.display import display

            display(f"Install nglview for interactive visualization.", f"{repr(self)}")

    @staticmethod
    def _inertial_tensor(geom, *, weight):
        r"""
        Compute the moment inertia tensor for a given geometry.
        """
        # Build inertia tensor
        tensor = np.zeros((3, 3))

        # Diagonal
        tensor[0][0] = np.sum(weight * (geom[:, 1] ** 2.0 + geom[:, 2] ** 2.0))
        tensor[1][1] = np.sum(weight * (geom[:, 0] ** 2.0 + geom[:, 2] ** 2.0))
        tensor[2][2] = np.sum(weight * (geom[:, 0] ** 2.0 + geom[:, 1] ** 2.0))

        # I(alpha, beta)
        # Off diagonal
        tensor[1][0] = tensor[0][1] = -1.0 * np.sum(weight * geom[:, 0] * geom[:, 1])
        tensor[2][0] = tensor[0][2] = -1.0 * np.sum(weight * geom[:, 0] * geom[:, 2])
        tensor[2][1] = tensor[1][2] = -1.0 * np.sum(weight * geom[:, 1] * geom[:, 2])
        return tensor

    def nuclear_repulsion_energy(self, ifr: int = None, real_only: bool = True) -> float:
        r"""Nuclear repulsion energy.

        Parameters
        ----------
        ifr
            If not `None`, only compute for the `ifr`-th (0-indexed) fragment.
        real_only
            Only include real atoms in the sum.

        Returns
        -------
        nre : float
            Nuclear repulsion energy in entire molecule or in fragment.

        """
        if real_only:
            Zeff = [z * int(real) for z, real in zip(cast(Iterable[int], self.atomic_numbers), self.real)]
        else:
            Zeff = self.atomic_numbers
        atoms = list(range(self.geometry.shape[0]))

        if ifr is not None:
            atoms = self.fragments[ifr]

        nre = 0.0
        for iat1, at1 in enumerate(atoms):
            for at2 in atoms[:iat1]:
                dist = np.linalg.norm(self.geometry[at1] - self.geometry[at2])
                nre += Zeff[at1] * Zeff[at2] / dist
        return nre

    def nelectrons(self, ifr: int = None, real_only: bool = True) -> int:
        r"""Number of electrons.

        Parameters
        ----------
        ifr
            If not `None`, only compute for the `ifr`-th (0-indexed) fragment.
        real_only
            Only include real atoms in the sum.

        Returns
        -------
        nelec : int
            Number of electrons in entire molecule or in fragment.

        """
        if real_only:
            Zeff = [z * int(real) for z, real in zip(cast(Iterable[int], self.atomic_numbers), self.real)]
        else:
            Zeff = self.atomic_numbers

        if ifr is None:
            nel = sum(Zeff) - self.molecular_charge

        else:
            nel = sum([zf for iat, zf in enumerate(Zeff) if iat in self.fragments[ifr]]) - self.fragment_charges[ifr]

        return int(nel)

    def molecular_weight(self, ifr: int = None, real_only: bool = True) -> float:
        r"""Molecular weight in uamu.

        Parameters
        ----------
        ifr
            If not `None`, only compute for the `ifr`-th (0-indexed) fragment.
        real_only
            Only include real atoms in the sum.

        Returns
        -------
        mw : float
            Molecular weight in entire molecule or in fragment.

        """
        if real_only:
            masses = [mas * int(real) for mas, real in zip(cast(Iterable[float], self.masses), self.real)]
        else:
            masses = self.masses

        if ifr is None:
            mw = sum(masses)

        else:
            mw = sum([mas for iat, mas in enumerate(masses) if iat in self.fragments[ifr]])

        return mw

    def element_composition(self, ifr: int = None, real_only: bool = True) -> Dict[str, int]:
        r"""Atomic count map.

        Parameters
        ----------
        ifr
            If not `None`, only compute for the `ifr`-th (0-indexed) fragment.
        real_only
            Only include real atoms.

        Returns
        -------
        composition : Dict[str, int]
            Atomic count map.

        Notes
        -----
        This excludes ghost atoms by default whereas get_molecular_formula always includes them.

        """
        if real_only:
            symbols = [sym * int(real) for sym, real in zip(cast(Iterable[str], self.symbols), self.real)]
        else:
            symbols = self.symbols

        if ifr is None:
            count = collections.Counter(sym.title() for sym in symbols)

        else:
            count = collections.Counter(sym.title() for iat, sym in enumerate(symbols) if iat in self.fragments[ifr])

        count.pop("", None)
        return dict(count)

    def align(
        self,
        ref_mol: "Molecule",
        *,
        do_plot: bool = False,
        verbose: int = 0,
        atoms_map: bool = False,
        run_resorting: bool = False,
        mols_align: Union[bool, float] = False,
        run_to_completion: bool = False,
        uno_cutoff: float = 1.0e-3,
        run_mirror: bool = False,
        generic_ghosts: bool = False,
    ) -> Tuple["Molecule", Dict[str, Any]]:
        r"""Finds shift, rotation, and atom reordering of `concern_mol` (self)
        that best aligns with `ref_mol`.

        Wraps :py:func:`qcelemental.molutil.B787` for :py:class:`qcelemental.models.Molecule`.
        Employs the Kabsch, Hungarian, and Uno algorithms to exhaustively locate
        the best alignment for non-oriented, non-ordered structures.

        Parameters
        ----------
        ref_mol : qcelemental.models.Molecule
            Molecule to match.
        atoms_map
            Whether atom1 of `ref_mol` corresponds to atom1 of `concern_mol`, etc.
            If true, specifying `True` can save much time.
        mols_align
            Whether ref_mol and concern_mol have identical geometries
            (barring orientation or atom mapping) and expected final RMSD = 0.
            If `True`, procedure is truncated when RMSD condition met, saving time.
            If float, RMSD tolerance at which search for alignment stops. If provided,
            the alignment routine will throw an error if it fails to align
            the molecule within the specified RMSD tolerance.
        do_plot
            Pops up a mpl plot showing before, after, and ref geometries.
        run_to_completion
            Run reorderings to completion (past RMSD = 0) even if unnecessary because
            `mols_align=True`. Used to test worst-case timings.
        run_resorting
            Run the resorting machinery even if unnecessary because `atoms_map=True`.
        uno_cutoff
            TODO
        run_mirror
            Run alternate geometries potentially allowing best match to `ref_mol`
            from mirror image of `concern_mol`. Only run if system confirmed to
            be nonsuperimposable upon mirror reflection.
        generic_ghosts
            When one or both molecules doesn't have meaningful element info for ghosts (can happen
            when harvesting from a printout with a generic ghost symbol), set this to True to
            place all real=False atoms into the same space for alignment. Only allowed when
            ``atoms_map=True``.
        verbose
            Print level.

        Returns
        -------
        mol : Molecule
        data : Dict[key, Any]
            Molecule is internal geometry of `self` optimally aligned and atom-ordered
            to `ref_mol`. Presently all fragment information is discarded.
            `data['rmsd']` is RMSD [A] between `ref_mol` and the optimally aligned
            geometry computed.
            `data['mill']` is a AlignmentMill with fields
            (shift, rotation, atommap, mirror) that prescribe the transformation
            from `concern_mol` and the optimally aligned geometry.

        """
        from ...molutil.align import B787

        rgeom = np.array(ref_mol.geometry)
        runiq = np.asarray(
            [
                hashlib.sha1((sym + str(mas)).encode("utf-8")).hexdigest()
                for sym, mas in zip(cast(Iterable[str], ref_mol.symbols), ref_mol.masses)
            ]
        )
        concern_mol = self
        cgeom = np.array(concern_mol.geometry)
        cuniq = np.asarray(
            [
                hashlib.sha1((sym + str(mas)).encode("utf-8")).hexdigest()
                for sym, mas in zip(cast(Iterable[str], concern_mol.symbols), concern_mol.masses)
            ]
        )

        if generic_ghosts:
            if not mols_align:
                raise ValueError("Too risky to lump ghosts together when mols not superimposable.")

            bq_hash = hashlib.sha1(("bq").encode("utf-8")).hexdigest()
            runiq = np.asarray([(rl_hash if rl else bq_hash) for rl, rl_hash in zip(ref_mol.real, runiq)])
            cuniq = np.asarray([(rl_hash if rl else bq_hash) for rl, rl_hash in zip(concern_mol.real, cuniq)])

        rmsd, solution = B787(
            cgeom=cgeom,
            rgeom=rgeom,
            cuniq=cuniq,
            runiq=runiq,
            do_plot=do_plot,
            verbose=verbose,
            atoms_map=atoms_map,
            run_resorting=run_resorting,
            mols_align=mols_align,
            run_to_completion=run_to_completion,
            run_mirror=run_mirror,
            uno_cutoff=uno_cutoff,
        )

        aupdate = {
            "symbols": solution.align_atoms(concern_mol.symbols),
            "geometry": solution.align_coordinates(concern_mol.geometry, reverse=False),
            "masses": solution.align_atoms(concern_mol.masses),
            "real": solution.align_atoms(concern_mol.real),
            "atom_labels": solution.align_atoms(concern_mol.atom_labels),
            "atomic_numbers": solution.align_atoms(concern_mol.atomic_numbers),
            "mass_numbers": solution.align_atoms(concern_mol.mass_numbers),
        }
        adict = {**concern_mol.model_dump(), **aupdate}

        # preserve intrinsic symmetry with lighter truncation
        amol = Molecule(validate=True, **adict, geometry_noise=13)

        # TODO -- can probably do more with fragments in amol now that
        #         Mol is something with non-contig frags. frags now discarded.

        assert compare_values(
            concern_mol.nuclear_repulsion_energy(),
            amol.nuclear_repulsion_energy(),
            "Q: concern_mol-->returned_mol NRE uncorrupted",
            atol=1.0e-4,
            quiet=(verbose > 1),
        )
        if mols_align:
            assert compare_values(
                ref_mol.nuclear_repulsion_energy(),
                amol.nuclear_repulsion_energy(),
                "Q: concern_mol-->returned_mol NRE matches ref_mol",
                atol=1.0e-4,
                quiet=(verbose > 1),
            )
            assert compare(
                True,
                np.allclose(ref_mol.geometry, amol.geometry, atol=4),
                "Q: concern_mol-->returned_mol geometry matches ref_mol",
                quiet=(verbose > 1),
            )

        return amol, {"rmsd": rmsd, "mill": solution}

    def scramble(
        self,
        *,
        do_shift: Union[bool, Array[float], List] = True,
        do_rotate: Union[bool, Array[float], List[List]] = True,
        do_resort: Union[bool, List] = True,
        deflection: float = 1.0,
        do_mirror: bool = False,
        do_plot: bool = False,
        do_test: bool = False,
        run_to_completion: bool = False,
        run_resorting: bool = False,
        verbose: int = 0,
    ) -> Tuple["Molecule", Dict[str, Any]]:
        r"""Generate a Molecule with random or directed translation, rotation, and atom shuffling.
        Optionally, check that the aligner returns the opposite transformation.

        Parameters
        ----------
        ref_mol : qcelemental.models.Molecule
            Molecule to perturb.
        do_shift
            Whether to generate a random atom shift on interval [-3, 3) in each
            dimension (`True`) or leave at current origin. To shift by a specified
            vector, supply a 3-element list.
        do_rotate
            Whether to generate a random 3D rotation according to algorithm of Arvo.
            To rotate by a specified matrix, supply a 9-element list of lists.
        do_resort
            Whether to shuffle atoms (`True`) or leave 1st atom 1st, etc. (`False`).
            To specify shuffle, supply a nat-element list of indices.
        deflection
            If `do_rotate`, how random a rotation: 0.0 is no change, 0.1 is small
            perturbation, 1.0 is completely random.
        do_mirror
            Whether to construct the mirror image structure by inverting y-axis.
        do_plot
            Pops up a mpl plot showing before, after, and ref geometries.
        do_test
            Additionally, run the aligner on the returned Molecule and check that
            opposite transformations obtained.
        run_to_completion
            By construction, scrambled systems are fully alignable (final RMSD=0).
            Even so, `True` turns off the mechanism to stop when RMSD reaches zero
            and instead proceed to worst possible time.
        run_resorting
            Even if atoms not shuffled, test the resorting machinery.
        verbose
            Print level.

        Returns
        -------
        mol : Molecule
        data : Dict[key, Any]
            Molecule is scrambled copy of `ref_mol` (self).
            `data['rmsd']` is RMSD [A] between `ref_mol` and the scrambled geometry.
            `data['mill']` is a AlignmentMill with fields
            (shift, rotation, atommap, mirror) that prescribe the transformation
            from `ref_mol` to the returned geometry.

        Raises
        ------
        AssertionError
            If `do_test=True` and aligner sanity check fails for any of the reverse
            transformations.

        """
        from ...molutil.align import compute_scramble

        ref_mol = self
        rgeom = ref_mol.geometry
        nat = rgeom.shape[0]

        perturbation = compute_scramble(
            nat,
            do_shift=do_shift,
            do_rotate=do_rotate,
            deflection=deflection,
            do_resort=do_resort,
            do_mirror=do_mirror,
        )

        cgeom = perturbation.align_coordinates(rgeom, reverse=True)
        cupdate = {
            "symbols": perturbation.align_atoms(ref_mol.symbols),
            "geometry": cgeom,
            "masses": perturbation.align_atoms(ref_mol.masses),
            "real": perturbation.align_atoms(ref_mol.real),
            "atom_labels": perturbation.align_atoms(ref_mol.atom_labels),
            "atomic_numbers": perturbation.align_atoms(ref_mol.atomic_numbers),
            "mass_numbers": perturbation.align_atoms(ref_mol.mass_numbers),
        }
        cdict = {**ref_mol.model_dump(), **cupdate}

        # preserve intrinsic symmetry with lighter truncation
        cmol = Molecule(validate=True, **cdict, geometry_noise=13)

        rmsd = np.linalg.norm(cgeom - rgeom) * constants.bohr2angstroms / np.sqrt(nat)
        if verbose >= 1:
            print("Start RMSD = {:8.4f} [A]".format(rmsd))

        if do_test:
            _, data = cmol.align(
                ref_mol,
                do_plot=do_plot,
                atoms_map=(not do_resort),
                run_resorting=run_resorting,
                mols_align=True,
                run_to_completion=run_to_completion,
                run_mirror=do_mirror,
                verbose=verbose,
            )
            solution = data["mill"]

            assert compare(
                True, np.allclose(solution.shift, perturbation.shift, atol=1.0e-6), "shifts equiv", quiet=(verbose > 1)
            )
            if not do_resort:
                assert compare(
                    True,
                    np.allclose(solution.rotation.T, perturbation.rotation),
                    "rotations transpose",
                    quiet=(verbose > 1),
                )
            if solution.mirror:
                assert compare(True, do_mirror, "mirror allowed", quiet=(verbose > 1))

        return cmol, {"rmsd": rmsd, "mill": perturbation}

    def convert_v(
        self, target_version: int, /
    ) -> Union["qcelemental.models.v1.Molecule", "qcelemental.models.v2.Molecule"]:
        """Convert to instance of particular QCSchema version."""
        import qcelemental as qcel

        if check_convertible_version(target_version, error="Molecule") == "self":
            return self

        dself = self.model_dump()
        if target_version == 1:
            # below is assignment rather than popping so Mol() records as set and future Mol.model_dump() includes the field.
            #   needed for QCEngine Psi4.
            dself["schema_version"] = 2

            self_vN = qcel.models.v1.Molecule(**dself)
        else:
            assert False, target_version

        return self_vN


def _filter_defaults(dicary):
    nat = len(dicary["symbols"])
    default_mass = np.array([periodictable.to_mass(e) for e in dicary["symbols"]])

    dicary.pop("atomic_numbers")

    if np.allclose(default_mass, dicary["masses"]):
        dicary.pop("mass_numbers")
        dicary.pop("masses")

    if all(dicary["real"]):
        dicary.pop("real")

    if dicary["atom_labels"].tolist() == nat * [""]:
        dicary.pop("atom_labels")

    if dicary.get("connectivity", "N/A") is None:
        dicary.pop("connectivity")

    if dicary["fragments"] == [list(np.arange(nat))]:
        dicary.pop("fragments")
        dicary.pop("fragment_charges")
        dicary.pop("fragment_multiplicities")

    return dicary


# auto_gen_docs_on_demand(Molecule)
