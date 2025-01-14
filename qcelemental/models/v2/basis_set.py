from enum import Enum
from typing import TYPE_CHECKING, Dict, List, Literal, Optional, Union

from pydantic import Field, constr, field_validator
from typing_extensions import Annotated

from ...exceptions import ValidationError
from .basemodels import ProtoModel, check_convertible_version, qcschema_draft

if TYPE_CHECKING:
    import qcelemental

NonnegativeInt = Annotated[int, Field(ge=0)]


class HarmonicType(str, Enum):
    """The angular momentum representation of a shell."""

    spherical = "spherical"
    cartesian = "cartesian"


def electron_shell_json_schema_extra(schema, model):
    # edit to allow string storage of basis sets as BSE uses.
    # alternately, could `Union[float, str]` above but that loses some validation
    schema["properties"]["exponents"]["items"] = {"anyOf": [{"type": "number"}, {"type": "string"}]}
    schema["properties"]["coefficients"]["items"]["items"] = {"anyOf": [{"type": "number"}, {"type": "string"}]}
    schema["properties"]["angular_momentum"].update({"uniqueItems": True})


class ElectronShell(ProtoModel):
    """Information for a single electronic shell."""

    angular_momentum: List[NonnegativeInt] = Field(
        ..., description="Angular momentum for the shell as an array of integers.", min_length=1
    )
    harmonic_type: HarmonicType = Field(..., description=str(HarmonicType.__doc__))
    exponents: List[float] = Field(..., description="Exponents for the contracted shell.", min_length=1)
    coefficients: List[List[float]] = Field(
        ...,
        description="General contraction coefficients for the shell; "
        "individual list components will be the individual segment contraction coefficients.",
        min_length=1,
    )

    model_config = ProtoModel._merge_config_with(json_schema_extra=electron_shell_json_schema_extra)

    @field_validator("coefficients")
    @classmethod
    def _check_coefficient_length(cls, v, info):
        len_exp = len(info.data["exponents"])
        for row in v:
            if len(row) != len_exp:
                raise ValueError("The length of coefficients does not match the length of exponents.")

        return v

    @field_validator("coefficients")
    @classmethod
    def _check_general_contraction_or_fused(cls, v, info):
        angular_momentum = info.data["angular_momentum"]
        if len(angular_momentum) > 1 and len(angular_momentum) != len(v):
            raise ValueError("The length for a fused shell must equal the length of coefficients.")
        return v

    def nfunctions(self) -> int:
        r"""
        Computes the number of basis functions on this shell.

        Returns
        -------
        int
            The number of basis functions on this shell.
        """

        if self.harmonic_type == "spherical":
            return sum((2 * L + 1) for L in self.angular_momentum)
        else:
            return sum(((L + 1) * (L + 2) // 2) for L in self.angular_momentum)

    def is_contracted(self) -> bool:
        r"""
        Checks if the shell represents a contracted Gaussian or not.

        Returns
        -------
        bool
            True if the shell is contracted.
        """

        return (len(self.coefficients) != 1) and (len(self.angular_momentum) == 1)


class ECPType(str, Enum):
    """The type of the ECP potential."""

    scalar = "scalar"
    spinorbit = "spinorbit"


def ecp_json_schema_extra(schema, model):
    # edit to allow string storage of basis sets as BSE uses.
    # alternately, could `Union[float, str]` above but that loses some validation
    schema["properties"]["gaussian_exponents"]["items"] = {"anyOf": [{"type": "number"}, {"type": "string"}]}
    schema["properties"]["coefficients"]["items"]["items"] = {"anyOf": [{"type": "number"}, {"type": "string"}]}
    schema["properties"]["angular_momentum"].update({"uniqueItems": True})


class ECPPotential(ProtoModel):
    """Information for a single ECP potential."""

    ecp_type: ECPType = Field(..., description=str(ECPType.__doc__))
    angular_momentum: List[NonnegativeInt] = Field(
        ..., description="Angular momentum for the potential as an array of integers.", min_length=1
    )
    r_exponents: List[int] = Field(..., description="Exponents of the 'r' term.", min_length=1)
    gaussian_exponents: List[float] = Field(..., description="Exponents of the 'gaussian' term.", min_length=1)
    coefficients: List[List[float]] = Field(
        ...,
        description="General contraction coefficients for the potential; "
        "individual list components will be the individual segment contraction coefficients.",
        min_length=1,
    )

    model_config = ProtoModel._merge_config_with(json_schema_extra=ecp_json_schema_extra)

    @field_validator("gaussian_exponents")
    @classmethod
    def _check_gaussian_exponents_length(cls, v, info):
        len_exp = len(info.data["r_exponents"])
        if len(v) != len_exp:
            raise ValueError("The length of gaussian_exponents does not match the length of `r` exponents.")

        return v

    @field_validator("coefficients")
    @classmethod
    def _check_coefficient_length(cls, v, info):
        len_exp = len(info.data["r_exponents"])
        for row in v:
            if len(row) != len_exp:
                raise ValueError("The length of coefficients does not match the length of `r` exponents.")

        return v


def basis_center_json_schema_extras(schema, model):
    schema["properties"]["electron_shells"].update({"uniqueItems": True})
    schema["properties"]["ecp_potentials"].update({"uniqueItems": True})


class BasisCenter(ProtoModel):
    """Data for a single atom/center in a basis set."""

    electron_shells: List[ElectronShell] = Field(..., description="Electronic shells for this center.", min_length=1)
    ecp_electrons: int = Field(0, description="Number of electrons replaced by ECP, MCP, or other field potentials.")
    ecp_potentials: Optional[List[ECPPotential]] = Field(
        None, description="ECPs, MCPs, or other field potentials for this center.", min_length=1
    )

    model_config = ProtoModel._merge_config_with(json_schema_extra=basis_center_json_schema_extras)


def basis_set_json_schema_extra(schema, model):
    schema["$schema"] = qcschema_draft


class BasisSet(ProtoModel):
    """
    A quantum chemistry basis description.
    """

    schema_name: Literal["qcschema_basis_set"] = Field(
        "qcschema_basis_set", description=(f"The QCSchema specification to which this model conforms.")
    )
    schema_version: Literal[2] = Field(  # type: ignore
        2,
        description="The version number of :attr:`~qcelemental.models.BasisSet.schema_name` "
        "to which this model conforms.",
    )

    name: str = Field(..., description="The standard basis name if available (e.g., 'cc-pVDZ').")
    description: Optional[str] = Field(None, description="Brief description of the basis set.")
    center_data: Dict[str, BasisCenter] = Field(
        ..., description="Shared basis data for all atoms/centers in the parent molecule"
    )
    atom_map: List[str] = Field(
        ..., description="Mapping of all atoms/centers in the parent molecule to centers in ``center_data``."
    )

    nbf: Optional[int] = Field(
        None, description="The number of basis functions. Use for convenience or as checksum", validate_default=True
    )

    model_config = ProtoModel._merge_config_with(json_schema_extra=basis_set_json_schema_extra)

    @field_validator("atom_map")
    @classmethod
    def _check_atom_map(cls, v, info):
        sv = set(v)

        # Center_data validation error, skipping
        try:
            missing = sv - info.data["center_data"].keys()
        except KeyError:
            return v

        if missing:
            raise ValueError(f"'atom_map' contains unknown keys to 'center_data': {missing}.")

        return v

    @field_validator("nbf")
    @classmethod
    def _check_nbf(cls, v, info):
        # Bad construction, pass on errors
        try:
            nbf = cls._calculate_nbf(info.data["atom_map"], info.data["center_data"])
        except KeyError:
            return v

        if v is None:
            v = nbf
        else:
            if v != nbf:
                raise ValidationError("Calculated nbf does not match supplied nbf.")

        return v

    @classmethod
    def _calculate_nbf(cls, atom_map, center_data) -> int:
        r"""
        Number of basis functions in the basis set.

        Returns
        -------
        int
            The number of basis functions.
        """

        center_count = {}
        for k, center in center_data.items():
            center_count[k] = sum(x.nfunctions() for x in center.electron_shells)

        ret = 0
        for center in atom_map:
            ret += center_count[center]

        return ret

    def convert_v(
        self, target_version: int, /
    ) -> Union["qcelemental.models.v1.BasisSet", "qcelemental.models.v2.BasisSet"]:
        """Convert to instance of particular QCSchema version."""
        import qcelemental as qcel

        if check_convertible_version(target_version, error="BasisSet") == "self":
            return self

        dself = self.model_dump()
        if target_version == 1:
            dself.pop("schema_name")  # changes in v1
            dself.pop("schema_version")  # changes in v1

            self_vN = qcel.models.v1.BasisSet(**dself)
        else:
            assert False, target_version

        return self_vN
