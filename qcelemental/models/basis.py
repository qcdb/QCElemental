import math
from enum import Enum
from functools import lru_cache
from typing import Dict, List, Optional

try:
    from pydantic.v1 import ConstrainedInt, Field, constr, validator
except ImportError:  # Will also trap ModuleNotFoundError
    from pydantic import ConstrainedInt, Field, constr, validator

from ..exceptions import ValidationError
from .basemodels import ProtoModel, qcschema_draft


class NonnegativeInt(ConstrainedInt):
    ge = 0

M_SQRTPI_CUBED = math.pi * math.sqrt(math.pi)


class HarmonicType(str, Enum):
    """The angular momentum representation of a shell."""

    spherical = "spherical"
    cartesian = "cartesian"


class NormalizationScheme(str, Enum):

    cca = "cca"
    erd = "erd"
    bse = "bse"


class ElectronShell(ProtoModel):
    """Information for a single electronic shell."""

    angular_momentum: List[NonnegativeInt] = Field(
        ..., description="Angular momentum for the shell as an array of integers.", min_items=1
    )
    harmonic_type: HarmonicType = Field(..., description=str(HarmonicType.__doc__))
    exponents: List[float] = Field(..., description="Exponents for the contracted shell.", min_items=1)
    coefficients: List[List[float]] = Field(
        ...,
        description="General contraction coefficients for the shell; individual list components will be the individual segment contraction coefficients.",
        min_items=1,
    )
    normalization_scheme: Optional[NormalizationScheme] = Field(
        None, description="Normalization scheme for this shell."
    )

    class Config(ProtoModel.Config):
        def schema_extra(schema, model):
            # edit to allow string storage of basis sets as BSE uses. alternately, could `Union[float, str]` above but that loses some validation
            schema["properties"]["exponents"]["items"] = {"anyOf": [{"type": "number"}, {"type": "string"}]}
            schema["properties"]["coefficients"]["items"]["items"] = {"anyOf": [{"type": "number"}, {"type": "string"}]}
            schema["properties"]["angular_momentum"].update({"uniqueItems": True})

    @validator("coefficients")
    def _check_coefficient_length(cls, v, values):
        len_exp = len(values["exponents"])
        for row in v:
            if len(row) != len_exp:
                raise ValueError("The length of coefficients does not match the length of exponents.")

        return v

    @validator("coefficients")
    def _check_general_contraction_or_fused(cls, v, values):
        if len(values["angular_momentum"]) > 1:
            if len(values["angular_momentum"]) != len(v):
                raise ValueError("The length for a fused shell must equal the length of coefficients.")

        return v

    @validator("normalization_scheme", always=True)
    def _check_normsch(cls, v, values):

        # Bad construction, pass on errors
        try:
            normsch = cls._calculate_normsch(values["angular_momentum"], values["exponents"], values["coefficients"])
        except KeyError:
            return v

        if v is None:
            v = normsch
        else:
            if v != normsch:
                raise ValidationError(f"Calculated normalization_scheme ({normsch}) does not match supplied ({v}).")

        return v

    @classmethod
    def _calculate_normsch(cls, am: int, exps: List[float], coefs: List[List[float]]) -> NormalizationScheme:
        def single_am(idx, l):
            m = l + 1.5

            # try CCA
            candidate_already_normalized_coefs = coefs[idx]
            norm = cls._cca_contraction_normalization(l, exps, candidate_already_normalized_coefs)
            if abs(norm - 1) < 1.0e-10:
                return NormalizationScheme.cca

            # try ERD
            candidate_already_normalized_coefs = [coefs[idx][i] / pow(exps[i], 0.5 * m) for i in range(len(exps))]
            norm = cls._erd_normalization(l, exps, candidate_already_normalized_coefs)
            if abs(norm - 1) < 1.0e-10:
                return NormalizationScheme.erd

            # BSE not confirmable
            return NormalizationScheme.bse

        return _collapse_equal_list(single_am(idx, l) for idx, l in enumerate(am))

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

    def normalize_shell(self, dtype: NormalizationScheme) -> "ElectronShell":
        """Construct new ElectronShell with coefficients normalized by ``dtype``."""

        naive_coefs = self._denormalize_to_bse()

        bse_shell = self.dict()
        bse_shell["coefficients"] = naive_coefs
        bse_shell["normalization_scheme"] = "bse"
        bse_shell = ElectronShell(**bse_shell)

        if dtype == "bse":
            return bse_shell
        elif dtype in ["cca", "psi4"]:
            norm_coef = bse_shell._cca_normalize_shell()
        elif dtype == "erd":
            norm_coef = bse_shell._erd_normalize_shell()

        new_shell = self.dict()
        new_shell["coefficients"] = norm_coef
        new_shell["normalization_scheme"] = dtype
        return ElectronShell(**new_shell)

    def _denormalize_to_bse(self) -> List[List[float]]:
        """Compute replacement coefficients for any-normalization shell ``self`` that are within a scale factor of BSE unnormalization."""

        def single_am(idx, l):

            if self.normalization_scheme == "cca":
                prim_norm = self._cca_primitive_normalization(l, self.exponents)
                return [self.coefficients[idx][i] / prim_norm[i] for i in range(len(self.exponents))]

            elif self.normalization_scheme == "erd":
                m = l + 1.5
                return [self.coefficients[idx][i] / pow(self.exponents[i], 0.5 * m) for i in range(len(self.exponents))]

            elif self.normalization_scheme == "bse":
                return self.coefficients[idx]

        return [single_am(idx, l) for idx, l in enumerate(self.angular_momentum)]

    @staticmethod
    def _cca_primitive_normalization(l: int, exps: List[float]) -> List[float]:
        """Compute CCA normalization factor for primitive shell using angular momentum ``l`` and exponents ``exps``."""
        m = l + 1.5
        prim_norm = [
            math.sqrt((pow(2.0, l) * pow(2.0 * exps[p], m)) / (M_SQRTPI_CUBED * _df(2 * l))) for p in range(len(exps))
        ]

        return prim_norm

    @staticmethod
    def _cca_contraction_normalization(l: int, exps: List[float], coefs: List[List[float]]) -> float:
        """Compute CCA normalization factor for coefficients ``coefs`` using angular momentum ``l`` and exponents ``exps``."""

        m = l + 1.5
        summ = 0.0
        for i in range(len(exps)):
            for j in range(len(exps)):
                z = pow(exps[i] + exps[j], m)
                summ += (coefs[i] * coefs[j]) / z

        tmp = (M_SQRTPI_CUBED * _df(2 * l)) / pow(2.0, l)
        norm = math.sqrt(1.0 / (tmp * summ))
        # except (ZeroDivisionError, ValueError): [idx][i] = 1.0

        return norm

    def _cca_normalize_shell(self) -> List[List[float]]:
        """Compute replacement coefficients for unnormalized (BSE-normalized) shell ``self`` that fulfill CCA normalization."""

        if self.normalization_scheme != "bse":
            raise TypeError('Unnormalized shells expected. Use ``normalize_shell(dtype="cca")`` for flexibility.')

        def single_am(idx, l):
            prim_norm = ElectronShell._cca_primitive_normalization(l, self.exponents)
            norm = ElectronShell._cca_contraction_normalization(
                l, self.exponents, [self.coefficients[idx][i] * prim_norm[i] for i in range(len(self.exponents))]
            )
            return [self.coefficients[idx][i] * norm * prim_norm[i] for i in range(len(self.exponents))]

        return [single_am(idx, l) for idx, l in enumerate(self.angular_momentum)]

    def _erd_normalization(l: int, exps: List[float], coefs: List[List[float]]) -> float:
        """Compute ERD normalization factor for coefficients ``coefs`` using angular momentum ``l`` and exponents ``exps``."""

        m = l + 1.5
        summ = 0.0
        for i in range(len(exps)):
            for j in range(i + 1):
                temp = coefs[i] * coefs[j]
                temp3 = 2.0 * math.sqrt(exps[i] * exps[j]) / (exps[i] + exps[j])
                temp *= pow(temp3, m)

                summ += temp
                if i != j:
                    summ += temp

        prefac = 1.0
        if l > 1:
            prefac = pow(2.0, 2 * l) / _df(2 * l)
        norm = math.sqrt(prefac / summ)

        return norm

    def _erd_normalize_shell(self) -> List[List[float]]:
        """Compute replacement coefficients for unnormalized (BSE-normalized) shell ``self`` that fulfill ERD normalization."""

        if self.normalization_scheme != "bse":
            raise TypeError('Unnormalized shells expected. Use ``normalize_shell(dtype="erd")`` for flexibility.')

        def single_am(idx, l):
            m = l + 1.5

            norm = ElectronShell._erd_normalization(l, self.exponents, self.coefficients[idx])
            return [
                self.coefficients[idx][i] * norm * pow(self.exponents[i], 0.5 * m) for i in range(len(self.exponents))
            ]

        return [single_am(idx, l) for idx, l in enumerate(self.angular_momentum)]


class ECPType(str, Enum):
    """The type of the ECP potential."""

    scalar = "scalar"
    spinorbit = "spinorbit"


class ECPPotential(ProtoModel):
    """Information for a single ECP potential."""

    ecp_type: ECPType = Field(..., description=str(ECPType.__doc__))
    angular_momentum: List[NonnegativeInt] = Field(
        ..., description="Angular momentum for the potential as an array of integers.", min_items=1
    )
    r_exponents: List[int] = Field(..., description="Exponents of the 'r' term.", min_items=1)
    gaussian_exponents: List[float] = Field(..., description="Exponents of the 'gaussian' term.", min_items=1)
    coefficients: List[List[float]] = Field(
        ...,
        description="General contraction coefficients for the potential; individual list components will be the individual segment contraction coefficients.",
        min_items=1,
    )

    class Config(ProtoModel.Config):
        def schema_extra(schema, model):
            # edit to allow string storage of basis sets as BSE uses. alternately, could `Union[float, str]` above but that loses some validation
            schema["properties"]["gaussian_exponents"]["items"] = {"anyOf": [{"type": "number"}, {"type": "string"}]}
            schema["properties"]["coefficients"]["items"]["items"] = {"anyOf": [{"type": "number"}, {"type": "string"}]}
            schema["properties"]["angular_momentum"].update({"uniqueItems": True})

    @validator("gaussian_exponents")
    def _check_gaussian_exponents_length(cls, v, values):
        len_exp = len(values["r_exponents"])
        if len(v) != len_exp:
            raise ValueError("The length of gaussian_exponents does not match the length of `r` exponents.")

        return v

    @validator("coefficients")
    def _check_coefficient_length(cls, v, values):
        len_exp = len(values["r_exponents"])
        for row in v:
            if len(row) != len_exp:
                raise ValueError("The length of coefficients does not match the length of `r` exponents.")

        return v


class BasisCenter(ProtoModel):
    """Data for a single atom/center in a basis set."""

    electron_shells: List[ElectronShell] = Field(..., description="Electronic shells for this center.", min_items=1)
    ecp_electrons: int = Field(0, description="Number of electrons replaced by ECP, MCP, or other field potentials.")
    ecp_potentials: Optional[List[ECPPotential]] = Field(
        None, description="ECPs, MCPs, or other field potentials for this center.", min_items=1
    )

    class Config(ProtoModel.Config):
        def schema_extra(schema, model):
            schema["properties"]["electron_shells"].update({"uniqueItems": True})
            schema["properties"]["ecp_potentials"].update({"uniqueItems": True})


class BasisSet(ProtoModel):
    """
    A quantum chemistry basis description.
    """

    schema_name: constr(strip_whitespace=True, regex="^(qcschema_basis)$") = Field(  # type: ignore
        "qcschema_basis",
        description=(f"The QCSchema specification to which this model conforms. Explicitly fixed as qcschema_basis."),
    )
    schema_version: int = Field(  # type: ignore
        1,
        description="The version number of :attr:`~qcelemental.models.BasisSet.schema_name` to which this model conforms.",
    )

    name: str = Field(..., description="The standard basis name if available (e.g., 'cc-pVDZ').")
    description: Optional[str] = Field(None, description="Brief description of the basis set.")
    center_data: Dict[str, BasisCenter] = Field(
        ..., description="Shared basis data for all atoms/centers in the parent molecule"
    )
    atom_map: List[str] = Field(
        ..., description="Mapping of all atoms/centers in the parent molecule to centers in ``center_data``."
    )

    nbf: Optional[int] = Field(None, description="The number of basis functions. Use for convenience or as checksum")

    class Config(ProtoModel.Config):
        def schema_extra(schema, model):
            schema["$schema"] = qcschema_draft

    @validator("atom_map")
    def _check_atom_map(cls, v, values):
        sv = set(v)

        # Center_data validation error, skipping
        try:
            missing = sv - values["center_data"].keys()
        except KeyError:
            return v

        if missing:
            raise ValueError(f"'atom_map' contains unknown keys to 'center_data': {missing}.")

        return v

    @validator("nbf", always=True)
    def _check_nbf(cls, v, values):
        # Bad construction, pass on errors
        try:
            nbf = cls._calculate_nbf(values["atom_map"], values["center_data"])
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

    def normalize_shell(self, dtype: NormalizationScheme) -> "BasisSet":
        """Construct new BasisSet with coefficients of all shells in center_data normalized by ``dtype``."""

        new_bs = self.dict()

        for lbl, center in self.center_data.items():
            for ish, sh in enumerate(center.electron_shells):
                new_bs["center_data"][lbl]["electron_shells"][ish] = sh.normalize_shell(dtype)

        return BasisSet(**new_bs)

    def normalization_scheme(self) -> NormalizationScheme:
        """Identify probable normalization scheme governing shell ``coefficients`` in center_data.

        Returns
        -------
        NormalizationScheme
            Satisfied by all ElectronShells.

        Raises
        ------
        TypeError
            If the BasisSet's ElectronShells are detected to have inconsistent normalization schemes.

        """
        shell_norm = []
        for lbl, center in self.center_data.items():
            for ish, sh in enumerate(center.electron_shells):
                shell_norm.append(sh.normalization_scheme)

        return _collapse_equal_list(shell_norm)


@lru_cache(maxsize=500)
def _df(i):
    if i in [0, 1, 2]:
        return 1.0
    else:
        return (i - 1) * _df(i - 2)


def _collapse_equal_list(lst):
    lst = list(lst)
    first = lst[0]
    if lst.count(first) == len(lst):
        return first
    else:
        raise TypeError(f"Inconsistent members in list: {lst}")
