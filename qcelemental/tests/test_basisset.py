# import numpy as np
import pytest

# import qcelemental as qcel
# from qcelemental.exceptions import NotAnElementError
from qcelemental.models import BasisSet
from qcelemental.testing import compare, compare_recursive, compare_values

# These are formed from Psi4's internal basis library, then running DGAS's schema_wrapper._convert_basis on it with the
#    three original_coef/coef/erd_coef forms exported.
_basissets = {
    "bse": BasisSet(
        **{
            "atom_map": ["CC-PVTZ_He1"],
            "center_data": {
                "CC-PVTZ_He1": {
                    "ecp_electrons": 0,
                    "ecp_potentials": None,
                    "electron_shells": [
                        {
                            "angular_momentum": [0],
                            "coefficients": [[0.002587, 0.019533, 0.090998, 0.27205]],
                            "exponents": [234.0, 35.16, 7.989, 2.212],
                            "harmonic_type": "spherical",
                        },
                        {
                            "angular_momentum": [0],
                            "coefficients": [[1.0]],
                            "exponents": [0.6669],
                            "harmonic_type": "spherical",
                        },
                        {
                            "angular_momentum": [0],
                            "coefficients": [[1.0]],
                            "exponents": [0.2089],
                            "harmonic_type": "spherical",
                        },
                        {
                            "angular_momentum": [1],
                            "coefficients": [[1.0]],
                            "exponents": [3.044],
                            "harmonic_type": "spherical",
                        },
                        {
                            "angular_momentum": [1],
                            "coefficients": [[1.0]],
                            "exponents": [0.758],
                            "harmonic_type": "spherical",
                        },
                        {
                            "angular_momentum": [2],
                            "coefficients": [[1.0]],
                            "exponents": [1.965],
                            "harmonic_type": "spherical",
                        },
                    ],
                }
            },
            "description": None,
            "name": "CC-PVTZ",
            "nbf": 14,
            "schema_name": "qcschema_basis",
            "schema_version": 1,
        }
    ),
    "cca": BasisSet(
        **{
            "atom_map": ["CC-PVTZ_He1"],
            "center_data": {
                "CC-PVTZ_He1": {
                    "ecp_electrons": 0,
                    "ecp_potentials": None,
                    "electron_shells": [
                        {
                            "angular_momentum": [0],
                            "coefficients": [
                                [0.3109111498784228, 0.5665439094158013, 0.8686186270728533, 0.9912097091412122]
                            ],
                            "exponents": [234.0, 35.16, 7.989, 2.212],
                            "harmonic_type": "spherical",
                        },
                        {
                            "angular_momentum": [0],
                            "coefficients": [[0.5259635285661938]],
                            "exponents": [0.6669],
                            "harmonic_type": "spherical",
                        },
                        {
                            "angular_momentum": [0],
                            "coefficients": [[0.22022363267224998]],
                            "exponents": [0.2089],
                            "harmonic_type": "spherical",
                        },
                        {
                            "angular_momentum": [1],
                            "coefficients": [[5.731204405620397]],
                            "exponents": [3.044],
                            "harmonic_type": "spherical",
                        },
                        {
                            "angular_momentum": [1],
                            "coefficients": [[1.0081533438537973]],
                            "exponents": [0.758],
                            "harmonic_type": "spherical",
                        },
                        {
                            "angular_momentum": [2],
                            "coefficients": [[5.367770348245947]],
                            "exponents": [1.965],
                            "harmonic_type": "spherical",
                        },
                    ],
                }
            },
            "description": None,
            "name": "CC-PVTZ",
            "nbf": 14,
            "schema_name": "qcschema_basis",
            "schema_version": 1,
        }
    ),
    "erd": BasisSet(
        **{
            "atom_map": ["CC-PVTZ_He1"],
            "center_data": {
                "CC-PVTZ_He1": {
                    "ecp_electrons": 0,
                    "ecp_potentials": None,
                    "electron_shells": [
                        {
                            "angular_momentum": [0],
                            "coefficients": [
                                [0.4362407232872251, 0.7949201079284722, 1.2187623965341599, 1.3907704519897992]
                            ],
                            "exponents": [234.0, 35.16, 7.989, 2.212],
                            "harmonic_type": "spherical",
                        },
                        {
                            "angular_momentum": [0],
                            "coefficients": [[0.7379816073310306]],
                            "exponents": [0.6669],
                            "harmonic_type": "spherical",
                        },
                        {
                            "angular_momentum": [0],
                            "coefficients": [[0.3089966919470384]],
                            "exponents": [0.2089],
                            "harmonic_type": "spherical",
                        },
                        {
                            "angular_momentum": [1],
                            "coefficients": [[4.0207383302149715]],
                            "exponents": [3.044],
                            "harmonic_type": "spherical",
                        },
                        {
                            "angular_momentum": [1],
                            "coefficients": [[0.7072720680477226]],
                            "exponents": [0.758],
                            "harmonic_type": "spherical",
                        },
                        {
                            "angular_momentum": [2],
                            "coefficients": [[7.531540827899531]],
                            "exponents": [1.965],
                            "harmonic_type": "spherical",
                        },
                    ],
                }
            },
            "description": None,
            "name": "CC-PVTZ",
            "nbf": 14,
            "schema_name": "qcschema_basis",
            "schema_version": 1,
        }
    ),
}


@pytest.mark.parametrize("normsch", ["cca", "erd", "bse"])
def test_norm_status(normsch):
    assert _basissets[normsch].normalization_scheme() == normsch


@pytest.mark.parametrize(
    "fromm,to",
    [("bse", "erd"), ("cca", "erd"), ("erd", "erd"), ("bse", "cca"), ("cca", "cca"), ("erd", "cca"), ("bse", "bse"),],
)
def test_norm_conversion(fromm, to):
    newbs = _basissets[fromm].normalize_shell(dtype=to)
    assert compare_recursive(_basissets[to].dict(), newbs.dict(), f"{fromm} --> {to}")
    newnewbs = newbs.normalize_shell(dtype=to)
    assert compare_recursive(_basissets[to].dict(), newnewbs.dict(), f"{fromm} --> {to} idempotent")
