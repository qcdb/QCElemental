import copy
import json
import warnings

import numpy as np
import pydantic
import pytest

import qcelemental as qcel

from .addons import drop_qcsk, schema_versions, using_qcmb

center_data = {
    "bs_sto3g_h": {
        "electron_shells": [
            {
                "harmonic_type": "spherical",
                "angular_momentum": [0],
                "exponents": [3.42525091, 0.62391373, 0.16885540],
                "coefficients": [[0.15432897, 0.53532814, 0.44463454]],
            }
        ]
    },
    "bs_sto3g_o": {
        "electron_shells": [
            {
                "harmonic_type": "spherical",
                "angular_momentum": [0],
                "exponents": [130.70939, "23.808861", 6.4436089],
                "coefficients": [[0.15432899, "0.53532814", 0.44463454]],
            },
            {
                "harmonic_type": "cartesian",
                "angular_momentum": [0, 1],
                "exponents": [5.0331513, 1.1695961, 0.3803890],
                "coefficients": [[-0.09996723, 0.39951283, 0.70011547], [0.15591629, 0.60768379, 0.39195739]],
            },
            {
                "harmonic_type": "cartesian",
                "angular_momentum": [0],
                "exponents": [5.0331513, 1.1695961, 0.3803890],
                "coefficients": [[-5.09996723, 0.39951283, 0.70011547], [0.15591629, 0.60768379, 0.39195739]],
            },
        ]
    },
    "bs_def2tzvp_zr": {
        "electron_shells": [
            {
                "harmonic_type": "spherical",
                "angular_momentum": [0],
                "exponents": [11.000000000, 9.5000000000, 3.6383667759, 0.76822026698],
                "coefficients": [
                    [-0.19075595259, 0.33895588754, 0.0000000, 0.0000000],
                    [0.0000000, 0.0000000, 1.0000000000, 0.0000000],
                ],
            },
            {
                "harmonic_type": "spherical",
                "angular_momentum": [2],
                "exponents": [4.5567957795, 1.2904939799, 0.51646987229],
                "coefficients": [
                    [-0.96190569023e-09, 0.20569990155, 0.41831381851],
                    [0.0000000, 0.0000000, 0.0000000],
                    [0.0000000, 0.0000000, 0.0000000],
                ],
            },
            {
                "harmonic_type": "spherical",
                "angular_momentum": [3],
                "exponents": [0.3926100],
                "coefficients": [[1.0000000]],
            },
        ],
        "ecp_electrons": 28,
        "ecp_potentials": [
            {
                "ecp_type": "scalar",
                "angular_momentum": [0],
                "r_exponents": [2, 2, 2, 2],
                "gaussian_exponents": [7.4880494, 3.7440249, 6.5842120, 3.2921060],
                "coefficients": [[135.15384419, 15.55244130, 19.12219811, 2.43637549]],
            },
            {
                "ecp_type": "spinorbit",
                "angular_momentum": [1],
                "r_exponents": [2, 2, 2, 2],
                "gaussian_exponents": [6.4453779, 3.2226886, 6.5842120, 3.2921060],
                "coefficients": [[87.78499169, 11.56406599, 19.12219811, 2.43637549]],
            },
        ],
    },
}


@pytest.fixture(scope="function")
def result_data_fixture(schema_versions):
    Molecule = schema_versions.Molecule

    mol = Molecule.from_data(
        """
        O 0 0 0
        H 0 0 2
        H 0 2 0
    """
    )

    return {
        "molecule": mol,
        "driver": "energy",
        "model": {"method": "UFF"},
        "return_result": 5,
        "success": True,
        "properties": {},
        "provenance": {"creator": "qcel"},
        "stdout": "I ran.",
    }


@pytest.fixture(scope="function")
def wavefunction_data_fixture(result_data_fixture, schema_versions):
    BasisSet = schema_versions.basis.BasisSet

    bas = BasisSet(name="custom_basis", center_data=center_data, atom_map=["bs_sto3g_o", "bs_sto3g_h", "bs_sto3g_h"])
    c_matrix = np.random.rand(bas.nbf, bas.nbf)
    result_data_fixture["protocols"] = {"wavefunction": "all"}
    result_data_fixture["wavefunction"] = {
        "basis": bas,
        "restricted": True,
        "scf_orbitals_a": c_matrix,
        "orbitals_a": "scf_orbitals_a",
    }

    return result_data_fixture


@pytest.fixture(scope="function")
def native_data_fixture(result_data_fixture):
    result_data_fixture["protocols"] = {"native_files": "all"}
    result_data_fixture["native_files"] = {
        "input": """
echo
geometry units bohr
H1                    0.000000000000     0.000000000000    -0.650000000000
H1                    0.000000000000     0.000000000000     0.650000000000
end
charge 0.0
memory 1669668536 double
task scf energy
""",
        "DIPOL": """
        0.0000000000        0.0000000000        0.0000000000
        0.0000000000        0.0000000000       -0.7855589278
""",
        "gms.dat": """
 $DATA
Methylene...3-B-1 state...ROHF/STO-2G
CNV      2

HYDROGEN    1.0      0.8288400000      0.0000000000      0.6060939022
   STO     2

CARBON      6.0      0.0000000000      0.0000000000     -0.1018060978
   STO     2

 $END
""",
    }
    return result_data_fixture


@pytest.fixture(scope="function")
def optimization_data_fixture(result_data_fixture):
    trajectory = []
    energies = []
    for x in range(5):
        result = result_data_fixture.copy()
        result["return_result"] = x
        trajectory.append(result)
        energies.append(x)

    ret = {
        "initial_molecule": result_data_fixture["molecule"],
        "final_molecule": result_data_fixture["molecule"],
        "trajectory": trajectory,
        "energies": energies,
        "success": True,
        "provenance": {"creator": "qcel"},
        "input_specification": {"model": {"method": "UFF"}},
    }

    return ret


@pytest.fixture(scope="function")
def ethane_data_fixture():
    # from QCEngine stock_mols
    return {
        "geometry": [
            [+1.54034068369141, -1.01730823913235, +0.93128102073425],
            [+4.07197633001232, -0.09756825926424, -0.02203578938791],
            [+0.00025636057017, +0.00139534039687, +0.00111211603233],
            [+1.30983130616505, -3.03614919350581, +0.54918567185649],
            [+1.38003941036405, -0.71812565437083, +2.97078783593882],
            [+5.61209917480096, -1.11612498901607, +0.90799157528946],
            [+4.30241880148479, +1.92102238874847, +0.36057345099335],
            [+4.23222331256867, -0.39619160402976, -2.06158817835790],
        ],
        "symbols": ["C", "C", "H", "H", "H", "H", "H", "H"],
        "connectivity": [[0, 1, 1], [0, 2, 1], [0, 3, 1], [0, 4, 1], [1, 5, 1], [1, 6, 1], [1, 7, 1]],
    }


@pytest.fixture(scope="function")
def torsiondrive_data_fixture(ethane_data_fixture, optimization_data_fixture):
    ethane = ethane_data_fixture.copy()
    optres = optimization_data_fixture.copy()

    input_data = {
        "keywords": {"dihedrals": [(2, 0, 1, 5)], "grid_spacing": [180]},
        "input_specification": {"driver": "gradient", "model": {"method": "UFF", "basis": None}},
        "initial_molecule": [ethane] * 2,
        "optimization_spec": {
            "procedure": "geomeTRIC",
            "keywords": {
                "coordsys": "hdlc",
                "maxiter": 500,
                "program": "rdkit",
            },
        },
    }

    ret = {
        "success": True,
        "provenance": {"creator": "qcel"},
        "final_energies": {"180": -2.3, "0": -4.5},
        "final_molecules": {"180": ethane, "0": ethane},
        "optimization_history": {"180": [optres, optres], "0": [optres]},
        **input_data,
    }

    return ret


@pytest.fixture(scope="function")
def manybody_data_fixture():
    input_data = {
        "molecule": {
            "symbols": ["ne", "ne", "ne"],
            "geometry": [[0, 0, 0], [0, 0, 2], [0, 0, 4]],
            "fragments": [[0], [1], [2]],
        },
        "specification": {
            "keywords": {
                "bsse_type": ["nocp"],
                "levels": {3: "(any)"},
                "supersystem_ie_only": True,
                "return_total_data": True,
            },
            "driver": "energy",
            "specification": {
                "(any)": {"program": "psi4", "driver": "energy", "model": {"method": "mp2", "basis": "cc-pvdz"}},
            },
        },
    }

    ret = {
        "success": True,
        "provenance": {"creator": "me"},
        "input_data": input_data,
        "return_result": -22,
        "properties": {"calcinfo_nmc": 1, "return_energy": -22},
        "component_properties": {
            '["(any)", [1, 2, 3], [1, 2, 3]]': {"calcinfo_natom": 3, "return_energy": -383.7231560517324},
            '["(any)", [1], [1, 2, 3]]': {"calcinfo_natom": 3, "return_energy": -128.68201344613635},
            '["(any)", [2], [1, 2, 3]]': {"calcinfo_natom": 3, "return_energy": -128.68610979339851},
            '["(any)", [3], [1, 2, 3]]': {"calcinfo_natom": 3, "return_energy": -128.68201344613036},
        },
    }

    return ret


@pytest.mark.parametrize("center_name", center_data.keys())
def test_basis_shell_centers(center_name, schema_versions):
    BasisCenter = schema_versions.basis.BasisCenter

    assert BasisCenter(**center_data[center_name])


def test_basis_set_build(request, schema_versions):
    BasisSet = schema_versions.basis.BasisSet

    bas = BasisSet(
        name="custom_basis",
        center_data=center_data,
        atom_map=["bs_sto3g_o", "bs_sto3g_h", "bs_sto3g_h", "bs_def2tzvp_zr"],
    )
    drop_qcsk(bas, request.node.name)

    assert len(bas.center_data) == 3
    assert len(bas.atom_map) == 4
    assert bas.nbf == 21

    es = bas.center_data["bs_sto3g_o"].electron_shells
    assert es[0].is_contracted() is False
    assert es[1].is_contracted() is False
    assert es[2].is_contracted()

    assert es[0].exponents == [130.70939, 23.808861, 6.4436089]
    assert es[0].coefficients == [[0.15432899, 0.53532814, 0.44463454]]


def test_basis_electron_center_raises(schema_versions):
    ElectronShell = schema_versions.basis.ElectronShell

    data = center_data["bs_sto3g_h"]["electron_shells"][0].copy()

    # Check bad coefficient length
    bad_coef = data.copy()
    bad_coef["coefficients"] = [[5, 3]]
    with pytest.raises(ValueError) as e:
        ElectronShell(**bad_coef)

    assert "does not match the" in str(e.value)

    # Check bad fused shell
    bad_fused = data.copy()
    bad_fused["angular_momentum"] = [0, 1]
    with pytest.raises(ValueError) as e:
        ElectronShell(**bad_fused)

    assert "fused shell" in str(e.value)


def test_basis_ecp_center_raises(schema_versions):
    basis = schema_versions.basis

    # Check coefficients
    data = center_data["bs_def2tzvp_zr"]["ecp_potentials"][0].copy()
    data["coefficients"] = [[5, 3]]

    with pytest.raises(ValueError):
        basis.ECPPotential(**data)

    # Check gaussian_exponents
    data = center_data["bs_def2tzvp_zr"]["ecp_potentials"][0].copy()
    data["gaussian_exponents"] = [5, 3]

    with pytest.raises(ValueError):
        basis.ECPPotential(**data)


def test_basis_map_raises(schema_versions):
    BasisSet = schema_versions.basis.BasisSet

    with pytest.raises(ValueError) as e:
        assert BasisSet(name="custom_basis", center_data=center_data, atom_map=["something_odd"])


def test_result_build(result_data_fixture, request, schema_versions):
    AtomicResult = schema_versions.AtomicResult

    ret = AtomicResult(**result_data_fixture)
    drop_qcsk(ret, request.node.name)
    assert ret.wavefunction is None


def test_result_build_wavefunction_delete(wavefunction_data_fixture, request, schema_versions):
    AtomicResult = schema_versions.AtomicResult

    del wavefunction_data_fixture["protocols"]
    ret = AtomicResult(**wavefunction_data_fixture)
    drop_qcsk(ret, request.node.name)
    assert ret.wavefunction is None


def test_wavefunction_build(wavefunction_data_fixture, request, schema_versions):
    AtomicResult = schema_versions.AtomicResult

    ret = AtomicResult(**wavefunction_data_fixture)
    drop_qcsk(ret, request.node.name)
    assert ret


def test_wavefunction_matrix_size_error(wavefunction_data_fixture, schema_versions):
    AtomicResult = schema_versions.AtomicResult

    wavefunction_data_fixture["wavefunction"]["scf_orbitals_a"] = np.random.rand(2, 2)
    with pytest.raises(ValueError) as e:
        AtomicResult(**wavefunction_data_fixture)

    assert "castable to shape" in str(e.value)


def test_wavefunction_return_result_pointer(wavefunction_data_fixture, schema_versions):
    AtomicResult = schema_versions.AtomicResult

    del wavefunction_data_fixture["wavefunction"]["scf_orbitals_a"]
    with pytest.raises(ValueError) as e:
        AtomicResult(**wavefunction_data_fixture)

    assert "does not exist" in str(e.value)


@pytest.mark.parametrize(
    "protocol, restricted, provided, expected",
    [
        ("none", True, ["orbitals_a", "orbitals_b"], []),
        (None, True, ["orbitals_a", "orbitals_b"], []),
        ("all", False, ["orbitals_a", "orbitals_b"], ["orbitals_a", "orbitals_b"]),
        ("all", True, ["orbitals_a", "orbitals_b"], ["orbitals_a"]),
        (
            "orbitals_and_eigenvalues",
            False,
            ["orbitals_a", "orbitals_b", "fock_a", "fock_b"],
            ["orbitals_a", "orbitals_b"],
        ),
        (
            "orbitals_and_eigenvalues",
            True,
            ["orbitals_a", "orbitals_b", "eigenvalues_a", "fock_a", "fock_b"],
            ["orbitals_a", "eigenvalues_a"],
        ),
        ("return_results", True, ["orbitals_a", "fock_a", "fock_b"], ["orbitals_a", "fock_a"]),
        (
            "occupations_and_eigenvalues",
            True,
            ["orbitals_a", "orbitals_b", "occupations_a", "occupations_b", "eigenvalues_a", "eigenvalues_b"],
            ["occupations_a", "eigenvalues_a"],
        ),
        (
            "occupations_and_eigenvalues",
            False,
            ["orbitals_a", "orbitals_b", "occupations_a", "occupations_b", "eigenvalues_a", "eigenvalues_b"],
            ["occupations_a", "occupations_b", "eigenvalues_a", "eigenvalues_b"],
        ),
    ],
)
def test_wavefunction_protocols(
    protocol, restricted, provided, expected, wavefunction_data_fixture, request, schema_versions
):
    AtomicResult = schema_versions.AtomicResult

    wfn_data = wavefunction_data_fixture["wavefunction"]

    if protocol is None:
        wavefunction_data_fixture.pop("protocols")
    else:
        wavefunction_data_fixture["protocols"]["wavefunction"] = protocol

    wfn_data["restricted"] = restricted
    bas = wfn_data["basis"]

    for name in provided:
        scf_name = "scf_" + name
        wfn_data[name] = scf_name
        if "eigen" in name:
            wfn_data[scf_name] = np.random.rand(bas.nbf)
        else:
            wfn_data[scf_name] = np.random.rand(bas.nbf, bas.nbf)

    wfn = AtomicResult(**wavefunction_data_fixture)
    drop_qcsk(wfn, request.node.name)

    if len(expected) == 0:
        assert wfn.wavefunction is None
    else:
        expected_keys = set(expected) | {"scf_" + x for x in expected} | {"basis", "restricted"}
        assert wfn.wavefunction.model_dump().keys() == expected_keys


@pytest.mark.parametrize(
    "protocol, provided, expected",
    [
        ("none", ["input", "gms.dat", "DIPOL"], []),
        (None, ["input", "gms.dat", "DIPOL"], []),
        ("input", ["input", "gms.dat", "DIPOL"], ["input"]),
        ("all", ["input", "gms.dat", "DIPOL"], ["input", "gms.dat", "DIPOL"]),
        ("all", ["DIPOL"], ["DIPOL"]),
        ("input", ["gms.dat"], ["input"]),
    ],
)
def test_native_protocols(protocol, provided, expected, native_data_fixture, request, schema_versions):
    AtomicResult = schema_versions.AtomicResult

    native_data = native_data_fixture["native_files"]

    if protocol is None:
        native_data_fixture.pop("protocols")
    else:
        native_data_fixture["protocols"]["native_files"] = protocol

    for name in list(native_data.keys()):
        if name not in provided:
            native_data.pop(name)

    wfn = AtomicResult(**native_data_fixture)
    drop_qcsk(wfn, request.node.name)

    if len(expected) == 0:
        assert wfn.native_files == {}
    else:
        expected_keys = set(expected)
        assert wfn.native_files.keys() == expected_keys


@pytest.mark.parametrize(
    "keep, indices",
    [(None, [0, 1, 2, 3, 4]), ("all", [0, 1, 2, 3, 4]), ("initial_and_final", [0, 4]), ("final", [4]), ("none", [])],
)
def test_optimization_trajectory_protocol(keep, indices, optimization_data_fixture, schema_versions):
    OptimizationResult = schema_versions.OptimizationResult

    if keep is not None:
        optimization_data_fixture["protocols"] = {"trajectory": keep}
    opt = OptimizationResult(**optimization_data_fixture)

    assert len(opt.trajectory) == len(indices)
    for result, index in zip(opt.trajectory, indices):
        assert result.return_result == index


@pytest.mark.parametrize(
    "default, defined, default_result, defined_result",
    [(None, None, True, None), (False, {"a": True}, False, {"a": True})],
)
def test_error_correction_protocol(
    default, defined, default_result, defined_result, result_data_fixture, request, schema_versions
):
    AtomicResult = schema_versions.AtomicResult

    policy = {}
    if default is not None:
        policy["default_policy"] = default
    if defined is not None:
        policy["policies"] = defined
    result_data_fixture["protocols"] = {"error_correction": policy}
    res = AtomicResult(**result_data_fixture)
    drop_qcsk(res, request.node.name)

    assert res.protocols.error_correction.default_policy == default_result
    assert res.protocols.error_correction.policies == defined_result


def test_error_correction_logic(schema_versions):
    ErrorCorrectionProtocol = schema_versions.results.ErrorCorrectionProtocol

    # Make sure we are permissive by default
    correction_policy = ErrorCorrectionProtocol()
    assert correction_policy.allows("a")

    # Add ability to turn off error correction
    correction_policy = ErrorCorrectionProtocol(policies={"a": False})
    correction_policy.policies["a"] = False
    assert not correction_policy.allows("a")

    # Try no execution by default
    correction_policy = ErrorCorrectionProtocol(default_policy=False)
    assert not correction_policy.allows("a")

    # Make sure it is still overridable
    correction_policy = ErrorCorrectionProtocol(default_policy=False, policies={"a": True})
    assert correction_policy.allows("a")


def test_result_build_stdout_delete(result_data_fixture, request, schema_versions):
    AtomicResult = schema_versions.AtomicResult

    result_data_fixture["protocols"] = {"stdout": False}
    ret = AtomicResult(**result_data_fixture)
    drop_qcsk(ret, request.node.name)
    assert ret.stdout is None


def test_result_build_stdout(result_data_fixture, request, schema_versions):
    AtomicResult = schema_versions.AtomicResult

    ret = AtomicResult(**result_data_fixture)
    drop_qcsk(ret, request.node.name)
    assert ret.stdout == "I ran."


def test_failed_operation(result_data_fixture, request, schema_versions):
    Molecule = schema_versions.Molecule
    FailedOperation = schema_versions.FailedOperation
    ComputeError = schema_versions.ComputeError

    water = Molecule.from_data(
        """
        O 0 0 0
        H 0 0 2
        H 0 2 0
    """
    )
    drop_qcsk(water, request.node.name)

    failed = FailedOperation(
        extras={"garbage": water},
        input_data=result_data_fixture,
        error={"error_type": "expected_testing_error", "error_message": "If you see this, its all good"},
    )
    assert isinstance(failed.error, ComputeError)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert isinstance(failed.dict(), dict)
    assert isinstance(failed.model_dump(), dict)

    failed_json = failed.model_dump_json()
    assert isinstance(failed_json, str)
    assert "its all good" in failed_json
    assert isinstance(failed_json, str)


def test_result_properties_array(request, schema_versions):
    AtomicResultProperties = schema_versions.AtomicResultProperties

    lquad = [1, 2, 3, 2, 4, 5, 3, 5, 6]

    obj = AtomicResultProperties(
        scf_one_electron_energy="-5.0", scf_dipole_moment=[1, 2, 3], scf_quadrupole_moment=lquad
    )
    drop_qcsk(obj, request.node.name)

    assert pytest.approx(obj.scf_one_electron_energy) == -5.0
    assert obj.scf_dipole_moment.shape == (3,)
    assert obj.scf_quadrupole_moment.shape == (3, 3)

    assert obj.model_dump().keys() == {"scf_one_electron_energy", "scf_dipole_moment", "scf_quadrupole_moment"}
    assert np.array_equal(obj.scf_quadrupole_moment, np.array(lquad).reshape(3, 3))
    # assert obj.dict()["scf_quadrupole_moment"] == lquad  # when properties.dict() was forced json
    assert np.array_equal(
        obj.model_dump()["scf_quadrupole_moment"], np.array(lquad).reshape(3, 3)
    )  # now remains ndarray


def test_result_derivatives_array(request, schema_versions):
    AtomicResultProperties = schema_versions.AtomicResultProperties

    nat = 4
    lgrad = list(range(nat * 3))
    lhess = list(range(nat * nat * 9))

    obj = AtomicResultProperties(calcinfo_natom=nat, return_gradient=lgrad, scf_total_hessian=lhess)
    drop_qcsk(obj, request.node.name)

    assert obj.calcinfo_natom == 4
    assert obj.return_gradient.shape == (4, 3)
    assert obj.scf_total_hessian.shape == (12, 12)
    assert obj.model_dump().keys() == {"calcinfo_natom", "return_gradient", "scf_total_hessian"}


@pytest.fixture(scope="function")
def every_model_fixture(request):
    datas = {}

    smodel = "Molecule-A"
    data = request.getfixturevalue("result_data_fixture")
    data = data["molecule"].model_dump()
    datas[smodel] = data

    smodel = "Molecule-B"
    data = {"symbols": ["O", "H", "H"], "geometry": [0, 0, 0, 0, 0, 2, 0, 2, 0]}
    datas[smodel] = data

    smodel = "BasisSet"
    data = {"name": "custom", "center_data": center_data, "atom_map": ["bs_sto3g_o", "bs_sto3g_h", "bs_sto3g_h"]}
    datas[smodel] = data

    smodel = "FailedOperation"
    data = {
        "input_data": request.getfixturevalue("result_data_fixture"),
        "error": {"error_type": "expected_testing_error", "error_message": "If you see this, its all good"},
    }
    datas[smodel] = data

    smodel = "AtomicInput"
    data = request.getfixturevalue("result_data_fixture")
    data = {k: data[k] for k in ["molecule", "model", "driver"]}
    datas[smodel] = data

    smodel = "QCInputSpecification"  # TODO "AtomicSpecification"
    data = {"driver": "hessian", "model": {"basis": "def2-svp", "method": "CC"}}
    datas[smodel] = data

    smodel = "AtomicResultProtocols"  # TODO "AtomicProtocols"
    data = {"wavefunction": "occupations_and_eigenvalues"}
    datas[smodel] = data

    smodel = "AtomicResult"
    data = request.getfixturevalue("result_data_fixture")
    datas[smodel] = data

    smodel = "AtomicResultProperties"  # TODO "AtomicProperties"
    data = {"scf_one_electron_energy": "-5.0", "scf_dipole_moment": [1, 2, 3], "ccsd_dipole_moment": None}
    datas[smodel] = data

    smodel = "WavefunctionProperties"
    data = request.getfixturevalue("wavefunction_data_fixture")
    data = data["wavefunction"]
    datas[smodel] = data

    smodel = "OptimizationInput"
    data = request.getfixturevalue("optimization_data_fixture")
    data = {k: data[k] for k in ["initial_molecule", "input_specification"]}
    datas[smodel] = data

    smodel = "OptimizationSpecification"
    data = {"procedure": "pyberny"}
    datas[smodel] = data

    smodel = "OptimizationProtocols"
    data = {"trajectory": "initial_and_final"}
    datas[smodel] = data

    smodel = "OptimizationResult"
    data = request.getfixturevalue("optimization_data_fixture")
    datas[smodel] = data

    smodel = "OptimizationProperties"  # TODO actually collect
    data = {"optimization_iterations": 14}
    datas[smodel] = data

    smodel = "TorsionDriveInput"
    data = request.getfixturevalue("torsiondrive_data_fixture")
    data = {k: data[k] for k in ["initial_molecule", "input_specification", "optimization_spec", "keywords"]}
    datas[smodel] = data

    # smodel = "TorsionDriveSpecification"  # DNE

    smodel = "TDKeywords"  # TODO "TorsionDriveKeywords"
    data = {"dihedrals": [(2, 0, 1, 5)], "grid_spacing": [180]}
    datas[smodel] = data

    # smodel = "TorsionDriveProtocols"  # DNE

    smodel = "TorsionDriveResult"
    data = request.getfixturevalue("torsiondrive_data_fixture")
    datas[smodel] = data

    # smodel = "TorsionDriveProperties"  # DNE

    smodel = "ManyBodyInput"
    data = request.getfixturevalue("manybody_data_fixture")
    data = data["input_data"]
    datas[smodel] = data

    smodel = "ManyBodySpecification"
    data = request.getfixturevalue("manybody_data_fixture")
    data = data["input_data"]["specification"]
    datas[smodel] = data

    smodel = "ManyBodyKeywords"
    data = {"bsse_type": "ssfc", "levels": {2: "md"}}
    datas[smodel] = data

    smodel = "ManyBodyProtocols"
    data = {"component_results": "all"}
    datas[smodel] = data

    smodel = "ManyBodyResult"
    data = request.getfixturevalue("manybody_data_fixture")
    datas[smodel] = data

    smodel = "ManyBodyResultProperties"  # "ManyBodyProperties"
    data = request.getfixturevalue("manybody_data_fixture")
    data = data["properties"]
    datas[smodel] = data

    return datas


_model_classes_struct = [
    # fmt: off
    # v1_class, v2_class, test ID
    pytest.param("Molecule-A",                  "Molecule-A",                   id="Mol-A"),
    pytest.param("Molecule-B",                  "Molecule-B",                   id="Mol-B"),
    pytest.param("BasisSet",                    "BasisSet",                     id="BasisSet"),
    pytest.param("FailedOperation",             "FailedOperation",              id="FailedOp"),
    pytest.param("AtomicInput",                 "AtomicInput",                  id="AtIn"),
    pytest.param("QCInputSpecification",        "QCInputSpecification",         id="AtSpec"),  # TODO AtomicSpecification
    pytest.param("AtomicResultProtocols",       "AtomicResultProtocols",        id="AtPtcl"),  # TODO AtomicProtocols
    pytest.param("AtomicResult",                "AtomicResult",                 id="AtRes"),
    pytest.param("AtomicResultProperties",      "AtomicResultProperties",       id="AtProp"),  # TODO AtomicProperties 
    pytest.param("WavefunctionProperties",      "WavefunctionProperties",       id="WfnProp"),
    pytest.param("OptimizationInput",           "OptimizationInput",            id="OptIn"), 
    pytest.param("OptimizationSpecification",   "OptimizationSpecification",    id="OptSpec"),
    pytest.param("OptimizationProtocols",       "OptimizationProtocols",        id="OptPtcl"),
    pytest.param("OptimizationResult",          "OptimizationResult",           id="OptRes"),
    # pytest.param(None,                        "OptimizationProperties",       id="OptProp"),
    pytest.param("TorsionDriveInput",           "TorsionDriveInput",            id="TDIn"), 
    # pytest.param(None,                        "TorsionDriveSpecification",    id="TDSpec"), 
    pytest.param("TDKeywords",                  "TDKeywords",                   id="TDKw"),  # TODO TorsionDriveKeywords
    # pytest.param(None,                        "TorsionDriveProtocols",        id="TDPtcl"),
    pytest.param("TorsionDriveResult",          "TorsionDriveResult",           id="TDRes"), 
    # pytest.param(None,                        "TorsionDriveProperties",       id="TDProp"),
    pytest.param("ManyBodyInput",               None,                           id="MBIn", marks=using_qcmb), 
    pytest.param("ManyBodySpecification",       None,                           id="MBSpec", marks=using_qcmb), 
    pytest.param("ManyBodyKeywords",            None,                           id="MBKw", marks=using_qcmb),
    pytest.param("ManyBodyProtocols",           None,                           id="MBPtcl", marks=using_qcmb),
    pytest.param("ManyBodyResult",              None,                           id="MBRes", marks=using_qcmb), 
    pytest.param("ManyBodyResultProperties",    None,                           id="MBProp", marks=using_qcmb),
    # TODO ManyBodyProperties
    # fmt: on
]


@pytest.mark.parametrize("smodel1,smodel2", _model_classes_struct)
def test_model_survey_success(smodel1, smodel2, every_model_fixture, request, schema_versions):
    anskey = request.node.callspec.id.replace("None", "v1")
    # fmt: off
    ans = {
        "v1-Mol-A"    : None,  "v2-Mol-A"    : None,
        "v1-Mol-B"    : None,  "v2-Mol-B"    : None,
        "v1-BasisSet" : None,  "v2-BasisSet" : None,
        "v1-FailedOp" : False, "v2-FailedOp" : False,
        "v1-AtIn"     : None,  "v2-AtIn"     : None,
        "v1-AtSpec"   : None,  "v2-AtSpec"   : None,
        "v1-AtPtcl"   : None,  "v2-AtPtcl"   : None,
        "v1-AtRes"    : True,  "v2-AtRes"    : True,
        "v1-AtProp"   : None,  "v2-AtProp"   : None,
        "v1-WfnProp"  : None,  "v2-WfnProp"  : None,
        "v1-OptIn"    : None,  "v2-OptIn"    : None,
        "v1-OptSpec"  : None,  "v2-OptSpec"  : None,
        "v1-OptPtcl"  : None,  "v2-OptPtcl"  : None,
        "v1-OptRes"   : True,  "v2-OptRes"   : True,
        "v1-OptProp"  : None,  "v2-OptProp"  : None,  # v1 DNE
        "v1-TDIn"     : None,  "v2-TDIn"     : None,
        "v1-TDSpec"   : None,  "v2-TDSpec"   : None,  # v1 DNE
        "v1-TDKw"     : None,  "v2-TDKw"     : None,
        "v1-TDPtcl"   : None,  "v2-TDPtcl"   : None,  # v1 DNE
        "v1-TDRes"    : True,  "v2-TDRes"    : True,
        "v1-TDProp"   : None,  "v2-TDProp"   : None,  # v1 DNE
        "v1-MBIn"     : None,  "v2-MBIn"     : None,  # v2 DNE
        "v1-MBSpec"   : None,  "v2-MBSpec"   : None,  # v2 DNE
        "v1-MBKw"     : None,  "v2-MBKw"     : None,  # v2 DNE
        "v1-MBPtcl"   : None,  "v2-MBPtcl"   : None,  # v2 DNE
        "v1-MBRes"    : True,  "v2-MBRes"    : True,  # v2 DNE
        "v1-MBProp"   : None,  "v2-MBProp"   : None,  # v2 DNE
    }[anskey]
    # fmt: on

    fieldsattr = "model_fields" if "v2" in anskey else "__fields__"
    smodel = smodel2 if "v2" in anskey else smodel1
    if smodel is None:
        pytest.skip("model not available for this schema version")
    if "ManyBody" in smodel:
        import qcmanybody

        model = getattr(qcmanybody.models, smodel.split("-")[0])
    else:
        model = getattr(schema_versions, smodel.split("-")[0])
    data = every_model_fixture[smodel]

    # check default success set
    instance = model(**data)
    fld = "success"
    if ans is None:
        assert fld not in (cptd := getattr(instance, fieldsattr)), f"[a] field {fld} unexpectedly present: {cptd}"
    else:
        assert (cptd := getattr(instance, fld, "not found!")) == ans, f"[a] field {fld} = {cptd} != {ans}"

    # check success override
    if ans is not None:
        data["success"] = not ans
        if "v2" in anskey:
            # v2 has enforced T/F
            with pytest.raises((pydantic.v1.ValidationError, pydantic.ValidationError)) as e:
                instance = model(**data)
            assert (cptd := getattr(instance, fld, "not found!")) == ans, f"[b] field {fld} = {cptd} != {ans}"
        else:
            # v1 can be reset to T/F
            instance = model(**data)
            assert (cptd := getattr(instance, fld, "not found!")) == (not ans), f"[b] field {fld} = {cptd} != {ans}"


@pytest.mark.parametrize(
    "smodel", ["molecule", "atomicresultproperties", "atomicinput", "atomicresult", "optimizationresult", "basisset"]
)
def test_model_dictable(result_data_fixture, optimization_data_fixture, smodel, schema_versions, request):
    qcsk_ver = "v2" if ("v2" in request.node.name) else "v1"

    if smodel == "molecule":
        model = schema_versions.Molecule
        data = result_data_fixture["molecule"].model_dump()
        sver = (2, 2)  # TODO , 3)

    elif smodel == "atomicresultproperties":
        model = schema_versions.AtomicResultProperties
        data = {"scf_one_electron_energy": "-5.0", "scf_dipole_moment": [1, 2, 3], "ccsd_dipole_moment": None}
        sver = (None, 2)

    elif smodel == "atomicinput":
        model = schema_versions.AtomicInput
        data = {k: result_data_fixture[k] for k in ["molecule", "model", "driver"]}
        sver = (1, 2)

    elif smodel == "atomicresult":
        model = schema_versions.AtomicResult
        data = result_data_fixture
        sver = (1, 2)

    elif smodel == "optimizationresult":
        model = schema_versions.OptimizationResult
        data = optimization_data_fixture
        sver = (1, 2)

    elif smodel == "basisset":
        model = schema_versions.BasisSet
        data = {"name": "custom", "center_data": center_data, "atom_map": ["bs_sto3g_o", "bs_sto3g_h", "bs_sto3g_h"]}
        sver = (1, 2)

    def ver_tests(qcsk_ver):
        if qcsk_ver == "v1":
            if sver[0] is not None:
                assert instance.schema_version == sver[0]
            assert isinstance(instance, pydantic.v1.BaseModel)
        elif qcsk_ver == "v2":
            if sver[1] is not None:
                assert instance.schema_version == sver[1]
            assert isinstance(instance, pydantic.BaseModel)

    instance = model(**data)
    ver_tests(qcsk_ver)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        instance = model(**instance.dict())
    assert instance
    ver_tests(qcsk_ver)

    instance2 = model(**data)
    ver_tests(qcsk_ver)
    instance2 = model(**instance2.model_dump())
    assert instance2
    ver_tests(qcsk_ver)


def test_result_model_deprecations(result_data_fixture, optimization_data_fixture, request):
    if "v1" not in request.node.name:
        # schema_versions coming from fixtures despite not being explicitly present
        pytest.skip("Deprecations from 2019 only available from qcel.models.v1")

    with pytest.warns(DeprecationWarning):
        qcel.models.v1.ResultProperties(scf_one_electron_energy="-5.0")

    with pytest.warns(DeprecationWarning):
        qcel.models.v1.ResultInput(**{k: result_data_fixture[k] for k in ["molecule", "model", "driver"]})

    with pytest.warns(DeprecationWarning):
        qcel.models.v1.Result(**result_data_fixture)

    with pytest.warns(DeprecationWarning):
        qcel.models.v1.Optimization(**optimization_data_fixture)


@pytest.mark.parametrize(
    "retres,atprop,rettyp,jsntyp",
    [
        (15, "mp2_correlation_energy", float, float),
        (15.0, "mp2_correlation_energy", float, float),
        (np.array([15.3]), "ccsd_total_energy", float, float),
        ([1.0, -2.5, 3, 0, 0, 0, 0, 0, 0], "return_gradient", np.ndarray, list),
        (np.array([1.0, -2.5, 3, 0, 0, 0, 0, 0, 0]), "return_gradient", np.ndarray, list),
        ({"cat1": "tail", "cat2": "whiskers"}, None, str, str),
        ({"float1": 4.4, "float2": -9.9}, None, float, float),
        ({"list1": [-1.0, 4.4], "list2": [-9.9, 2]}, None, list, list),
        ({"arr1": np.array([-1.0, 4.4]), "arr2": np.array([-9.9, 2])}, None, np.ndarray, list),
    ],
)
def test_return_result_types(result_data_fixture, retres, atprop, rettyp, jsntyp, request, schema_versions):
    AtomicResult = schema_versions.AtomicResult

    working_res = copy.deepcopy(result_data_fixture)
    working_res["return_result"] = retres
    if atprop:
        working_res["properties"]["calcinfo_natom"] = 3
        working_res["properties"][atprop] = retres
    atres = AtomicResult(**working_res)

    if isinstance(retres, dict):
        for v in atres.return_result.values():
            assert isinstance(v, rettyp)
    else:
        if atprop:
            assert isinstance(getattr(atres.properties, atprop), rettyp)
        assert isinstance(atres.return_result, rettyp)

    datres = atres.model_dump()
    if isinstance(retres, dict):
        for v in datres["return_result"].values():
            assert isinstance(v, rettyp)
    else:
        if atprop:
            assert isinstance(datres["properties"][atprop], rettyp)
        assert isinstance(datres["return_result"], rettyp)

    jatres = json.loads(atres.model_dump_json())
    if isinstance(retres, dict):
        for v in jatres["return_result"].values():
            assert isinstance(v, jsntyp)
    else:
        if atprop:
            assert isinstance(jatres["properties"][atprop], jsntyp)
        assert isinstance(jatres["return_result"], jsntyp)
