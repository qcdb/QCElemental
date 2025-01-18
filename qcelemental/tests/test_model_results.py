import numpy as np
import pytest

import qcelemental as qcel
from qcelemental.models import basis

from .addons import drop_qcsk

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
def result_data_fixture():
    mol = qcel.models.Molecule.from_data(
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
def wavefunction_data_fixture(result_data_fixture):
    bas = basis.BasisSet(
        name="custom_basis", center_data=center_data, atom_map=["bs_sto3g_o", "bs_sto3g_h", "bs_sto3g_h"]
    )
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


@pytest.mark.parametrize("center_name", center_data.keys())
def test_basis_shell_centers(center_name):
    assert basis.BasisCenter(**center_data[center_name])


def test_basis_set_build(request):
    bas = basis.BasisSet(
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


def test_basis_electron_center_raises():
    data = center_data["bs_sto3g_h"]["electron_shells"][0].copy()

    # Check bad coefficient length
    bad_coef = data.copy()
    bad_coef["coefficients"] = [[5, 3]]
    with pytest.raises(ValueError) as e:
        basis.ElectronShell(**bad_coef)

    assert "does not match the" in str(e.value)

    # Check bad fused shell
    bad_fused = data.copy()
    bad_fused["angular_momentum"] = [0, 1]
    with pytest.raises(ValueError) as e:
        basis.ElectronShell(**bad_fused)

    assert "fused shell" in str(e.value)


def test_basis_ecp_center_raises():
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


def test_basis_map_raises():
    with pytest.raises(ValueError) as e:
        assert basis.BasisSet(name="custom_basis", center_data=center_data, atom_map=["something_odd"])


def test_result_build(result_data_fixture, request):
    ret = qcel.models.AtomicResult(**result_data_fixture)
    drop_qcsk(ret, request.node.name)
    assert ret.wavefunction is None


def test_result_build_wavefunction_delete(wavefunction_data_fixture, request):
    del wavefunction_data_fixture["protocols"]
    ret = qcel.models.AtomicResult(**wavefunction_data_fixture)
    drop_qcsk(ret, request.node.name)
    assert ret.wavefunction is None


def test_wavefunction_build(wavefunction_data_fixture, request):
    ret = qcel.models.AtomicResult(**wavefunction_data_fixture)
    drop_qcsk(ret, request.node.name)
    assert ret


def test_wavefunction_matrix_size_error(wavefunction_data_fixture):
    wavefunction_data_fixture["wavefunction"]["scf_orbitals_a"] = np.random.rand(2, 2)
    with pytest.raises(ValueError) as e:
        qcel.models.AtomicResult(**wavefunction_data_fixture)

    assert "castable to shape" in str(e.value)


def test_wavefunction_return_result_pointer(wavefunction_data_fixture):
    del wavefunction_data_fixture["wavefunction"]["scf_orbitals_a"]
    with pytest.raises(ValueError) as e:
        qcel.models.AtomicResult(**wavefunction_data_fixture)

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
def test_wavefunction_protocols(protocol, restricted, provided, expected, wavefunction_data_fixture, request):
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

    wfn = qcel.models.AtomicResult(**wavefunction_data_fixture)
    drop_qcsk(wfn, request.node.name)

    if len(expected) == 0:
        assert wfn.wavefunction is None
    else:
        expected_keys = set(expected) | {"scf_" + x for x in expected} | {"basis", "restricted"}
        assert wfn.wavefunction.dict().keys() == expected_keys


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
def test_native_protocols(protocol, provided, expected, native_data_fixture, request):
    native_data = native_data_fixture["native_files"]

    if protocol is None:
        native_data_fixture.pop("protocols")
    else:
        native_data_fixture["protocols"]["native_files"] = protocol

    for name in list(native_data.keys()):
        if name not in provided:
            native_data.pop(name)

    wfn = qcel.models.AtomicResult(**native_data_fixture)
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
def test_optimization_trajectory_protocol(keep, indices, optimization_data_fixture):
    if keep is not None:
        optimization_data_fixture["protocols"] = {"trajectory": keep}
    opt = qcel.models.OptimizationResult(**optimization_data_fixture)

    assert len(opt.trajectory) == len(indices)
    for result, index in zip(opt.trajectory, indices):
        assert result.return_result == index


@pytest.mark.parametrize(
    "default, defined, default_result, defined_result",
    [(None, None, True, None), (False, {"a": True}, False, {"a": True})],
)
def test_error_correction_protocol(default, defined, default_result, defined_result, result_data_fixture, request):
    policy = {}
    if default is not None:
        policy["default_policy"] = default
    if defined is not None:
        policy["policies"] = defined
    result_data_fixture["protocols"] = {"error_correction": policy}
    res = qcel.models.AtomicResult(**result_data_fixture)
    drop_qcsk(res, request.node.name)

    assert res.protocols.error_correction.default_policy == default_result
    assert res.protocols.error_correction.policies == defined_result


def test_error_correction_logic():
    # Make sure we are permissive by default
    correction_policy = qcel.models.results.ErrorCorrectionProtocol()
    assert correction_policy.allows("a")

    # Add ability to turn off error correction
    correction_policy = qcel.models.results.ErrorCorrectionProtocol(policies={"a": False})
    correction_policy.policies["a"] = False
    assert not correction_policy.allows("a")

    # Try no execution by default
    correction_policy = qcel.models.results.ErrorCorrectionProtocol(default_policy=False)
    assert not correction_policy.allows("a")

    # Make sure it is still overridable
    correction_policy = qcel.models.results.ErrorCorrectionProtocol(default_policy=False, policies={"a": True})
    assert correction_policy.allows("a")


def test_result_build_stdout_delete(result_data_fixture, request):
    result_data_fixture["protocols"] = {"stdout": False}
    ret = qcel.models.AtomicResult(**result_data_fixture)
    drop_qcsk(ret, request.node.name)
    assert ret.stdout is None


def test_result_build_stdout(result_data_fixture, request):
    ret = qcel.models.AtomicResult(**result_data_fixture)
    drop_qcsk(ret, request.node.name)
    assert ret.stdout == "I ran."


def test_failed_operation(result_data_fixture, request):
    water = qcel.models.Molecule.from_data(
        """
        O 0 0 0
        H 0 0 2
        H 0 2 0
    """
    )
    drop_qcsk(water, request.node.name)

    failed = qcel.models.FailedOperation(
        extras={"garbage": water},
        input_data=result_data_fixture,
        error={"error_type": "expected_testing_error", "error_message": "If you see this, its all good"},
    )
    assert isinstance(failed.error, qcel.models.ComputeError)
    assert isinstance(failed.dict(), dict)
    failed_json = failed.json()
    assert isinstance(failed_json, str)
    assert "its all good" in failed_json


def test_model_json_allows_kwargs(result_data_fixture):
    result = qcel.models.AtomicResult(**result_data_fixture)
    no_indent = result.json()
    assert "\n" not in no_indent
    indent = result.json(indent=2)
    assert "\n" in indent


def test_result_properties_array(request):
    lquad = [1, 2, 3, 2, 4, 5, 3, 5, 6]

    obj = qcel.models.AtomicResultProperties(
        scf_one_electron_energy="-5.0", scf_dipole_moment=[1, 2, 3], scf_quadrupole_moment=lquad
    )
    drop_qcsk(obj, request.node.name)

    assert pytest.approx(obj.scf_one_electron_energy) == -5.0
    assert obj.scf_dipole_moment.shape == (3,)
    assert obj.scf_quadrupole_moment.shape == (3, 3)

    assert obj.dict().keys() == {"scf_one_electron_energy", "scf_dipole_moment", "scf_quadrupole_moment"}
    assert np.array_equal(obj.scf_quadrupole_moment, np.array(lquad).reshape(3, 3))
    # assert obj.dict()["scf_quadrupole_moment"] == lquad  # when properties.dict() was forced json
    assert np.array_equal(obj.dict()["scf_quadrupole_moment"], np.array(lquad).reshape(3, 3))  # now remains ndarray


def test_result_derivatives_array(request):
    nat = 4
    lgrad = list(range(nat * 3))
    lhess = list(range(nat * nat * 9))

    obj = qcel.models.AtomicResultProperties(calcinfo_natom=nat, return_gradient=lgrad, scf_total_hessian=lhess)
    drop_qcsk(obj, request.node.name)

    assert obj.calcinfo_natom == 4
    assert obj.return_gradient.shape == (4, 3)
    assert obj.scf_total_hessian.shape == (12, 12)
    assert obj.dict().keys() == {"calcinfo_natom", "return_gradient", "scf_total_hessian"}


@pytest.mark.parametrize(
    "smodel", ["molecule", "atomicresultproperties", "atomicinput", "atomicresult", "optimizationresult"]
)
def test_model_dictable(result_data_fixture, optimization_data_fixture, smodel):
    if smodel == "molecule":
        model = qcel.models.Molecule
        data = result_data_fixture["molecule"].dict()

    elif smodel == "atomicresultproperties":
        model = qcel.models.AtomicResultProperties
        data = {"scf_one_electron_energy": "-5.0", "scf_dipole_moment": [1, 2, 3], "ccsd_dipole_moment": None}

    elif smodel == "atomicinput":
        model = qcel.models.AtomicInput
        data = {k: result_data_fixture[k] for k in ["molecule", "model", "driver"]}

    elif smodel == "atomicresult":
        model = qcel.models.AtomicResult
        data = result_data_fixture

    elif smodel == "optimizationresult":
        model = qcel.models.OptimizationResult
        data = optimization_data_fixture

    instance = model(**data)
    assert model(**instance.dict())


def test_result_model_deprecations(result_data_fixture, optimization_data_fixture):
    with pytest.warns(DeprecationWarning):
        qcel.models.ResultProperties(scf_one_electron_energy="-5.0")

    with pytest.warns(DeprecationWarning):
        qcel.models.ResultInput(**{k: result_data_fixture[k] for k in ["molecule", "model", "driver"]})

    with pytest.warns(DeprecationWarning):
        qcel.models.Result(**result_data_fixture)

    with pytest.warns(DeprecationWarning):
        qcel.models.Optimization(**optimization_data_fixture)
