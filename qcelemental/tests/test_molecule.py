"""
Tests the imports and exports of the Molecule object.
"""

import warnings

import numpy as np
import pydantic
import pytest

import qcelemental as qcel
from qcelemental.exceptions import NotAnElementError
from qcelemental.testing import compare, compare_values

from .addons import Molecule, serialize_extensions, using_msgpack, using_nglview


@pytest.fixture(scope="function")
def water_molecule_data():
    return """
    0 1
    O  -1.551007  -0.114520   0.000000
    H  -1.934259   0.762503   0.000000
    H  -0.599677   0.040712   0.000000
    """


@pytest.fixture(scope="function")
def water_dimer_minima_data():
    dmol = """
    0 1
    O  -1.551007  -0.114520   0.000000
    H  -1.934259   0.762503   0.000000
    H  -0.599677   0.040712   0.000000
    --
    O   1.350625   0.111469   0.000000
    H   1.680398  -0.373741  -0.758561
    H   1.680398  -0.373741   0.758561
    """
    return {"data": dmol, "dtype": "psi4", "orient": True}


def test_molecule_data_constructor_numpy(Molecule, water_dimer_minima_data):
    water_dimer_minima = Molecule.from_data(**water_dimer_minima_data)

    water_psi = water_dimer_minima.model_copy()
    ele = np.array(water_psi.atomic_numbers).reshape(-1, 1)
    npwater = np.hstack((ele, water_psi.geometry * qcel.constants.conversion_factor("Bohr", "angstrom")))

    water_from_np = Molecule.from_data(npwater, name="water dimer", dtype="numpy", frags=[3])
    assert water_psi == water_from_np

    water_from_np = Molecule.from_data(npwater, name="water dimer", frags=[3])
    assert water_psi == water_from_np
    assert water_psi.get_molecular_formula() == "H4O2"
    assert water_psi.get_molecular_formula(order="alphabetical") == "H4O2"
    assert water_psi.get_molecular_formula(order="hill") == "H4O2"


def test_molecule_data_constructor_dict(Molecule, water_dimer_minima_data, request):
    water_dimer_minima = Molecule.from_data(**water_dimer_minima_data)

    water_psi = water_dimer_minima.model_copy()

    # Check the JSON construct/deconstruct
    water_from_json = Molecule.from_data(water_psi.model_dump())
    assert water_psi == water_from_json

    water_from_json = Molecule.from_data(water_psi.model_dump_json(), "json")
    assert water_psi == water_from_json
    assert water_psi == Molecule.from_data(water_psi.to_string("psi4"), dtype="psi4")

    assert (
        water_psi.get_hash() == "3c4b98f515d64d1adc1648fe1fe1d6789e978d34"  # pragma: allowlist secret
    )  # copied from schema_version=1
    assert water_psi.schema_version == 3 if "v2" in request.node.name else 2
    assert water_psi.schema_name == "qcschema_molecule"


def test_molecule_data_constructor_error(Molecule):
    with pytest.raises(TypeError):
        Molecule.from_data([])

    with pytest.raises(KeyError):
        Molecule.from_data({}, dtype="bad")


def test_hash_canary(Molecule):
    water_dimer_minima = Molecule.from_data(
        """
    0 1
    O  -1.551007  -0.114520   0.000000
    H  -1.934259   0.762503   0.000000
    H  -0.599677   0.040712   0.000000
    --
    O   1.350625   0.111469   0.000000
    H   1.680398  -0.373741  -0.758561
    H   1.680398  -0.373741   0.758561
    """,
        dtype="psi4",
    )
    assert water_dimer_minima.get_hash() == "42f3ac52af52cf2105c252031334a2ad92aa911c"  # pragma: allowlist secret

    # Check orientation
    mol = water_dimer_minima.orient_molecule()
    assert mol.get_hash() == "632490a0601500bfc677e9277275f82fbc45affe"  # pragma: allowlist secret

    frag_0 = mol.get_fragment(0, orient=True)
    frag_1 = mol.get_fragment(1, orient=True)
    assert frag_0.get_hash() == "d0b499739f763e8d3a5556b4ddaeded6a148e4d5"  # pragma: allowlist secret
    assert frag_1.get_hash() == "bdc1f75bd1b7b999ff24783d7c1673452b91beb9"  # pragma: allowlist secret


def test_molecule_np_constructors(Molecule):
    """
    Neon tetramer fun
    """
    ### Neon Tetramer
    neon_from_psi = Molecule.from_data(
        """
        Ne 0.000000 0.000000 0.000000
        --
        Ne 3.100000 0.000000 0.000000
        --
        Ne 0.000000 3.200000 0.000000
        --
        Ne 0.000000 0.000000 3.300000
        units bohr""",
        dtype="psi4",
    )
    ele = np.array([10, 10, 10, 10]).reshape(-1, 1)
    npneon = np.hstack((ele, neon_from_psi.geometry))
    neon_from_np = Molecule.from_data(npneon, name="neon tetramer", dtype="numpy", frags=[1, 2, 3], units="bohr")

    assert neon_from_psi == neon_from_np

    # Check the JSON construct/deconstruct
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        neon_from_json = Molecule.from_data(neon_from_psi.json(), dtype="json")
    assert neon_from_psi == neon_from_json
    assert neon_from_json.get_molecular_formula() == "Ne4"


def test_molecule_compare(Molecule, water_molecule_data):
    water_molecule = Molecule.from_data(water_molecule_data)

    water_molecule2 = water_molecule.model_copy()
    assert water_molecule2 == water_molecule

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        water_molecule3 = water_molecule.copy(update={"geometry": (water_molecule.geometry + np.array([0.1, 0, 0]))})
    assert water_molecule != water_molecule3


def test_water_minima_data(Molecule, water_dimer_minima_data):
    water_dimer_minima = Molecule.from_data(**water_dimer_minima_data)

    # Give it a name
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mol_dict = water_dimer_minima.dict()
    mol_dict["name"] = "water dimer"
    mol = Molecule(orient=True, **mol_dict)

    assert len(mol.pretty_print()) == 661
    assert len(mol.to_string("psi4")) == 479

    assert sum(x == y for x, y in zip(mol.symbols, ["O", "H", "H", "O", "H", "H"])) == mol.geometry.shape[0]
    assert mol.name == "water dimer"
    assert mol.molecular_charge == 0
    assert mol.molecular_multiplicity == 1
    assert np.sum(mol.real) == mol.geometry.shape[0]
    assert np.allclose(mol.fragments, [[0, 1, 2], [3, 4, 5]])
    assert np.allclose(mol.fragment_charges, [0, 0])
    assert np.allclose(mol.fragment_multiplicities, [1, 1])
    assert hasattr(mol, "provenance")
    assert np.allclose(
        mol.geometry,
        [
            [2.81211080, 0.1255717, 0.0],
            [3.48216664, -1.55439981, 0.0],
            [1.00578203, -0.1092573, 0.0],
            [-2.6821528, -0.12325075, 0.0],
            [-3.27523824, 0.81341093, 1.43347255],
            [-3.27523824, 0.81341093, -1.43347255],
        ],
    )
    assert mol.get_hash() == "3c4b98f515d64d1adc1648fe1fe1d6789e978d34"  # pragma: allowlist secret


def test_water_minima_fragment(Molecule, water_dimer_minima_data):
    water_dimer_minima = Molecule.from_data(**water_dimer_minima_data)

    mol = water_dimer_minima.model_copy()
    frag_0 = mol.get_fragment(0, orient=True)
    frag_1 = mol.get_fragment(1, orient=True)
    assert frag_0.get_hash() == "5f31757232a9a594c46073082534ca8a6806d367"  # pragma: allowlist secret
    assert frag_1.get_hash() == "bdc1f75bd1b7b999ff24783d7c1673452b91beb9"  # pragma: allowlist secret

    frag_0_1 = mol.get_fragment(0, 1)
    frag_1_0 = mol.get_fragment(1, 0)

    assert np.array_equal(mol.symbols[:3], frag_0.symbols)
    assert np.allclose(mol.masses[:3], frag_0.masses)

    assert np.array_equal(mol.symbols, frag_0_1.symbols)
    assert np.allclose(mol.geometry, frag_0_1.geometry)

    assert np.array_equal(np.hstack((mol.symbols[3:], mol.symbols[:3])), frag_1_0.symbols)
    assert np.allclose(np.hstack((mol.masses[3:], mol.masses[:3])), frag_1_0.masses)


def test_pretty_print(Molecule, water_dimer_minima_data):
    water_dimer_minima = Molecule.from_data(**water_dimer_minima_data)

    mol = water_dimer_minima.model_copy()
    assert isinstance(mol.pretty_print(), str)


def test_to_string(Molecule, water_dimer_minima_data):
    water_dimer_minima = Molecule.from_data(**water_dimer_minima_data)

    mol = water_dimer_minima.model_copy()
    assert isinstance(mol.to_string("psi4"), str)


@pytest.mark.parametrize(
    "dtype, filext",
    [("json", "json"), ("xyz", "xyz"), ("numpy", "npy"), pytest.param("msgpack", "msgpack", marks=using_msgpack)],
)
def test_to_from_file_simple(tmp_path, dtype, filext, Molecule):
    benchmol = Molecule.from_data(
        """
    O 0 0 0
    H 0 1.5 0
    H 0 0 1.5
    """
    )

    p = tmp_path / ("water." + filext)
    benchmol.to_file(p, dtype=dtype)

    mol = Molecule.from_file(p)

    assert mol == benchmol


@pytest.mark.parametrize("dtype", ["json", "psi4"])
def test_to_from_file_complex(tmp_path, dtype, Molecule, water_dimer_minima_data):
    water_dimer_minima = Molecule.from_data(**water_dimer_minima_data)

    p = tmp_path / ("water." + dtype)
    water_dimer_minima.to_file(p)

    mol = Molecule.from_file(p)
    assert mol == water_dimer_minima


@pytest.mark.parametrize(
    "dtype, filext", [("json", "json"), ("xyz+", "xyz"), pytest.param("msgpack", "msgpack", marks=using_msgpack)]
)
def test_to_from_file_charge_spin(tmp_path, dtype, filext, Molecule):
    benchmol = Molecule.from_data(
        """
    1 2
    O 0 0 0
    H 0 1.5 0
    H 0 0 1.5
    """
    )

    p = tmp_path / ("water." + filext)
    benchmol.to_file(p, dtype=dtype)

    mol = Molecule.from_file(p, dtype=dtype)

    assert mol.molecular_charge == 1
    assert mol.molecular_multiplicity == 2
    assert mol.fragment_charges[0] == 1
    assert mol.fragment_multiplicities[0] == 2
    assert mol == benchmol


def test_from_data_kwargs(Molecule):
    mol = Molecule.from_data(
        """
        O 0 0 0
        H 0 1.5 0
        H 0 0 1.5
        """,
        molecular_charge=1,
        molecular_multiplicity=2,
        fragment_charges=[1],
        fragment_multiplicities=[2],
    )
    assert mol.molecular_charge == 1
    assert mol.molecular_multiplicity == 2
    assert mol.fragment_charges[0] == 1
    assert mol.fragment_multiplicities[0] == 2

    mol = Molecule.from_data(
        """
            O 0 0 0
            H 0 1.5 0
            H 0 0 1.5
            """,
        molecular_charge=1,
        molecular_multiplicity=2,
    )
    assert mol.molecular_charge == 1
    assert mol.molecular_multiplicity == 2
    assert mol.fragment_charges[0] == 1
    assert mol.fragment_multiplicities[0] == 2

    with pytest.raises(qcel.ValidationError) as e:
        mol = Molecule.from_data(
            """
            O 0 0 0
            H 0 1.5 0
            H 0 0 1.5
            """,
            molecular_charge=1,
            molecular_multiplicity=2,
            fragment_charges=[2],
        )
    assert "Inconsistent or unspecified chg/mult" in str(e.value)


def test_water_orient(Molecule):
    # These are identical molecules, should find the correct results
    mol = Molecule.from_data(
        """
        O  -1.551007  -0.114520   0.000000
        H  -1.934259   0.762503   0.000000
        H  -0.599677   0.040712   0.000000
        --
        O  -0.114520  -1.551007  10.000000
        H   0.762503  -1.934259  10.000000
        H   0.040712  -0.599677  10.000000
        """
    )

    frag_0 = mol.get_fragment(0, orient=True)
    frag_1 = mol.get_fragment(1, orient=True)

    # Make sure the fragments match
    assert frag_0.get_hash() == frag_1.get_hash()

    # Make sure the complexes match
    frag_0_1 = mol.get_fragment(0, 1, orient=True, group_fragments=True)
    frag_1_0 = mol.get_fragment(1, 0, orient=True, group_fragments=True)
    assert frag_0_1.get_hash() == frag_1_0.get_hash()

    # Fragments not reordered, should be different molecules.
    frag_0_1 = mol.get_fragment(0, 1, orient=True, group_fragments=False)
    frag_1_0 = mol.get_fragment(1, 0, orient=True, group_fragments=False)
    assert frag_0_1.get_hash() != frag_1_0.get_hash()

    # These are identical molecules, but should be different with ghost
    mol = Molecule.from_data(
        """
        O  -1.551007  -0.114520   0.000000
        H  -1.934259   0.762503   0.000000
        H  -0.599677   0.040712   0.000000
        --
        O  -11.551007  -0.114520   0.000000
        H  -11.934259   0.762503   0.000000
        H  -10.599677   0.040712   0.000000
        """,
        dtype="psi4",
        orient=True,
    )

    frag_0 = mol.get_fragment(0, orient=True)
    frag_1 = mol.get_fragment(1, orient=True)

    # Make sure the fragments match
    assert frag_0.molecular_multiplicity == 1
    assert frag_0.get_hash() == frag_1.get_hash()

    # Make sure the complexes match
    frag_0_1 = mol.get_fragment(0, 1, orient=True)
    frag_1_0 = mol.get_fragment(1, 0, orient=True)

    # Ghost fragments should prevent overlap
    assert frag_0_1.molecular_multiplicity == 1
    assert frag_0_1.get_hash() != frag_1_0.get_hash()


def test_molecule_errors_extra(Molecule, water_dimer_minima_data):
    water_dimer_minima = Molecule.from_data(**water_dimer_minima_data)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        data = water_dimer_minima.dict(exclude_unset=True)
    data["whatever"] = 5
    with pytest.raises(Exception):
        Molecule(**data, validate=False)


def test_molecule_errors_connectivity(Molecule, water_molecule_data):
    water_molecule = Molecule.from_data(water_molecule_data)

    data = water_molecule.model_dump()
    data["connectivity"] = [(-1, 5, 5)]
    with pytest.raises(Exception):
        Molecule(**data)


def test_molecule_errors_shape(Molecule, water_molecule_data):
    water_molecule = Molecule.from_data(water_molecule_data)

    data = water_molecule.model_dump()
    data["geometry"] = list(range(8))
    with pytest.raises(Exception):
        Molecule(**data)


def test_molecule_json_serialization(Molecule, water_dimer_minima_data):
    water_dimer_minima = Molecule.from_data(**water_dimer_minima_data)

    assert isinstance(water_dimer_minima.model_dump_json(), str)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert isinstance(water_dimer_minima.dict(encoding="json")["geometry"], list)

    assert water_dimer_minima == Molecule.from_data(water_dimer_minima.model_dump_json(), dtype="json")


@pytest.mark.parametrize("encoding", serialize_extensions)
def test_molecule_serialization(encoding, Molecule, water_dimer_minima_data):
    water_dimer_minima = Molecule.from_data(**water_dimer_minima_data)

    blob = water_dimer_minima.serialize(encoding)
    assert water_dimer_minima == Molecule.parse_raw(blob, encoding=encoding)


def test_charged_fragment(Molecule):
    mol = Molecule(
        symbols=["Li", "Li"],
        geometry=[0, 0, 0, 0, 0, 5],
        fragment_charges=[0.0, 0.0],
        fragment_multiplicities=[2, 2],
        fragments=[[0], [1]],
    )
    assert mol.molecular_multiplicity == 3
    assert mol.molecular_charge == 0
    f1 = mol.get_fragment(0)
    assert f1.molecular_multiplicity == 2
    assert f1.fragment_multiplicities == [2]
    assert pytest.approx(f1.molecular_charge) == 0
    assert pytest.approx(f1.fragment_charges) == [0]


# someday, when we can have noncontig frags
#    mol = Molecule(**{
#        'fragments': [[1, 5, 6], [0], [2, 3, 4]],
#        'symbols': ["he", "o", "o", "h", "h", "h", "h"],
#        # same geom as test_water_orient but with He at origin
#        'geometry': [
#        0.0, 0.0, 0.0,
#        -1.551007,  -0.114520,   0.000000,
#        -0.114520,  -1.551007,  10.000000,
#         0.762503,  -1.934259,  10.000000,
#         0.040712,  -0.599677,  10.000000,
#        -1.934259,   0.762503,   0.000000,
#        -0.599677,   0.040712,   0.000000],
#    })


@pytest.mark.parametrize("group_fragments, orient", [(True, True), (False, False)])  # original  # Psi4-like
def test_get_fragment(group_fragments, orient, Molecule):
    mol = Molecule(
        **{
            "fragments": [[0], [1, 2, 3], [4, 5, 6]],
            "symbols": ["he", "o", "h", "h", "o", "h", "h"],
            # same geom as test_water_orient but with He at origin
            "geometry": np.array(
                [
                    [0.0, 0.0, 0.0],
                    [-1.551007, -0.114520, 0.000000],
                    [-1.934259, 0.762503, 0.000000],
                    [-0.599677, 0.040712, 0.000000],
                    [-0.114520, -1.551007, 10.000000],
                    [0.762503, -1.934259, 10.000000],
                    [0.040712, -0.599677, 10.000000],
                ]
            )
            / qcel.constants.bohr2angstroms,
        }
    )

    assert mol.nelectrons() == 22
    assert compare_values(32.25894779318589, mol.nuclear_repulsion_energy(), atol=1.0e-5)
    assert mol.symbols[0] == "He"

    monomers_nelectrons = [2, 10, 10]
    monomers_nre = [0.0, 9.163830150548483, 9.163830150548483]
    monomers = [mol.get_fragment(ifr, group_fragments=group_fragments, orient=orient) for ifr in range(3)]
    for fr in range(3):
        assert monomers[fr].nelectrons() == monomers_nelectrons[fr]
        assert compare_values(monomers[fr].nuclear_repulsion_energy(), monomers_nre[fr], "monomer nre", atol=1.0e-5)

    idimers = [(0, 1), (0, 2), (1, 2), (1, 0), (2, 0), (2, 1)]
    dimers_nelectrons = [12, 12, 20, 12, 12, 20]
    dimers_nre = [16.8777971, 10.2097206, 23.4990904, 16.8777971, 10.2097206, 23.4990904]
    dimers = [mol.get_fragment(rl, group_fragments=group_fragments, orient=orient) for rl in idimers]
    for ifr in range(len(idimers)):
        # print('dd', ifr, idimers[ifr], dimers[ifr].nuclear_repulsion_energy(), dimers[ifr].get_hash())
        assert dimers[ifr].nelectrons() == dimers_nelectrons[ifr], "dimer nelec"
        assert compare_values(dimers[ifr].nuclear_repulsion_energy(), dimers_nre[ifr], "dimer nre", atol=1.0e-5)
    if group_fragments and orient:
        assert dimers[0].get_hash() != dimers[3].get_hash()  # atoms out of order
        assert dimers[1].get_hash() != dimers[4].get_hash()  # atoms out of order
        assert dimers[2].get_hash() == dimers[5].get_hash()
    elif not group_fragments and not orient:
        assert dimers[0].get_hash() == dimers[3].get_hash()
        assert dimers[1].get_hash() == dimers[4].get_hash()
        assert dimers[2].get_hash() == dimers[5].get_hash()
    else:
        assert 0

    ghdimers_nelectrons = [2, 2, 10, 10, 10, 10]
    ghdimers_nre = [0.0, 0.0, 9.163830150548483, 9.163830150548483, 9.163830150548483, 9.163830150548483]
    ghdimers = [mol.get_fragment(rl, gh, group_fragments=group_fragments, orient=orient) for rl, gh in idimers]
    for ifr in range(len(idimers)):
        # print('gh', ifr, idimers[ifr], ghdimers[ifr].nuclear_repulsion_energy(), ghdimers[ifr].get_hash())
        assert ghdimers[ifr].nelectrons() == ghdimers_nelectrons[ifr], "gh dimer nelec"
        assert compare_values(ghdimers[ifr].nuclear_repulsion_energy(), ghdimers_nre[ifr], "gh dimer nre", atol=1.0e-5)

    if group_fragments and orient:
        assert ghdimers[0].get_hash() != ghdimers[3].get_hash()  # diff atoms ghosted
        assert ghdimers[1].get_hash() != ghdimers[4].get_hash()  # diff atoms ghosted
        assert ghdimers[2].get_hash() == ghdimers[5].get_hash()
    elif not group_fragments and not orient:
        assert ghdimers[0].get_hash() != ghdimers[3].get_hash()  # diff atoms ghosted
        assert ghdimers[1].get_hash() != ghdimers[4].get_hash()  # diff atoms ghosted
        assert ghdimers[2].get_hash() != ghdimers[5].get_hash()  # real pattern different
        assert not np.allclose(ghdimers[2].real, ghdimers[5].real)
    else:
        assert 0


def test_molecule_repeated_hashing(Molecule):
    mol = Molecule(
        **{
            "symbols": ["H", "O", "O", "H"],
            "geometry": [
                [1.7317, 1.2909, 1.037100000000001],
                [1.3156, -0.0074, -0.2807],
                [-1.3143, 0.0084, -0.2741],
                [-1.7241, -1.3079, 1.0277],
            ],
        }
    )

    h1 = mol.get_hash()
    assert mol.get_molecular_formula() == "H2O2"

    mol2 = Molecule(orient=False, **mol.model_dump())
    assert h1 == mol2.get_hash()

    mol3 = Molecule(orient=False, **mol2.model_dump())
    assert h1 == mol3.get_hash()


@pytest.mark.parametrize(
    "measure,result",
    [
        ([0, 1], 1.8086677572537304),
        ([0, 1, 2], 37.98890673587713),
        ([0, 1, 2, 3], 180.0),
        ([[0, 1, 2, 3]], [180.0]),
        ([[1, 3], [3, 1], [1, 2, 3]], [6.3282716, 6.3282716, 149.51606694803903]),
        ([[0, 1, 2, 3], [3, 2, 1, 0]], [180.0, 180.0]),
    ],
)
def test_measurements(measure, result, Molecule, water_dimer_minima_data):
    water_dimer_minima = Molecule.from_data(**water_dimer_minima_data)

    Molecule(
        **{
            "symbols": ["H", "O", "O", "H"],
            "geometry": [
                [1.7317, 1.2909, 1.0371],
                [1.3156, -0.0074, -0.2807],
                [-1.3143, 0.0084, -0.2741],
                [-1.7241, -1.3079, 1.0277],
            ],
        }
    )

    assert pytest.approx(water_dimer_minima.measure(measure)) == result


@pytest.mark.parametrize(
    "f1c,f1m,f2c,f2m,tc,tm",
    [
        (0, 2, 0, 2, 0.0, 3),
        (0, 4, 0, 2, 0.0, 5),
        (1, 1, 1, 1, 2.0, 1),
        (1, 1, 0, 2, 1.0, 2),
        (0, 2, 1, 1, 1.0, 2),
        (-1, 1, 1, 1, 0.0, 1),
    ],
)
def test_fragment_charge_configurations(f1c, f1m, f2c, f2m, tc, tm, Molecule):
    mol = Molecule.from_data(
        """
    {f1c} {f1m}
    Li 0 0 0
    --
    {f2c} {f2m}
    Li 0 0 5
    """.format(
            f1c=f1c, f1m=f1m, f2c=f2c, f2m=f2m
        )
    )

    assert pytest.approx(mol.molecular_charge) == tc
    assert mol.molecular_multiplicity == tm

    # Test fragment1
    assert pytest.approx(mol.get_fragment(0).molecular_charge) == f1c
    assert mol.get_fragment(0).molecular_multiplicity == f1m

    assert pytest.approx(mol.get_fragment(0, 1).molecular_charge) == f1c
    assert mol.get_fragment(0, 1).molecular_multiplicity == f1m

    # Test fragment2
    assert pytest.approx(mol.get_fragment(1).molecular_charge) == f2c
    assert mol.get_fragment(1).molecular_multiplicity == f2m

    assert pytest.approx(mol.get_fragment([1], 0).molecular_charge) == f2c
    assert mol.get_fragment(1, [0]).molecular_multiplicity == f2m


def test_nuclearrepulsionenergy_nelectrons(Molecule):
    mol = Molecule.from_data(
        """
    0 1
    --
    O          0.75119       -0.61395        0.00271
    H          1.70471       -0.34686        0.00009
    --
    1 1
    N         -2.77793        0.00179       -0.00054
    H         -2.10136        0.51768        0.60424
    H         -3.45559       -0.51904        0.60067
    H         -2.26004       -0.67356       -0.60592
    H         -3.29652        0.68076       -0.60124
    units ang
    """
    )

    assert compare_values(34.60370459, mol.nuclear_repulsion_energy(), "D", atol=1.0e-5)
    assert compare_values(4.275210518, mol.nuclear_repulsion_energy(ifr=0), "M1", atol=1.0e-5)
    assert compare_values(16.04859029, mol.nuclear_repulsion_energy(ifr=1), "M2", atol=1.0e-5)

    assert compare(20, mol.nelectrons(), "D")
    assert compare(10, mol.nelectrons(ifr=0), "M1")
    assert compare(10, mol.nelectrons(ifr=1), "M2")

    mol = mol.get_fragment([1], 0, group_fragments=False)
    # Notice the 0th/1st fragments change if default group_fragments=True.
    ifr0 = 0
    ifr1 = 1
    assert compare_values(16.04859029, mol.nuclear_repulsion_energy(), "D", atol=1.0e-5)
    assert compare_values(0.0, mol.nuclear_repulsion_energy(ifr=ifr0), "M1", atol=1.0e-5)
    assert compare_values(16.04859029, mol.nuclear_repulsion_energy(ifr=ifr1), "M2", atol=1.0e-5)

    assert compare(10, mol.nelectrons(), "D")
    assert compare(0, mol.nelectrons(ifr=ifr0), "M1")
    assert compare(10, mol.nelectrons(ifr=ifr1), "M2")


@using_nglview
def test_show(Molecule, water_dimer_minima_data):
    water_dimer_minima = Molecule.from_data(**water_dimer_minima_data)

    water_dimer_minima.show()


def test_molecule_connectivity(Molecule):
    data = {"geometry": np.random.rand(5, 3), "symbols": ["he"] * 5, "validate": False}
    Molecule(**data, connectivity=None)

    connectivity = [[n, n + 1, 1] for n in range(4)]
    Molecule(**data, connectivity=connectivity)

    connectivity[0][0] = -1
    with pytest.raises(ValueError):
        Molecule(**data, connectivity=connectivity)


def test_orient_nomasses(Molecule):
    """
    Masses must be auto generated on the fly
    """

    mol = Molecule(symbols=["He", "He"], geometry=[0, 0, -2, 0, 0, 2], orient=True, validated=True)

    assert mol.__dict__["masses_"] is None
    assert compare_values([[2, 0, 0], [-2, 0, 0]], mol.geometry)


@pytest.mark.parametrize(
    "mol_string, extra_keys",
    [
        ("He 0 0 0", None),
        ("He 0 0 0\n--\nHe 0 0 5", {"fragments", "fragment_charges", "fragment_multiplicities"}),
        ("He 0 0 0\n--\n@He 0 0 5", {"fragments", "fragment_charges", "fragment_multiplicities", "real"}),
        ("He4 0 0 0", {"atom_labels"}),
        ("He@3.14 0 0 0", {"masses", "mass_numbers"}),
    ],
)
def test_sparse_molecule_fields(mol_string, extra_keys, Molecule):
    expected_keys = {
        "schema_name",
        "schema_version",
        "validated",
        "symbols",
        "geometry",
        "name",
        "molecular_charge",
        "molecular_multiplicity",
        "fix_com",
        "fix_orientation",
        "provenance",
        "extras",
    }
    mol = Molecule.from_data(mol_string)

    if extra_keys is not None:
        expected_keys |= extra_keys

    diff_keys = mol.model_dump().keys() ^ expected_keys
    assert len(diff_keys) == 0, f"Diff Keys {diff_keys}"


def test_sparse_molecule_connectivity(Molecule):
    """
    A bit of a weird test, but because we set connectivity it should carry through.
    """
    mol = Molecule(symbols=["He", "He"], geometry=[0, 0, -2, 0, 0, 2], connectivity=None)
    assert "connectivity" in mol.model_dump()
    assert mol.model_dump()["connectivity"] is None

    mol = Molecule(symbols=["He", "He"], geometry=[0, 0, -2, 0, 0, 2])
    assert "connectivity" not in mol.model_dump()


def test_bad_isotope_spec(Molecule):
    with pytest.raises(NotAnElementError):
        Molecule(symbols=["He3"], geometry=[0, 0, 0])


def test_good_isotope_spec(Molecule):
    assert compare_values(
        [3.01602932], Molecule(symbols=["He"], mass_numbers=[3], geometry=[0, 0, 0]).masses, "nonstd mass"
    )


def test_nonphysical_spec(Molecule):
    mol = Molecule(symbols=["He"], masses=[100], geometry=[0, 0, 0], nonphysical=True)
    assert compare_values([100.0], mol.masses, "nonphysical mass")

    print(mol.to_string(dtype="psi4"))


def test_extras(Molecule):
    mol = Molecule(symbols=["He"], geometry=[0, 0, 0])
    assert mol.extras is not None

    mol = Molecule(symbols=["He"], geometry=[0, 0, 0], extras={"foo": "bar"})
    assert mol.extras["foo"] == "bar"


_one_helium_mass = 4.00260325413


@pytest.mark.parametrize(
    "mol_string,args,formula,formula_dict,molecular_weight,nelec, nre",
    [
        ("He 0 0 0", {}, "He", {"He": 1}, _one_helium_mass, 2, 0.0),
        ("He 0 0 0\n--\nHe 0 0 5", {}, "He2", {"He": 2}, 2 * _one_helium_mass, 4, 0.4233417684),
        ("He 0 0 0\n--\n@He 0 0 5", {}, "He2", {"He": 1}, _one_helium_mass, 2, 0.0),
        ("He 0 0 0\n--\n@He 0 0 5", {"ifr": 0}, "He2", {"He": 1}, _one_helium_mass, 2, 0.0),
        ("He 0 0 0\n--\n@He 0 0 5", {"ifr": 1}, "He2", {}, 0.0, 0, 0.0),
        ("He 0 0 0\n--\n@He 0 0 5", {"real_only": False}, "He2", {"He": 2}, 2 * _one_helium_mass, 4, 0.4233417684),
        ("He 0 0 0\n--\n@He 0 0 5", {"real_only": False, "ifr": 0}, "He2", {"He": 1}, _one_helium_mass, 2, 0.0),
        ("He 0 0 0\n--\n@He 0 0 5", {"real_only": False, "ifr": 1}, "He2", {"He": 1}, _one_helium_mass, 2, 0.0),
        ("4He 0 0 0", {}, "He", {"He": 1}, _one_helium_mass, 2, 0.0),
        ("5He4 0 0 0", {}, "He", {"He": 1}, 5.012057, 2, 0.0),  # suffix-4 is label
        ("He@3.14 0 0 0", {}, "He", {"He": 1}, 3.14, 2, 0.0),
    ],
)
def test_molecular_weight(mol_string, args, formula, formula_dict, molecular_weight, nelec, nre, Molecule):
    mol = Molecule.from_data(mol_string)

    assert (ret := mol.molecular_weight(**args)) == molecular_weight, f"molecular_weight: {ret} != {molecular_weight}"
    assert (ret := mol.nelectrons(**args)) == nelec, f"nelectrons: {ret} != {nelec}"
    assert (abs(ret := mol.nuclear_repulsion_energy(**args)) - nre) < 1.0e-5, f"nre: {ret} != {nre}"
    assert (ret := mol.element_composition(**args)) == formula_dict, f"element_composition: {ret} != {formula_dict}"
    assert (ret := mol.get_molecular_formula()) == formula, f"get_molecular_formula: {ret} != {formula}"
