import copy

import numpy as np
import pytest

import qcelemental
from qcelemental.models import Molecule
from qcelemental.molparse.from_string import _filter_xyz
from qcelemental.testing import compare, compare_molrecs, compare_recursive, compare_values, tnm

_arrays_prov_stamp = {"creator": "QCElemental", "version": "1.0", "routine": "qcelemental.molparse.from_arrays"}
_string_prov_stamp = {"creator": "QCElemental", "version": "1.0", "routine": "qcelemental.molparse.from_string"}

_trans_molrec_to_model = {
    "geometry": "geom",
    "mass_numbers": "elea",
    "atomic_numbers": "elez",
    "symbols": "elem",
    "masses": "mass",
    "real": "real",
    "atom_labels": "elbl",
    "fix_com": "fix_com",
    "fix_orientation": "fix_orientation",
    "fragment_charges": "fragment_charges",
    "fragment_multiplicities": "fragment_multiplicities",
    "molecular_charge": "molecular_charge",
    "molecular_multiplicity": "molecular_multiplicity",
}


def _check_eq_molrec_minimal_model(keepers, model, molrec=None):
    usual_stored = set(
        ["geometry", "symbols", "molecular_charge", "molecular_multiplicity", "fix_com", "fix_orientation"]
    )
    usual_filtered = set(
        [
            "masses",
            "mass_numbers",
            "real",
            "atomic_numbers",
            "atom_labels",
            "connectivity",
            "fragments",
            "fragment_charges",
            "fragment_multiplicities",
        ]
    )

    for field in usual_filtered.difference(keepers):
        assert field not in model, f"Field '{field}' strangely present: {model[field]}"

    for field in usual_stored.union(keepers):
        assert field in model, f"Field '{field}' strangely absent"
        if molrec and field not in ["geometry", "fragments"]:
            skf = model[field]
            mrf = molrec[_trans_molrec_to_model[field]]
            if field in ["molecular_charge", "fragment_charges", "geometry", "masses"]:
                assert compare_values(mrf, skf, field, atol=1.0e-6)  # MASS_NOISE
            else:
                assert compare(mrf, skf, field)  # MASS_NOISE


subject1 = """O 0 0   0
no_com

H 1 ,, 0 \t  0 # stuff-n-nonsense"""

ans1 = {
    "geom": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    "elbl": ["O", "H"],
    "fix_com": True,
    "fragment_separators": [],
    "fragment_charges": [None],
    "fragment_multiplicities": [None],
    "fragment_files": [],
    "geom_hints": [],
    "hint_types": [],
}

fullans1a = {
    "geom": np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
    "elea": np.array([16, 1]),
    "elez": np.array([8, 1]),
    "elem": np.array(["O", "H"]),
    "mass": np.array([15.99491462, 1.00782503]),
    "real": np.array([True, True]),
    "elbl": np.array(["", ""]),
    "units": "Angstrom",
    "fix_com": True,
    "fix_orientation": False,
    "fragment_separators": [],
    "fragment_charges": [0.0],
    "fragment_multiplicities": [2],
    "molecular_charge": 0.0,
    "molecular_multiplicity": 2,
}
fullans1c = copy.deepcopy(fullans1a)
fullans1c.update(
    {"fragment_charges": [1.0], "fragment_multiplicities": [1], "molecular_charge": 1.0, "molecular_multiplicity": 1}
)


def test_psi4_qm_1a():
    subject = subject1
    fullans = copy.deepcopy(fullans1a)
    fullans["provenance"] = _string_prov_stamp

    final, intermed = qcelemental.molparse.from_string(subject, return_processed=True)
    assert compare_recursive(ans1, intermed, atol=1.0e-4)
    assert compare_molrecs(fullans, final["qm"], tnm() + ": full")

    kmol = Molecule.from_data(subject)
    _check_eq_molrec_minimal_model([], kmol.dict(), fullans)


def test_psi4_qm_1ab():
    subject = subject1
    ans = copy.deepcopy(ans1)
    ans["fix_orientation"] = False
    ans["fix_com"] = False
    fullans = copy.deepcopy(fullans1a)
    fullans["fix_com"] = False
    fullans["provenance"] = _string_prov_stamp

    final, intermed = qcelemental.molparse.from_string(
        subject, return_processed=True, fix_orientation=False, fix_com=False
    )
    assert compare_recursive(ans, intermed, tnm() + ": intermediate")
    assert compare_molrecs(fullans, final["qm"], tnm() + ": full")


def test_psi4_qm_1b():
    subject = "\n" + "\t" + subject1 + "\n\n"
    fullans = copy.deepcopy(fullans1a)
    fullans["provenance"] = _string_prov_stamp

    final, intermed = qcelemental.molparse.from_string(subject, return_processed=True)
    assert compare_recursive(ans1, intermed, tnm() + ": intermediate")
    assert compare_molrecs(fullans, final["qm"], tnm() + ": full")


def test_psi4_qm_1c():
    subject = "1 1\n  -- \n" + subject1
    ans = copy.deepcopy(ans1)
    ans.update({"molecular_charge": 1.0, "molecular_multiplicity": 1})
    fullans = copy.deepcopy(fullans1c)
    fullans["provenance"] = _string_prov_stamp

    final, intermed = qcelemental.molparse.from_string(subject, return_processed=True)
    assert compare_recursive(ans, intermed, tnm() + ": intermediate")
    assert compare_molrecs(fullans, final["qm"], tnm() + ": full")

    kmol = Molecule.from_data(subject)
    _check_eq_molrec_minimal_model([], kmol.dict(), fullans)


def test_psi4_qm_1d():
    subject = subject1 + "\n1 1"
    ans = copy.deepcopy(ans1)
    ans.update({"fragment_charges": [1.0], "fragment_multiplicities": [1]})
    fullans = copy.deepcopy(fullans1c)
    fullans["provenance"] = _string_prov_stamp

    final, intermed = qcelemental.molparse.from_string(subject, return_processed=True)
    assert compare_recursive(ans, intermed, tnm() + ": intermediate")
    assert compare_molrecs(fullans, final["qm"], tnm() + ": full")


def test_psi4_qm_1e():
    """duplicate com"""
    subject = subject1 + "\n  nocom"

    with pytest.raises(qcelemental.MoleculeFormatError):
        qcelemental.molparse.from_string(subject, return_processed=True)


def test_psi4_qm_1f():
    qcelemental.molparse.from_arrays(
        geom=np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
        elez=np.array([8, 1]),
        units="Angstrom",
        fix_com=True,
        fix_orientation=False,
    )


def test_psi4_qm_iutautoobig_error_1g():
    with pytest.raises(qcelemental.ValidationError) as e:
        qcelemental.molparse.from_arrays(
            geom=np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
            elez=np.array([8, 1]),
            input_units_to_au=1.1 / 0.52917721067,
            units="Angstrom",
            fix_com=True,
            fix_orientation=False,
        )

    assert "No big perturbations to physical constants" in str(e.value)


def test_psi4_qm_iutau_1h():
    fullans = copy.deepcopy(fullans1a)
    iutau = 1.01 / 0.52917721067
    fullans["input_units_to_au"] = iutau
    fullans["provenance"] = _arrays_prov_stamp

    final = qcelemental.molparse.from_arrays(
        geom=np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
        elez=np.array([8, 1]),
        input_units_to_au=iutau,
        units="Angstrom",
        fix_com=True,
        fix_orientation=False,
    )

    assert compare_molrecs(fullans, final, tnm() + ": full")

    smol = qcelemental.molparse.to_string(final, dtype="xyz", units="Bohr")
    rsmol = """2 au
0 2 HO
O                     0.000000000000     0.000000000000     0.000000000000
H                     1.908623386712     0.000000000000     0.000000000000
"""
    assert compare(rsmol, smol, tnm() + ": str")


def test_psi4_qm_iutau_1i():
    fullans = copy.deepcopy(fullans1a)
    iutau = 1.01 / 0.52917721067
    fullans["input_units_to_au"] = iutau
    fullans["fix_symmetry"] = "cs"
    fullans["provenance"] = _arrays_prov_stamp

    final = qcelemental.molparse.from_arrays(
        geom=np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
        elez=np.array([8, 1]),
        input_units_to_au=iutau,
        units="Angstrom",
        fix_symmetry="CS",
        fix_com=True,
        fix_orientation=False,
    )

    assert compare_molrecs(fullans, final, tnm() + ": full")

    kmol = qcelemental.molparse.to_schema(final, dtype=1, units="Bohr")
    schema14_1_iutau = {
        "schema_name": "qc_schema_input",
        "schema_version": 1,
        "molecule": {
            "validated": True,
            "geometry": [0.0, 0.0, 0.0, 1.908623386712, 0.0, 0.0],
            "symbols": ["O", "H"],
            "atomic_numbers": [8, 1],
            "mass_numbers": [16, 1],
            "atom_labels": ["", ""],
            "fragments": [[0, 1]],
            "fragment_charges": [0.0],
            "fragment_multiplicities": [2],
            "masses": [15.99491462, 1.00782503],
            "name": "HO",
            "fix_com": True,
            "fix_orientation": False,
            "fix_symmetry": "cs",
            "molecular_charge": 0.0,
            "molecular_multiplicity": 2,
            "real": [True, True],
            "provenance": _arrays_prov_stamp,
        },
    }

    assert compare_molrecs(schema14_1_iutau["molecule"], kmol["molecule"], tnm() + ": sch", atol=1.0e-8)


subject2 = [
    """
6Li 0.0 0.0 0.0
  units  a.u.
H_specIAL@2.014101  100 0 0""",
    """@Ne 2 4 6""",
    """h .0,1,2
Gh(he3) 0 1 3
noreorient""",
]

ans2 = {
    "geom": [0.0, 0.0, 0.0, 100.0, 0.0, 0.0, 2.0, 4.0, 6.0, 0.0, 1.0, 2.0, 0.0, 1.0, 3.0],
    "elbl": ["6Li", "H_specIAL@2.014101", "@Ne", "h", "Gh(he3)"],
    "units": "Bohr",
    "fix_orientation": True,
    "fragment_separators": [2, 3],
    "fragment_charges": [None, None, None],
    "fragment_multiplicities": [None, None, None],
    "fragment_files": [],
    "geom_hints": [],
    "hint_types": [],
}

fullans2 = {
    "geom": np.array([0.0, 0.0, 0.0, 100.0, 0.0, 0.0, 2.0, 4.0, 6.0, 0.0, 1.0, 2.0, 0.0, 1.0, 3.0]),
    "elea": np.array([6, 2, 20, 1, 4]),
    "elez": np.array([3, 1, 10, 1, 2]),
    "elem": np.array(["Li", "H", "Ne", "H", "He"]),
    "mass": np.array([6.015122794, 2.014101, 19.99244017542, 1.00782503, 4.00260325415]),
    "real": np.array([True, True, False, True, False]),
    "elbl": np.array(["", "_special", "", "", "3"]),
    "units": "Bohr",
    "fix_com": False,
    "fix_orientation": True,
    "fragment_separators": [2, 3],
}
fullans2_unnp = copy.deepcopy(fullans2)
fullans2_unnp.update(
    {
        "geom": [0.0, 0.0, 0.0, 100.0, 0.0, 0.0, 2.0, 4.0, 6.0, 0.0, 1.0, 2.0, 0.0, 1.0, 3.0],
        "elea": [6, 2, 20, 1, 4],
        "elez": [3, 1, 10, 1, 2],
        "elem": ["Li", "H", "Ne", "H", "He"],
        "mass": [6.015122794, 2.014101, 19.99244017542, 1.00782503, 4.00260325415],
        "real": [True, True, False, True, False],
        "elbl": ["", "_special", "", "", "3"],
    }
)


def test_psi4_qm_2a():
    subject = "\n--\n".join(subject2)
    fullans = copy.deepcopy(fullans2)
    fullans_unnp = copy.deepcopy(fullans2_unnp)
    ud = {
        "molecular_charge": 0.0,
        "molecular_multiplicity": 2,
        "fragment_charges": [0.0, 0.0, 0.0],
        "fragment_multiplicities": [1, 1, 2],
    }
    fullans.update(ud)
    fullans_unnp.update(ud)
    fullans["provenance"] = _string_prov_stamp
    fullans_unnp["provenance"] = _string_prov_stamp

    final, intermed = qcelemental.molparse.from_string(subject, return_processed=True)
    assert compare_recursive(ans2, intermed, tnm() + ": intermediate")
    final_unnp = qcelemental.util.unnp(final["qm"])
    assert compare_molrecs(fullans_unnp, final_unnp, tnm() + ": full unnp")
    assert compare_molrecs(fullans, final["qm"], tnm() + ": full")

    kmol = Molecule.from_data(subject)
    _check_eq_molrec_minimal_model(
        ["fragments", "fragment_charges", "fragment_multiplicities", "mass_numbers", "masses", "atom_labels", "real"],
        kmol.dict(),
        fullans,
    )


def test_psi4_qm_2b():
    subject = copy.deepcopy(subject2)
    subject.insert(0, "1 3")
    subject = "\n--\n".join(subject)
    ans = copy.deepcopy(ans2)
    ans.update({"molecular_charge": 1.0, "molecular_multiplicity": 3})
    fullans = copy.deepcopy(fullans2)
    fullans.update(
        {
            "molecular_charge": 1.0,
            "molecular_multiplicity": 3,
            "fragment_charges": [1.0, 0.0, 0.0],
            "fragment_multiplicities": [2, 1, 2],
        }
    )
    fullans["provenance"] = _string_prov_stamp

    final, intermed = qcelemental.molparse.from_string(subject, return_processed=True)
    assert compare_recursive(ans, intermed, tnm() + ": intermediate")
    assert compare_molrecs(fullans, final["qm"], tnm() + ": full")


def test_psi4_qm_2c():
    """double overall chg/mult spec"""
    subject = copy.deepcopy(subject2)
    subject.insert(0, "1 3\n1 3")
    subject = "\n--\n".join(subject)

    with pytest.raises(qcelemental.MoleculeFormatError):
        qcelemental.molparse.from_string(subject, return_processed=True)


def test_psi4_qm_2d():
    """trailing comma"""
    subject = copy.deepcopy(subject2)
    subject.insert(0, "H 10,10,10,")
    subject = "\n--\n".join(subject)

    with pytest.raises(qcelemental.MoleculeFormatError):
        qcelemental.molparse.from_string(subject, return_processed=True)


# def test_psi4_qm_2e():
#    """empty fragment"""
#    subject = copy.deepcopy(subject2)
#    subject.insert(2, '\n')
#    subject = '\n--\n'.join(subject)
#
#    with pytest.raises(qcelemental.MoleculeFormatError):
#        final, intermed = qcelemental.molparse.from_string(subject, return_processed=True)


def test_psi4_qm_2f():
    """double frag chgmult"""
    subject = copy.deepcopy(subject2)
    subject[1] += "\n 1 2\n 5 6"
    subject = "\n--\n".join(subject)

    with pytest.raises(qcelemental.MoleculeFormatError):
        qcelemental.molparse.from_string(subject, return_processed=True)


def test_psi4_qm_2g():
    """illegal chars in nucleus"""
    subject = copy.deepcopy(subject2)
    subject[1] = """@Ne_{CN}_O 2 4 6"""
    subject = "\n--\n".join(subject)

    with pytest.raises(qcelemental.MoleculeFormatError):
        qcelemental.molparse.from_string(subject, return_processed=True)


def test_psi4_qm_3():
    """psi4/psi4#731"""
    subject = """0 1
Mg 0 0"""

    with pytest.raises(qcelemental.ValidationError):
        qcelemental.molparse.from_string(subject, return_processed=True)


subject5 = """
efp C6H6 -0.30448173 -2.24210052 -0.29383131 -0.642499 7.817407 -0.568147  # second to last equiv to 1.534222
--
efp C6H6 -0.60075437  1.36443336  0.78647823  3.137879 1.557344 -2.568550
"""

ans5 = {
    "fragment_files": ["C6H6", "C6H6"],
    "hint_types": ["xyzabc", "xyzabc"],
    "geom_hints": [
        [-0.30448173, -2.24210052, -0.29383131, -0.642499, 7.817407, -0.568147],
        [-0.60075437, 1.36443336, 0.78647823, 3.137879, 1.557344, -2.568550],
    ],
    "geom": [],
    "elbl": [],
    "fragment_charges": [None],
    "fragment_multiplicities": [None],
    "fragment_separators": [],
}

fullans5b = {"efp": {}}
fullans5b["efp"]["hint_types"] = ans5["hint_types"]
fullans5b["efp"]["geom_hints"] = ans5["geom_hints"]
fullans5b["efp"]["units"] = "Bohr"
fullans5b["efp"]["fix_com"] = True
fullans5b["efp"]["fix_orientation"] = True
fullans5b["efp"]["fix_symmetry"] = "c1"
fullans5b["efp"]["fragment_files"] = ["c6h6", "c6h6"]


def test_psi4_efp_5a():
    subject = subject5

    hintsans = [
        [(val / qcelemental.constants.bohr2angstroms if i < 3 else val) for i, val in enumerate(ans5["geom_hints"][0])],
        [(val / qcelemental.constants.bohr2angstroms if i < 3 else val) for i, val in enumerate(ans5["geom_hints"][1])],
    ]
    hintsans[0][4] = 1.534222
    fullans = copy.deepcopy(fullans5b)
    fullans["efp"]["units"] = "Angstrom"
    fullans["efp"]["provenance"] = _string_prov_stamp

    final, intermed = qcelemental.molparse.from_string(subject, return_processed=True)
    assert compare_recursive(ans5, intermed, tnm() + ": intermediate")
    assert compare_molrecs(fullans["efp"], final["efp"], tnm() + ": final efp")

    hintsstd = qcelemental.util.standardize_efp_angles_units("Angstrom", final["efp"]["geom_hints"])
    final["efp"]["geom_hints"] = hintsstd
    fullans["efp"]["geom_hints"] = hintsans
    assert compare_molrecs(fullans["efp"], final["efp"], tnm() + ": final efp standardized")


def test_psi4_efp_5b():
    subject = subject5 + "\nunits bohr"

    ans = copy.deepcopy(ans5)
    ans["units"] = "Bohr"
    fullans = copy.deepcopy(fullans5b)
    fullans["efp"]["provenance"] = _string_prov_stamp

    final, intermed = qcelemental.molparse.from_string(subject, return_processed=True)
    assert compare_recursive(ans, intermed, tnm() + ": intermediate")
    assert compare_molrecs(fullans["efp"], final["efp"], tnm() + ": final efp")


def test_psi4_efp_5c():
    """fix_orientation not mol kw"""
    subject = subject5 + "\nno_com\nfix_orientation\nsymmetry c1"

    with pytest.raises(qcelemental.MoleculeFormatError):
        qcelemental.molparse.from_string(subject, return_processed=True)


def test_psi4_efp_5d():
    subject = subject5 + "\nno_com\nno_reorient\nsymmetry c1\nunits a.u."

    ans = copy.deepcopy(ans5)
    ans["units"] = "Bohr"
    ans["fix_com"] = True
    ans["fix_orientation"] = True
    ans["fix_symmetry"] = "c1"
    fullans = copy.deepcopy(fullans5b)
    fullans["efp"]["provenance"] = _string_prov_stamp

    final, intermed = qcelemental.molparse.from_string(subject, return_processed=True)
    assert compare_recursive(ans, intermed, tnm() + ": intermediate")
    assert compare_molrecs(fullans["efp"], final["efp"], tnm() + ": final")


def test_psi4_efp_5e():
    """symmetry w/efp"""
    subject = subject5 + "symmetry cs\nunits a.u."

    with pytest.raises(qcelemental.ValidationError):
        qcelemental.molparse.from_string(subject, return_processed=True)


subject6 = """
    0 1
    O1    0         0     0.118720
    h2   -0.753299, 0.0, -0.474880

    H3    0.753299, 0.0, -0.474880

    --
    efp h2O -2.12417561  1.22597097 -0.95332054 -2.902133 -4.5481863 -1.953647  # second to last equiv to 1.734999
 --
efp ammoniA
     0.98792    1.87681    2.85174
units au
     1.68798    1.18856    3.09517

     1.45873    2.55904    2.27226

"""

ans6 = {
    "units": "Bohr",
    "geom": [0.0, 0.0, 0.118720, -0.753299, 0.0, -0.474880, 0.753299, 0.0, -0.474880],
    "elbl": ["O1", "h2", "H3"],
    "fragment_charges": [0.0],
    "fragment_multiplicities": [1],
    "fragment_separators": [],
    "fragment_files": ["h2O", "ammoniA"],
    "geom_hints": [
        [-2.12417561, 1.22597097, -0.95332054, -2.902133, -4.5481863, -1.953647],
        [0.98792, 1.87681, 2.85174, 1.68798, 1.18856, 3.09517, 1.45873, 2.55904, 2.27226],
    ],
    "hint_types": ["xyzabc", "points"],
}

fullans6 = {
    "qm": {
        "geom": np.array([0.0, 0.0, 0.118720, -0.753299, 0.0, -0.474880, 0.753299, 0.0, -0.474880]),
        "elea": np.array([16, 1, 1]),
        "elez": np.array([8, 1, 1]),
        "elem": np.array(["O", "H", "H"]),
        "mass": np.array([15.99491462, 1.00782503, 1.00782503]),
        "real": np.array([True, True, True]),
        "elbl": np.array(["1", "2", "3"]),
        "units": "Bohr",
        "fix_com": True,
        "fix_orientation": True,
        "fix_symmetry": "c1",
        "fragment_charges": [0.0],
        "fragment_multiplicities": [1],
        "fragment_separators": [],
        "molecular_charge": 0.0,
        "molecular_multiplicity": 1,
    },
    "efp": {
        "fragment_files": ["h2o", "ammonia"],
        "geom_hints": [
            [-2.12417561, 1.22597097, -0.95332054, -2.902133, -4.5481863, -1.953647],
            [0.98792, 1.87681, 2.85174, 1.68798, 1.18856, 3.09517, 1.45873, 2.55904, 2.27226],
        ],
        "hint_types": ["xyzabc", "points"],
        "units": "Bohr",
        "fix_com": True,
        "fix_orientation": True,
        "fix_symmetry": "c1",
    },
}


def test_psi4_qmefp_6a():
    subject = subject6
    fullans = copy.deepcopy(fullans6)
    fullans["qm"]["provenance"] = _string_prov_stamp
    fullans["efp"]["provenance"] = _string_prov_stamp

    final, intermed = qcelemental.molparse.from_string(subject, return_processed=True)
    assert compare_recursive(ans6, intermed, tnm() + ": intermediate")
    assert compare_molrecs(fullans["efp"], final["efp"], tnm() + ": full efp")
    assert compare_molrecs(fullans["qm"], final["qm"], tnm() + ": full qm")

    hintsstd = qcelemental.util.standardize_efp_angles_units("Bohr", final["efp"]["geom_hints"])
    final["efp"]["geom_hints"] = hintsstd
    fullans["efp"]["geom_hints"][0][4] = 1.734999
    assert compare_molrecs(fullans["efp"], final["efp"], tnm() + ": final efp standardized")


def test_psi4_qmefp_6b():
    subject = subject6.replace("au", "ang")

    ans = copy.deepcopy(ans6)
    ans["units"] = "Angstrom"

    fullans = copy.deepcopy(fullans6)
    fullans["qm"]["units"] = "Angstrom"
    fullans["efp"]["units"] = "Angstrom"
    fullans["qm"]["provenance"] = _string_prov_stamp
    fullans["efp"]["provenance"] = _string_prov_stamp

    final, intermed = qcelemental.molparse.from_string(subject, return_processed=True)
    assert compare_recursive(ans, intermed, tnm() + ": intermediate")
    assert compare_molrecs(fullans["efp"], final["efp"], tnm() + ": full efp")
    assert compare_molrecs(fullans["qm"], final["qm"], tnm() + ": full qm")


def test_psi4_qmefpformat_error_6c():
    """try to give chgmult to an efp"""

    subject = subject6.replace("    efp h2O", "0 1\n    efp h2O")

    with pytest.raises(qcelemental.MoleculeFormatError):
        qcelemental.molparse.from_string(subject, return_processed=True)


def test_qmefp_array_6d():
    fullans = copy.deepcopy(fullans6)
    fullans["qm"]["provenance"] = _arrays_prov_stamp
    fullans["efp"]["provenance"] = _arrays_prov_stamp

    final = qcelemental.molparse.from_input_arrays(
        units="Bohr",
        geom=[0.0, 0.0, 0.118720, -0.753299, 0.0, -0.474880, 0.753299, 0.0, -0.474880],
        elbl=["O1", "h2", "H3"],
        fragment_files=["h2O", "ammoniA"],
        geom_hints=[
            [-2.12417561, 1.22597097, -0.95332054, -2.902133, -4.5481863, -1.953647],
            [0.98792, 1.87681, 2.85174, 1.68798, 1.18856, 3.09517, 1.45873, 2.55904, 2.27226],
        ],
        hint_types=["xyzabc", "points"],
    )

    assert compare_molrecs(fullans["efp"], final["efp"], tnm() + ": full efp")
    assert compare_molrecs(fullans["qm"], final["qm"], tnm() + ": full qm")


def test_qmefp_badhint_error_6e():
    with pytest.raises(qcelemental.ValidationError) as e:
        qcelemental.molparse.from_input_arrays(
            units="Bohr",
            geom=[0.0, 0.0, 0.118720, -0.753299, 0.0, -0.474880, 0.753299, 0.0, -0.474880],
            elbl=["O1", "h2", "H3"],
            fragment_files=["h2O", "ammoniA"],
            geom_hints=[
                [-2.12417561, 1.22597097, -0.95332054, -2.902133, -4.5481863, -1.953647],
                [0.98792, 1.87681, 2.85174, 1.68798, 1.18856, 3.09517, 1.45873, 2.55904, 2.27226],
            ],
            hint_types=["xyzabc", "efp1"],
        )

    assert "hint_types not among 'xyzabc', 'points', 'rotmat'" in str(e.value)


def test_qmefp_badefpgeom_error_6f():
    with pytest.raises(qcelemental.ValidationError) as e:
        qcelemental.molparse.from_input_arrays(
            units="Bohr",
            geom=[0.0, 0.0, 0.118720, -0.753299, 0.0, -0.474880, 0.753299, 0.0, -0.474880],
            elbl=["O1", "h2", "H3"],
            fragment_files=["h2O", "ammoniA"],
            geom_hints=[
                [-2.12417561, None, -0.95332054, -2.902133, -4.5481863, -1.953647],
                [0.98792, 1.87681, 2.85174, 1.68798, 1.18856, 3.09517, 1.45873, 2.55904, 2.27226],
            ],
            hint_types=["xyzabc", "points"],
        )

    assert "Un float-able elements in geom_hints" in str(e.value)


def test_qmefp_badhintgeom_error_6g():
    with pytest.raises(qcelemental.ValidationError) as e:
        qcelemental.molparse.from_input_arrays(
            units="Bohr",
            geom=[0.0, 0.0, 0.118720, -0.753299, 0.0, -0.474880, 0.753299, 0.0, -0.474880],
            elbl=["O1", "h2", "H3"],
            fragment_files=["h2O", "ammoniA"],
            geom_hints=[
                [-2.12417561, 1.22597097, -0.95332054, -2.902133, -4.5481863, -1.953647],
                [0.98792, 1.87681, 2.85174, 1.68798, 1.18856, 3.09517, 1.45873, 2.55904, 2.27226],
            ],
            hint_types=["points", "xyzabc"],
        )

    assert "EFP hint type points not 9 elements" in str(e.value)


def test_qmefp_badfragfile_error_6h():
    with pytest.raises(qcelemental.ValidationError) as e:
        qcelemental.molparse.from_input_arrays(
            units="Bohr",
            geom=[0.0, 0.0, 0.118720, -0.753299, 0.0, -0.474880, 0.753299, 0.0, -0.474880],
            elbl=["O1", "h2", "H3"],
            fragment_files=["h2O", 5],
            geom_hints=[
                [-2.12417561, 1.22597097, -0.95332054, -2.902133, -4.5481863, -1.953647],
                [0.98792, 1.87681, 2.85174, 1.68798, 1.18856, 3.09517, 1.45873, 2.55904, 2.27226],
            ],
            hint_types=["xyzabc", "points"],
        )

    assert "fragment_files not strings" in str(e.value)


def test_qmefp_hintlen_error_6i():
    with pytest.raises(qcelemental.ValidationError) as e:
        qcelemental.molparse.from_input_arrays(
            units="Bohr",
            geom=[0.0, 0.0, 0.118720, -0.753299, 0.0, -0.474880, 0.753299, 0.0, -0.474880],
            elbl=["O1", "h2", "H3"],
            fragment_files=["h2O", "ammoniA"],
            geom_hints=[
                [-2.12417561, 1.22597097, -0.95332054, -2.902133, -4.5481863, -1.953647],
                [0.98792, 1.87681, 2.85174, 1.68798, 1.18856, 3.09517, 1.45873, 2.55904, 2.27226],
            ],
            hint_types=["xyzabc", "points", "points"],
        )

    assert "Missing or inconsistent length among efp quantities" in str(e.value)


def test_qmefp_fixcom_error_6j():
    with pytest.raises(qcelemental.ValidationError) as e:
        qcelemental.molparse.from_input_arrays(
            units="Bohr",
            fix_com=False,
            geom=[0.0, 0.0, 0.118720, -0.753299, 0.0, -0.474880, 0.753299, 0.0, -0.474880],
            elbl=["O1", "h2", "H3"],
            fragment_files=["h2O", "ammoniA"],
            geom_hints=[
                [-2.12417561, 1.22597097, -0.95332054, -2.902133, -4.5481863, -1.953647],
                [0.98792, 1.87681, 2.85174, 1.68798, 1.18856, 3.09517, 1.45873, 2.55904, 2.27226],
            ],
            hint_types=["xyzabc", "points"],
        )

    assert "Invalid fix_com (False) with extern (True)" in str(e.value)


def test_qmefp_fixori_error_6k():
    with pytest.raises(qcelemental.ValidationError) as e:
        qcelemental.molparse.from_input_arrays(
            units="Bohr",
            fix_orientation=False,
            geom=[0.0, 0.0, 0.118720, -0.753299, 0.0, -0.474880, 0.753299, 0.0, -0.474880],
            elbl=["O1", "h2", "H3"],
            fragment_files=["h2O", "ammoniA"],
            geom_hints=[
                [-2.12417561, 1.22597097, -0.95332054, -2.902133, -4.5481863, -1.953647],
                [0.98792, 1.87681, 2.85174, 1.68798, 1.18856, 3.09517, 1.45873, 2.55904, 2.27226],
            ],
            hint_types=["xyzabc", "points"],
        )

    assert "Invalid fix_orientation (False) with extern (True)" in str(e.value)


# QCEL@using_pylibefp
# QCELdef test_psi4_qmefp_6d():
# QCEL    subject = subject6
# QCEL
# QCEL    fullans = copy.deepcopy(fullans6)
# QCEL    fullans['efp']['geom'] = np.array([-2.22978429,  1.19270015, -0.99721732, -1.85344873,  1.5734809 ,
# QCEL        0.69660583, -0.71881655,  1.40649303, -1.90657336,  0.98792   ,
# QCEL        1.87681   ,  2.85174   ,  2.31084386,  0.57620385,  3.31175679,
# QCEL        1.87761143,  3.16604791,  1.75667803,  0.55253064,  2.78087794,
# QCEL        4.47837555])
# QCEL    fullans['efp']['elea'] = np.array([16, 1, 1, 14, 1, 1, 1])
# QCEL    fullans['efp']['elez'] = np.array([8, 1, 1, 7, 1, 1, 1])
# QCEL    fullans['efp']['elem'] = np.array(['O', 'H', 'H', 'N', 'H', 'H', 'H'])
# QCEL    fullans['efp']['mass'] = np.array([15.99491462, 1.00782503, 1.00782503, 14.00307400478, 1.00782503, 1.00782503, 1.00782503])
# QCEL    fullans['efp']['real'] = np.array([True, True, True, True, True, True, True])
# QCEL    fullans['efp']['elbl'] = np.array(['_a01o1', '_a02h2', '_a03h3', '_a01n1', '_a02h2', '_a03h3', '_a04h4'])
# QCEL    fullans['efp']['fragment_separators'] = [3]
# QCEL    fullans['efp']['fragment_charges'] = [0., 0.]
# QCEL    fullans['efp']['fragment_multiplicities'] = [1, 1]
# QCEL    fullans['efp']['molecular_charge'] = 0.
# QCEL    fullans['efp']['molecular_multiplicity'] = 1
# QCEL    fullans['efp']['hint_types'] = ['xyzabc', 'xyzabc']
# QCEL    fullans['efp']['geom_hints'][1] = [1.093116487139866, 1.9296501432128303, 2.9104336205167156, -1.1053108079381473, 2.0333070957565544, -1.488586877218809]
# QCEL
# QCEL    final, intermed = qcelemental.molparse.from_string(subject, return_processed=True)
# QCEL
# QCEL    import pylibefp
# QCEL    efpobj = pylibefp.from_dict(final['efp'])
# QCEL    efpfinal = efpobj.to_dict()
# QCEL    efpfinal = qcelemental.molparse.from_arrays(speclabel=False, domain='efp', **efpfinal)
# QCEL
# QCEL    assert compare_molrecs(fullans['qm'], final['qm'], tnm() + ': full qm')
# QCEL    assert compare_molrecs(fullans['efp'], efpfinal, tnm() + ': full efp')

subject7 = """\
5
   stuffs
6Li 0.0 0.0 0.0
H_specIAL@2.014101  100 0 0
@Ne 2 4 6
h .0,1,2
Gh(he3) 0 1 3
"""

ans7 = {
    "geom": [0.0, 0.0, 0.0, 100.0, 0.0, 0.0, 2.0, 4.0, 6.0, 0.0, 1.0, 2.0, 0.0, 1.0, 3.0],
    "elbl": ["6Li", "H_specIAL@2.014101", "@Ne", "h", "Gh(he3)"],
    "units": "Angstrom",
    "geom_hints": [],  # shouldn't be needed
}

fullans7 = {
    "geom": np.array([0.0, 0.0, 0.0, 100.0, 0.0, 0.0, 2.0, 4.0, 6.0, 0.0, 1.0, 2.0, 0.0, 1.0, 3.0]),
    "elea": np.array([6, 2, 20, 1, 4]),
    "elez": np.array([3, 1, 10, 1, 2]),
    "elem": np.array(["Li", "H", "Ne", "H", "He"]),
    "mass": np.array([6.015122794, 2.014101, 19.99244017542, 1.00782503, 4.00260325415]),
    "real": np.array([True, True, False, True, False]),
    "elbl": np.array(["", "_special", "", "", "3"]),
    "units": "Angstrom",
    "fix_com": False,
    "fix_orientation": False,
    "fragment_separators": [],
    "fragment_charges": [0.0],
    "fragment_multiplicities": [2],
    "molecular_charge": 0.0,
    "molecular_multiplicity": 2,
}


def test_xyzp_qm_7a():
    """XYZ doesn't fit into psi4 string"""
    subject = subject7

    with pytest.raises(qcelemental.MoleculeFormatError):
        qcelemental.molparse.from_string(subject, return_processed=True, dtype="psi4")


def test_xyzp_qm_7b():
    """XYZ doesn't fit into strict xyz string"""
    subject = subject7

    with pytest.raises(qcelemental.MoleculeFormatError):
        qcelemental.molparse.from_string(subject, return_processed=True, dtype="xyz")


def test_xyzp_qm_7c():
    subject = subject7
    fullans = copy.deepcopy(fullans7)
    fullans["provenance"] = _string_prov_stamp

    final, intermed = qcelemental.molparse.from_string(subject, return_processed=True, dtype="xyz+")
    assert compare_recursive(ans7, intermed, tnm() + ": intermediate")
    assert compare_molrecs(fullans, final["qm"], tnm() + ": full qm")


def test_xyzp_qm_7d():
    subject = subject7.replace("5", "5 au ")
    subject = subject.replace("stuff", "-1 3 slkdjfl2 32#$^& ")

    ans = copy.deepcopy(ans7)
    ans["units"] = "Bohr"
    ans["molecular_charge"] = -1.0
    ans["molecular_multiplicity"] = 3

    fullans = copy.deepcopy(fullans7)
    fullans["units"] = "Bohr"
    fullans["fragment_charges"] = [-1.0]
    fullans["fragment_multiplicities"] = [3]
    fullans["molecular_charge"] = -1.0
    fullans["molecular_multiplicity"] = 3
    fullans["provenance"] = _string_prov_stamp

    final, intermed = qcelemental.molparse.from_string(subject, return_processed=True, dtype="xyz+")
    assert compare_recursive(ans, intermed, tnm() + ": intermediate")
    assert compare_molrecs(fullans, final["qm"], tnm() + ": full qm")


def test_xyzp_qm_7e():
    subject = subject7.replace("5", "5 ang")
    fullans = copy.deepcopy(fullans7)
    fullans["provenance"] = _string_prov_stamp

    final, intermed = qcelemental.molparse.from_string(subject, return_processed=True, dtype="xyz+")
    assert compare_recursive(ans7, intermed, tnm() + ": intermediate")
    assert compare_molrecs(fullans, final["qm"], tnm() + ": full qm")


@pytest.mark.parametrize(
    "string,elbl,geom",
    (
        (
            """5
gdb 1	157.7118	157.70997	157.70699	0.	13.21	-0.3877	0.1171	0.5048	35.3641	0.044749	-40.47893	-40.476062	-40.475117	-40.498597	6.469	
C	-0.0126981359	 1.0858041578	 0.0080009958	-0.535689
H	 0.002150416	-0.0060313176	 0.0019761204	 0.133921
H	 1.0117308433	 1.4637511618	 0.0002765748	 0.133922
H	-0.540815069	 1.4475266138	-0.8766437152	 0.133923
H	-0.5238136345	 1.4379326443	 0.9063972942	 0.133923
1341.307	1341.3284	1341.365	1562.6731	1562.7453	3038.3205	3151.6034	3151.6788	3151.7078
C	C	
InChI=1S/CH4/h1H4	InChI=1S/CH4/h1H4""",
            ["C", "H", "H", "H", "H"],
            [
                -0.0126981359,
                1.0858041578,
                0.0080009958,
                0.002150416,
                -0.0060313176,
                0.0019761204,
                1.0117308433,
                1.4637511618,
                0.0002765748,
                -0.540815069,
                1.4475266138,
                -0.8766437152,
                -0.5238136345,
                1.4379326443,
                0.9063972942,
            ],
        ),
        (
            """20
        gdb 52625	3.48434	0.81389	0.77349	4.0931	85.49	-0.2471	0.0275	0.2746	1578.0163	0.16756	-403.158572	-403.147447	-403.146503	-403.196607	38.82
        C	 0.0219866132	 1.4617007325	 0.0778162941	-0.5003
        C	 0.0172170008	 0.0062570163	 0.0278221402	 0.060798
        C	 0.0082754818	-1.1968179556	-0.0219490036	 0.058561
        C	-0.0048543985	-2.6632285567	-0.0864436684	 0.086487
        C	-1.4625346892	-3.1775528493	-0.0739515886	-0.401401
        C	 0.724612823	-3.1404242978	-1.3630324305	-0.401392
        N	 0.7132087245	-3.1481618818	 1.1179206119	-0.295483
        C	 0.9247284349	-4.4479813471	 1.4397739679	 0.162556
        O	 0.5626675182	-5.4147891002	 0.7982523365	-0.336127
        H	 1.0431199948	 1.8581192318	 0.0749725881	 0.150933
        H	-0.4997253695	 1.8847111976	-0.7874860262	 0.154682
        H	-0.4780438734	 1.8320081388	 0.979321682	 0.150927
        H	-1.9940013133	-2.8054928472	-0.9536444371	 0.11928
        H	-1.4636215253	-4.2685570577	-0.0833553058	 0.154594
        H	-1.981857154	-2.824763824	 0.8204749775	 0.116725
        H	 1.749444309	-2.7614376976	-1.3787041159	 0.11672
        H	 0.7443784569	-4.2310758294	-1.3847506905	 0.154572
        H	 0.2030559276	-2.7681902401	-2.2485640369	 0.11929
        H	 1.0701803562	-2.4436691628	 1.7431541049	 0.251085
        H	 1.4874481325	-4.5389119798	 2.3916335805	 0.077494
        9.6558	74.9555	86.4308	130.4187	209.6956	213.4664	245.0078	277.3227	292.8775	293.3837	344.8375	364.9338	411.1024	508.5101	542.9334	570.3714	618.7374	820.0082	829.2715	936.286	986.4059	1028.4996	1033.1508	1051.8084	1054.6924	1076.9172	1172.1103	1179.8358	1222.2743	1289.1699	1387.6509	1408.9379	1414.4085	1431.5059	1476.1153	1477.0789	1478.6118	1481.2338	1494.3428	1511.202	1519.6039	1792.8486	2370.6229	2920.0767	3029.2922	3050.3711	3054.2985	3090.8073	3092.1476	3124.8526	3128.7105	3148.8786	3152.1387	3639.6323
        CC#CC(C)(C)NC=O	CC#CC(C)(C)NC=O
        InChI=1S/C7H11NO/c1-4-5-7(2,3)8-6-9/h6H,1-3H3,(H,8,9)	InChI=1S/C7H11NO/c1-4-5-7(2,3)8-6-9/h6H,1-3H3,(H,8,9)
        """,
            ["C", "C", "C", "C", "C", "C", "N", "C", "O", "H", "H", "H", "H", "H", "H", "H", "H", "H", "H", "H"],
            [
                0.0219866132,
                1.4617007325,
                0.0778162941,
                0.0172170008,
                0.0062570163,
                0.0278221402,
                0.0082754818,
                -1.1968179556,
                -0.0219490036,
                -0.0048543985,
                -2.6632285567,
                -0.0864436684,
                -1.4625346892,
                -3.1775528493,
                -0.0739515886,
                0.724612823,
                -3.1404242978,
                -1.3630324305,
                0.7132087245,
                -3.1481618818,
                1.1179206119,
                0.9247284349,
                -4.4479813471,
                1.4397739679,
                0.5626675182,
                -5.4147891002,
                0.7982523365,
                1.0431199948,
                1.8581192318,
                0.0749725881,
                -0.4997253695,
                1.8847111976,
                -0.7874860262,
                -0.4780438734,
                1.8320081388,
                0.979321682,
                -1.9940013133,
                -2.8054928472,
                -0.9536444371,
                -1.4636215253,
                -4.2685570577,
                -0.0833553058,
                -1.981857154,
                -2.824763824,
                0.8204749775,
                1.749444309,
                -2.7614376976,
                -1.3787041159,
                0.7443784569,
                -4.2310758294,
                -1.3847506905,
                0.2030559276,
                -2.7681902401,
                -2.2485640369,
                1.0701803562,
                -2.4436691628,
                1.7431541049,
                1.4874481325,
                -4.5389119798,
                2.3916335805,
            ],
        ),
        (
            """17
        gdb 107395	3.22974	1.27619	1.028	3.0629	69.23	-0.225	0.0454	0.2704	1155.6964	0.136978	-455.1439	-455.135732	-455.134788	-455.176691	31.549
        N	 0.0166534686	 1.2958609713	-0.1502735485	-0.580455
        C	-0.0682337319	-0.0600800983	-0.0096861238	 0.457461
        N	-1.1305816717	-0.7597824714	 0.1390863864	-0.408126
        C	-0.7009986293	-2.1439574926	 0.1108213578	 0.311595
        C	-1.4984924036	-3.0589993793	 1.002599169	-0.138474
        O	-2.8585678608	-3.0949107355	 0.6105794873	-0.417767
        C	-0.0617590849	-2.6053899658	-1.1855277252	-0.354308
        C	 0.8014538633	-2.0927793706	-0.0638522254	 0.002142
        O	 1.1296621	-0.7079465651	-0.1012706145	-0.24761
        H	 0.8924648972	 1.705797935	 0.136582908	 0.26608
        H	-0.8059467993	 1.7929265086	 0.153937696	 0.270518
        H	-1.1201801933	-4.0847553601	 0.9397586373	 0.099132
        H	-1.3951151651	-2.7241068393	 2.0475874391	 0.085667
        H	-3.1252104475	-2.1733539114	 0.5051511429	 0.295168
        H	-0.0217120799	-3.6759992339	-1.3586546368	 0.121153
        H	-0.1831965908	-1.9943417386	-2.0738763382	 0.124725
        H	 1.5549266689	-2.687136563	 0.4367361986	 0.113098
        70.6081	153.9333	180.5159	278.5491	345.0174	381.8663	413.9812	430.6681	448.7324	482.9536	553.6703	659.1349	739.6911	741.9888	783.4372	872.8339	876.0968	943.3731	959.9083	980.6717	997.5511	1046.3298	1069.8255	1090.0307	1128.0369	1148.246	1187.611	1267.6525	1296.0396	1370.3317	1379.5593	1424.1337	1443.8219	1468.4333	1503.1705	1614.9407	1733.6758	2971.4118	3070.4849	3127.1322	3195.3399	3220.8339	3586.8344	3701.0432	3786.2797
        NC1=NC2(CO)CC2O1	NC1=N[C@@]2(CO)C[C@H]2O1
        InChI=1S/C5H8N2O2/c6-4-7-5(2-8)1-3(5)9-4/h3,8H,1-2H2,(H2,6,7)	InChI=1S/C5H8N2O2/c6-4-7-5(2-8)1-3(5)9-4/h3,8H,1-2H2,(H2,6,7)/t3-,5-/m1/s1
        """,
            ["N", "C", "N", "C", "C", "O", "C", "C", "O", "H", "H", "H", "H", "H", "H", "H", "H"],
            [
                0.0166534686,
                1.2958609713,
                -0.1502735485,
                -0.0682337319,
                -0.0600800983,
                -0.0096861238,
                -1.1305816717,
                -0.7597824714,
                0.1390863864,
                -0.7009986293,
                -2.1439574926,
                0.1108213578,
                -1.4984924036,
                -3.0589993793,
                1.002599169,
                -2.8585678608,
                -3.0949107355,
                0.6105794873,
                -0.0617590849,
                -2.6053899658,
                -1.1855277252,
                0.8014538633,
                -2.0927793706,
                -0.0638522254,
                1.1296621,
                -0.7079465651,
                -0.1012706145,
                0.8924648972,
                1.705797935,
                0.136582908,
                -0.8059467993,
                1.7929265086,
                0.153937696,
                -1.1201801933,
                -4.0847553601,
                0.9397586373,
                -1.3951151651,
                -2.7241068393,
                2.0475874391,
                -3.1252104475,
                -2.1733539114,
                0.5051511429,
                -0.0217120799,
                -3.6759992339,
                -1.3586546368,
                -0.1831965908,
                -1.9943417386,
                -2.0738763382,
                1.5549266689,
                -2.687136563,
                0.4367361986,
            ],
        ),
    ),
)
def test_xyz_gdb_format(string, elbl, geom):
    unprocessed, processed = _filter_xyz(string, strict=True)

    assert not unprocessed
    assert processed["elbl"] == elbl
    assert processed["geom"] == geom


subject8 = """\
3
   stuffs
Li 0.0 0.0 0.0
1  100 0 0
Ne 2 4 6
h .0,1,2
 2 0 1 3
"""

ans8 = {
    "geom": [0.0, 0.0, 0.0, 100.0, 0.0, 0.0, 2.0, 4.0, 6.0, 0.0, 1.0, 2.0, 0.0, 1.0, 3.0],
    "elbl": ["Li", "1", "Ne", "h", "2"],
    "units": "Angstrom",
    "geom_hints": [],  # shouldn't be needed
}

fullans8 = {
    "geom": np.array([0.0, 0.0, 0.0, 100.0, 0.0, 0.0, 2.0, 4.0, 6.0, 0.0, 1.0, 2.0, 0.0, 1.0, 3.0]),
    "elea": np.array([7, 1, 20, 1, 4]),
    "elez": np.array([3, 1, 10, 1, 2]),
    "elem": np.array(["Li", "H", "Ne", "H", "He"]),
    "mass": np.array([7.016004548, 1.00782503, 19.99244017542, 1.00782503, 4.00260325415]),
    "real": np.array([True, True, True, True, True]),
    "elbl": np.array(["", "", "", "", ""]),
    "units": "Angstrom",
    "fix_com": False,
    "fix_orientation": False,
    "fragment_separators": [],
    "fragment_charges": [0.0],
    "fragment_multiplicities": [2],
    "molecular_charge": 0.0,
    "molecular_multiplicity": 2,
}


def test_xyzp_qm_8a():
    subject = subject8
    fullans = copy.deepcopy(fullans8)
    fullans["provenance"] = _string_prov_stamp

    final, intermed = qcelemental.molparse.from_string(subject, return_processed=True, dtype="xyz+")
    assert compare_recursive(ans8, intermed, tnm() + ": intermediate")
    assert compare_molrecs(fullans, final["qm"], tnm() + ": full qm", atol=1.0e-4)


fullans10qm = {
    "geom": np.array([0.0, 0.0, 0.0]),
    "elea": np.array([12]),
    "elez": np.array([6]),
    "elem": np.array(["C"]),
    "mass": np.array([12.0]),
    "real": np.array([True]),
    "elbl": np.array([""]),
    "units": "Angstrom",
    "fix_com": False,
    "fix_orientation": False,
    "fragment_separators": [],
    "fragment_charges": [0.0],
    "fragment_multiplicities": [1],
    "molecular_charge": 0.0,
    "molecular_multiplicity": 1,
}
fullans10efp = {
    "fragment_files": ["cl2"],
    "hint_types": ["xyzabc"],
    "geom_hints": [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
    "units": "Angstrom",
    "fix_com": True,
    "fix_orientation": True,
    "fix_symmetry": "c1",
}
blankqm = {
    "geom": np.array([]),
    "elea": np.array([]),
    "elez": np.array([]),
    "elem": np.array([]),
    "mass": np.array([]),
    "real": np.array([]),
    "elbl": np.array([]),
    "units": "Angstrom",
    "fix_com": False,
    "fix_orientation": False,
    "fragment_separators": [],
    "fragment_charges": [0.0],
    "fragment_multiplicities": [1],
    "molecular_charge": 0.0,
    "molecular_multiplicity": 1,
}
blankefp = {
    "fragment_files": [],
    "hint_types": [],
    "geom_hints": [],
    "units": "Angstrom",
    "fix_com": True,
    "fix_orientation": True,
    "fix_symmetry": "c1",
}


def test_arrays_10a():
    subject = {
        "geom": [0, 0, 0],
        "elem": ["C"],
        "fragment_files": ["cl2"],
        "hint_types": ["xyzabc"],
        "geom_hints": [[0, 0, 0, 0, 0, 0]],
        "enable_qm": True,
        "enable_efp": True,
    }

    fullans = {"qm": copy.deepcopy(fullans10qm), "efp": copy.deepcopy(fullans10efp)}
    fullans["qm"]["fix_com"] = True
    fullans["qm"]["fix_orientation"] = True
    fullans["qm"]["fix_symmetry"] = "c1"
    fullans["qm"]["provenance"] = _arrays_prov_stamp
    fullans["efp"]["provenance"] = _arrays_prov_stamp

    final = qcelemental.molparse.from_input_arrays(**subject)
    assert compare_molrecs(fullans["qm"], final["qm"], tnm() + ": full qm")
    assert compare_molrecs(fullans["efp"], final["efp"], tnm() + ": full efp")


def test_arrays_10b():
    subject = {
        "geom": [0, 0, 0],
        "elem": ["C"],
        "fragment_files": ["cl2"],
        "hint_types": ["xyzabc"],
        "geom_hints": [[0, 0, 0, 0, 0, 0]],
        "enable_qm": False,
        "enable_efp": True,
    }

    fullans = {"efp": fullans10efp}
    fullans["efp"]["provenance"] = _arrays_prov_stamp

    final = qcelemental.molparse.from_input_arrays(**subject)
    with pytest.raises(KeyError):
        final["qm"]
    assert compare_molrecs(fullans["efp"], final["efp"], tnm() + ": full efp")


def test_arrays_10c():
    subject = {
        "geom": [0, 0, 0],
        "elem": ["C"],
        "fragment_files": ["cl2"],
        "hint_types": ["xyzabc"],
        "geom_hints": [[0, 0, 0, 0, 0, 0]],
        "enable_qm": True,
        "enable_efp": False,
    }

    fullans = {"qm": fullans10qm}
    fullans["qm"]["provenance"] = _arrays_prov_stamp

    final = qcelemental.molparse.from_input_arrays(**subject)
    assert compare_molrecs(fullans["qm"], final["qm"], tnm() + ": full qm")
    with pytest.raises(KeyError):
        final["efp"]


def test_arrays_10d():
    subject = {
        "geom": [0, 0, 0],
        "elem": ["C"],
        "enable_qm": True,
        "enable_efp": True,
        "missing_enabled_return_efp": "none",
    }

    fullans = {"qm": fullans10qm}
    fullans["qm"]["provenance"] = _arrays_prov_stamp

    final = qcelemental.molparse.from_input_arrays(**subject)
    assert compare_molrecs(fullans["qm"], final["qm"], tnm() + ": full qm")
    with pytest.raises(KeyError):
        final["efp"]


def test_arrays_10e():
    subject = {
        "geom": [0, 0, 0],
        "elem": ["C"],
        "enable_qm": True,
        "enable_efp": True,
        "missing_enabled_return_efp": "minimal",
    }

    fullans = {"qm": fullans10qm, "efp": blankefp}
    fullans["qm"]["provenance"] = _arrays_prov_stamp
    fullans["efp"]["provenance"] = _arrays_prov_stamp

    final = qcelemental.molparse.from_input_arrays(**subject)
    assert compare_molrecs(fullans["qm"], final["qm"], tnm() + ": full qm")
    assert compare_molrecs(fullans["efp"], final["efp"], tnm() + ": full efp")


def test_arrays_10f():
    subject = {
        "geom": [0, 0, 0],
        "elem": ["C"],
        "enable_qm": True,
        "enable_efp": True,
        "missing_enabled_return_efp": "error",
    }

    with pytest.raises(qcelemental.ValidationError):
        qcelemental.molparse.from_input_arrays(**subject)


def test_arrays_10g():
    subject = {
        "geom": [0, 0, 0],
        "elem": ["C"],
        "enable_qm": False,
        "enable_efp": True,
        "missing_enabled_return_efp": "none",
    }

    final = qcelemental.molparse.from_input_arrays(**subject)
    with pytest.raises(KeyError):
        final["qm"]
    with pytest.raises(KeyError):
        final["efp"]


def test_arrays_10h():
    subject = {
        "geom": [0, 0, 0],
        "elem": ["C"],
        "enable_qm": False,
        "enable_efp": True,
        "missing_enabled_return_efp": "minimal",
    }

    fullans = {"efp": blankefp}
    fullans["efp"]["provenance"] = _arrays_prov_stamp

    final = qcelemental.molparse.from_input_arrays(**subject)
    with pytest.raises(KeyError):
        final["qm"]
    assert compare_molrecs(fullans["efp"], final["efp"], tnm() + ": full efp")


def test_arrays_10i():
    subject = {
        "geom": [0, 0, 0],
        "elem": ["C"],
        "enable_qm": False,
        "enable_efp": True,
        "missing_enabled_return_efp": "error",
    }

    with pytest.raises(qcelemental.ValidationError):
        qcelemental.molparse.from_input_arrays(**subject)


def test_arrays_10j():
    subject = {"geom": [0, 0, 0], "elem": ["C"], "enable_qm": True, "enable_efp": False}

    fullans = {"qm": fullans10qm}
    fullans["qm"]["provenance"] = _arrays_prov_stamp

    final = qcelemental.molparse.from_input_arrays(**subject)
    assert compare_molrecs(fullans["qm"], final["qm"], tnm() + ": full qm")
    with pytest.raises(KeyError):
        final["efp"]


def test_arrays_10k():
    subject = {
        "fragment_files": ["cl2"],
        "hint_types": ["xyzabc"],
        "geom_hints": [[0, 0, 0, 0, 0, 0]],
        "enable_qm": True,
        "enable_efp": True,
        "missing_enabled_return_qm": "none",
    }

    fullans = {"efp": fullans10efp}
    fullans["efp"]["provenance"] = _arrays_prov_stamp

    final = qcelemental.molparse.from_input_arrays(**subject)
    with pytest.raises(KeyError):
        final["qm"]
    assert compare_molrecs(fullans["efp"], final["efp"], tnm() + ": full efp")


def test_arrays_10l():
    subject = {
        "fragment_files": ["cl2"],
        "hint_types": ["xyzabc"],
        "geom_hints": [[0, 0, 0, 0, 0, 0]],
        "enable_qm": True,
        "enable_efp": True,
        "missing_enabled_return_qm": "minimal",
    }

    fullans = {"qm": copy.deepcopy(blankqm), "efp": fullans10efp}
    fullans["qm"]["fix_com"] = True
    fullans["qm"]["fix_orientation"] = True
    fullans["qm"]["fix_symmetry"] = "c1"
    fullans["qm"]["provenance"] = _arrays_prov_stamp
    fullans["efp"]["provenance"] = _arrays_prov_stamp

    final = qcelemental.molparse.from_input_arrays(**subject)
    assert compare_molrecs(fullans["qm"], final["qm"], tnm() + ": full qm")
    assert compare_molrecs(fullans["efp"], final["efp"], tnm() + ": full efp")


def test_arrays_10m():
    subject = {
        "fragment_files": ["cl2"],
        "hint_types": ["xyzabc"],
        "geom_hints": [[0, 0, 0, 0, 0, 0]],
        "enable_qm": True,
        "enable_efp": True,
        "missing_enabled_return_qm": "error",
    }

    with pytest.raises(qcelemental.ValidationError):
        qcelemental.molparse.from_input_arrays(**subject)


def test_arrays_10n():
    subject = {
        "fragment_files": ["cl2"],
        "hint_types": ["xyzabc"],
        "geom_hints": [[0, 0, 0, 0, 0, 0]],
        "enable_qm": False,
        "enable_efp": True,
    }

    fullans = {"efp": fullans10efp}
    fullans["efp"]["provenance"] = _arrays_prov_stamp

    final = qcelemental.molparse.from_input_arrays(**subject)
    with pytest.raises(KeyError):
        final["qm"]
    assert compare_molrecs(fullans["efp"], final["efp"], tnm() + ": full efp")


def test_arrays_10o():
    subject = {
        "fragment_files": ["cl2"],
        "hint_types": ["xyzabc"],
        "geom_hints": [[0, 0, 0, 0, 0, 0]],
        "enable_qm": True,
        "enable_efp": False,
        "missing_enabled_return_qm": "none",
    }

    final = qcelemental.molparse.from_input_arrays(**subject)
    with pytest.raises(KeyError):
        final["qm"]
    with pytest.raises(KeyError):
        final["efp"]


def test_arrays_10p():
    subject = {
        "fragment_files": ["cl2"],
        "hint_types": ["xyzabc"],
        "geom_hints": [[0, 0, 0, 0, 0, 0]],
        "enable_qm": True,
        "enable_efp": False,
        "missing_enabled_return_qm": "minimal",
    }

    fullans = {"qm": blankqm}
    fullans["qm"]["provenance"] = _arrays_prov_stamp

    final = qcelemental.molparse.from_input_arrays(**subject)
    assert compare_molrecs(fullans["qm"], final["qm"], tnm() + ": full qm")
    with pytest.raises(KeyError):
        final["efp"]


def test_arrays_10q():
    subject = {
        "fragment_files": ["cl2"],
        "hint_types": ["xyzabc"],
        "geom_hints": [[0, 0, 0, 0, 0, 0]],
        "enable_qm": True,
        "enable_efp": False,
        "missing_enabled_return_qm": "error",
    }

    with pytest.raises(qcelemental.ValidationError):
        qcelemental.molparse.from_input_arrays(**subject)


def test_strings_10r():
    subject = ""

    final = qcelemental.molparse.from_string(
        subject, enable_qm=True, enable_efp=True, missing_enabled_return_qm="none", missing_enabled_return_efp="none"
    )

    print("final", final)
    with pytest.raises(KeyError):
        final["qm"]
    with pytest.raises(KeyError):
        final["efp"]


def test_strings_10s():
    subject = ""

    final = qcelemental.molparse.from_string(
        subject,
        enable_qm=True,
        enable_efp=True,
        missing_enabled_return_qm="minimal",
        missing_enabled_return_efp="minimal",
    )

    fullans = {"qm": blankqm, "efp": blankefp}
    fullans["qm"]["provenance"] = _string_prov_stamp
    fullans["efp"]["provenance"] = _string_prov_stamp

    assert compare_molrecs(fullans["qm"], final["qm"], tnm() + ": full qm")
    assert compare_molrecs(fullans["efp"], final["efp"], tnm() + ": full efp")


def test_strings_10t():
    subject = ""

    with pytest.raises(qcelemental.ValidationError):
        qcelemental.molparse.from_string(
            subject,
            enable_qm=True,
            enable_efp=True,
            missing_enabled_return_qm="error",
            missing_enabled_return_efp="error",
        )


def test_qmol_11c():
    fullans = copy.deepcopy(fullans1a)
    fullans["provenance"] = _string_prov_stamp

    asdf = qcelemental.molparse.from_string("""nocom\n8 0 0 0\n1 1 0 0""", dtype="psi4")
    assert compare_molrecs(fullans, asdf["qm"], 4)


def test_qmol_11d():
    fullans = copy.deepcopy(fullans1a)
    fullans.update({"variables": [], "geom_unsettled": [["0", "0", "0"], ["1", "0", "0"]]})
    fullans.pop("geom")
    fullans["provenance"] = _string_prov_stamp

    asdf = qcelemental.molparse.from_string("""nocom\n8 0 0 0\n1 1 0 0""", dtype="psi4+")
    assert compare_molrecs(fullans, asdf["qm"], 4)


def test_qmol_11e():
    fullans = copy.deepcopy(fullans1a)
    fullans["provenance"] = _string_prov_stamp

    asdf = qcelemental.molparse.from_string("""2\n\nO 0 0 0 \n1 1 0 0 """, dtype="xyz", fix_com=True)
    assert compare_molrecs(fullans, asdf["qm"], 4)


def test_qmol_11g():
    fullans = copy.deepcopy(fullans1a)
    fullans["provenance"] = _arrays_prov_stamp

    asdf = qcelemental.molparse.from_arrays(geom=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0], elez=[8, 1], fix_com=True, verbose=2)
    assert compare_molrecs(fullans, asdf, 4)


def test_qmol_11h():
    fullans = copy.deepcopy(fullans1a)
    fullans["provenance"] = _string_prov_stamp

    asdf = qcelemental.molparse.from_string("""nocom\n8 0 0 0\n1 1 0 0""")
    assert compare_molrecs(fullans, asdf["qm"], 4)


def test_qmol_11i():
    fullans = copy.deepcopy(fullans1a)
    fullans["provenance"] = _string_prov_stamp

    asdf = qcelemental.molparse.from_string("""nocom\n8 0 0 0\n1 1 0 0""")
    assert compare_molrecs(fullans, asdf["qm"], 4)


def test_qmol_11j():
    fullans = copy.deepcopy(fullans1a)
    fullans["provenance"] = _string_prov_stamp

    asdf = qcelemental.molparse.from_string("""2\n\nO 0 0 0 \n1 1 0 0 """, fix_com=True)
    assert compare_molrecs(fullans, asdf["qm"], 4)


def test_qmol_11p():
    fullans = copy.deepcopy(fullans1a)
    fullans["provenance"] = _arrays_prov_stamp

    asdf = qcelemental.molparse.from_arrays(
        geom=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0], elez=[8, 1], fix_com=True, units="AngSTRom"
    )
    assert compare_molrecs(fullans, asdf, 4)


def test_qmol_11q():
    with pytest.raises(KeyError):
        qcelemental.molparse.from_string("""2\n\nO 0 0 0 \n1 1 0 0 """, fix_com=True, dtype="psi3")


# QCELdef test_qmol_12():
# QCEL    asdf = qcdb.Molecule(geom=[ 0.,  0.,  0.,  1.,  0.,  0.], elez=[8, 1], fix_com=True)
# QCEL    assess_mol_11(asdf, 'qcdb.Molecule(geom, elez)')
# QCEL
# QCEL    import json
# QCEL    smol = json.dumps(asdf.to_dict(np_out=False))
# QCEL    dmol = json.loads(smol)
# QCEL
# QCEL    asdf2 = qcdb.Molecule(dmol)
# QCEL    assess_mol_11(asdf, 'qcdb.Molecule(jsondict)')

subject12 = """
 0 1
 1
 8 1 0.95
 O 2 1.40 1 A
 H 3 0.95 2 A 1 120.0

A = 105.0
"""

fullans12 = {
    "elbl": np.array(["", "", "", ""]),
    "elea": np.array([1, 16, 16, 1]),
    "elem": np.array(["H", "O", "O", "H"]),
    "elez": np.array([1, 8, 8, 1]),
    "fix_com": False,
    "fix_orientation": False,
    "fragment_charges": [0.0],
    "fragment_multiplicities": [1],
    "fragment_separators": [],
    "geom_unsettled": [[], ["1", "0.95"], ["2", "1.40", "1", "A"], ["3", "0.95", "2", "A", "1", "120.0"]],
    "mass": np.array([1.00782503, 15.99491462, 15.99491462, 1.00782503]),
    "molecular_charge": 0.0,
    "molecular_multiplicity": 1,
    "real": np.array([True, True, True, True]),
    "units": "Angstrom",
    "variables": [["A", 105.0]],
    "fix_symmetry": "c1",
}
ans12 = {
    "elbl": ["1", "8", "O", "H"],
    "fragment_charges": [0.0],
    "fragment_files": [],
    "fragment_multiplicities": [1],
    "fragment_separators": [],
    "geom_hints": [],
    "geom_unsettled": [[], ["1", "0.95"], ["2", "1.40", "1", "A"], ["3", "0.95", "2", "A", "1", "120.0"]],
    "hint_types": [],
    "variables": [("A", "105.0")],
    "fix_symmetry": "c1",
}


def test_psi4_qm_12a():
    subject = subject12
    fullans = copy.deepcopy(fullans12)
    fullans["provenance"] = _string_prov_stamp

    final, intermed = qcelemental.molparse.from_string(subject, return_processed=True, fix_symmetry="c1")
    assert compare_recursive(ans12, intermed, tnm() + ": intermediate")
    assert compare_molrecs(fullans, final["qm"], tnm() + ": full")


def test_tooclose_error():
    subject = """2 -1 -1 -1\n2 -1 -1 -1.05"""

    with pytest.raises(qcelemental.ValidationError) as e:
        qcelemental.molparse.from_string(subject)

    assert "too close" in str(e.value)


def test_cartbeforezmat_error():
    subject = """He 0 0 0\nHe 1 2"""

    with pytest.raises(qcelemental.ValidationError) as e:
        qcelemental.molparse.from_string(subject)

    assert "Mixing Cartesian and Zmat" in str(e.value)


def test_jumbledzmat_error():
    subject = """He\nHe 1 2. 2 100. 3 35.\nHe 1 2."""

    with pytest.raises(qcelemental.ValidationError) as e:
        qcelemental.molparse.from_string(subject)

    assert "aim for lower triangular" in str(e.value)


def test_steepzmat_error():
    subject = """He\nHe 1 2.\nHe 1 2. 2 100. 3 35."""

    with pytest.raises(qcelemental.ValidationError) as e:
        qcelemental.molparse.from_string(subject)

    assert "aim for lower triangular" in str(e.value)


def test_zmatvar_error():
    with pytest.raises(qcelemental.ValidationError) as e:
        qcelemental.molparse.from_arrays(
            domain="qmvz", elem=["Rn", "Rn"], variables=[["bond", 2.0, "badextra"]], geom_unsettled=[[], ["1", "bond"]]
        )

    assert "Variables should come in pairs" in str(e.value)


def test_toomanyfrag_error():
    with pytest.raises(qcelemental.ValidationError) as e:
        qcelemental.molparse.from_arrays(
            domain="qmvz",
            speclabel=True,
            elbl=["ar1", "42AR2"],
            fragment_multiplicities=[3, 3],
            fragment_separators=[1, 2],
            geom_unsettled=[[], ["1", "bond"]],
            hint_types=[],
            units="Bohr",
            variables=[("bond", "3")],
        )

    assert "zero-length fragment" in str(e.value)


def test_fragsep_error():
    with pytest.raises(qcelemental.ValidationError) as e:
        qcelemental.molparse.from_arrays(
            domain="qmvz",
            speclabel=True,
            elbl=["ar1", "42AR2"],
            fragment_multiplicities=[3, 3],
            fragment_separators=np.array(["1"]),
            geom_unsettled=[[], ["1", "bond"]],
            hint_types=[],
            units="Bohr",
            variables=[("bond", "3")],
        )

    assert "unable to perform trial np.split on geometry" in str(e.value)


def test_cartzmatcart():
    subject = """1 1
# This part is just a normal Cartesian geometry specification for benzene
C          0.710500000000    -0.794637665924    -1.230622098778
C          1.421000000000    -0.794637665924     0.000000000000
C          0.710500000000    -0.794637665924     1.230622098778
C         -0.710500000000    -0.794637665924     1.230622098778
H          1.254500000000    -0.794637665924    -2.172857738095
H         -1.254500000000    -0.794637665924     2.172857738095
C         -0.710500000000    -0.794637665924    -1.230622098778
C         -1.421000000000    -0.794637665924     0.000000000000
H          2.509000000000    -0.794637665924     0.000000000000
H          1.254500000000    -0.794637665924     2.172857738095
H         -1.254500000000    -0.794637665924    -2.172857738095
H         -2.509000000000    -0.794637665924     0.000000000000
# And the hydronium part is specified using a zmatrix, referencing the benzene coordinates
X  1  CC  3  30   2  A2
O  13 R   1  90   2  90
H  14 OH  13 TDA  1  0
H  14 OH  15 TDA  13 A1
H  14 OH  15 TDA  13 -A1
He         3.710500000000    -0.794637665924    -1.230622098778

CC    = 1.421
CH    = 1.088
A1    = 120.0
A2    = 180.0
OH    = 1.05
R     = 4.0
"""

    qcelemental.molparse.from_string(subject)


def test_fixcom_error():
    with pytest.raises(qcelemental.ValidationError) as e:
        qcelemental.molparse.from_arrays(elez=[3], molecular_charge=1, geom=[0, 0, 0], fix_com="thanks!")

    assert "Invalid fix_com" in str(e.value)


def test_fixori_error():
    with pytest.raises(qcelemental.ValidationError) as e:
        qcelemental.molparse.from_arrays(elez=[3], molecular_charge=1, geom=[0, 0, 0], fix_orientation=-1)

    assert "Invalid fix_orientation" in str(e.value)


def test_units_error():
    with pytest.raises(qcelemental.ValidationError) as e:
        qcelemental.molparse.from_arrays(elez=[3], molecular_charge=1, geom=[0, 0, 0], units="furlong")

    assert "Invalid molecule geometry units" in str(e.value)


def test_domain_error():
    with pytest.raises(qcelemental.ValidationError) as e:
        qcelemental.molparse.from_arrays(domain="kitten")

    assert "Topology domain kitten not available" in str(e.value)


def test_natom_error():
    with pytest.raises(qcelemental.ValidationError) as e:
        qcelemental.molparse.from_arrays(elem=["C"], elea=[12, 13], geom=[0, 0, 0])

    assert "Dimension mismatch natom" in str(e.value)


def test_incompletefrag_error():
    with pytest.raises(qcelemental.ValidationError) as e:
        qcelemental.molparse.from_arrays(
            domain="qmvz",
            speclabel=True,
            elbl=["ar1", "42AR2"],
            fragment_multiplicities=[3, 3],
            geom_unsettled=[[], ["1", "bond"]],
            hint_types=[],
            units="Bohr",
            variables=[("bond", "3")],
        )

    assert "Fragment quantities given without separation info" in str(e.value)


def test_badmult_error():
    with pytest.raises(qcelemental.ValidationError) as e:
        qcelemental.molparse.from_arrays(
            domain="qmvz",
            speclabel=True,
            elbl=["ar1", "42AR2"],
            fragment_multiplicities=[-3, 3],
            fragment_separators=np.array([1]),
            geom_unsettled=[[], ["1", "bond"]],
            hint_types=[],
            units="Bohr",
            variables=[("bond", "3")],
        )

    # formerly: assert "fragment_multiplicities not among None or positive integer" in str(e.value)
    assert "Multiplicity must be positive" in str(e.value)


def test_badchg_error():
    with pytest.raises(qcelemental.ValidationError) as e:
        qcelemental.molparse.from_arrays(
            domain="qmvz",
            speclabel=True,
            elbl=["ar1", "42AR2"],
            fragment_charges=[[], {}],
            fragment_separators=np.array([1]),
            geom_unsettled=[[], ["1", "bond"]],
            hint_types=[],
            units="Bohr",
            variables=[("bond", "3")],
        )

    assert "fragment_charges not among None or float" in str(e.value)


def test_fraglen_error():
    with pytest.raises(qcelemental.ValidationError) as e:
        qcelemental.molparse.from_arrays(
            domain="qmvz",
            speclabel=True,
            elbl=["na", "cl"],
            fragment_charges=[1, -1, 0],
            fragment_separators=np.array([1]),
            geom_unsettled=[[], ["1", "bond"]],
            hint_types=[],
            units="Bohr",
            variables=[("bond", "3")],
        )

    assert "mismatch among fragment quantities" in str(e.value)


def test_zmatfragarr_14a():
    fullans = copy.deepcopy(fullans14)
    fullans["provenance"] = _arrays_prov_stamp

    final = qcelemental.molparse.from_arrays(
        domain="qmvz",
        speclabel=True,
        elbl=["ar1", "42AR2"],
        fragment_multiplicities=[3, 3],
        fragment_separators=[1],
        geom_unsettled=[[], ["1", "bond"]],
        hint_types=[],
        units="Bohr",
        variables=[("bond", "3")],
    )

    assert compare_molrecs(fullans, final, tnm() + ": full")


def test_zmatfragarr_14b():
    fullans = copy.deepcopy(fullans14)
    fullans["elbl"] = ["", ""]
    fullans["provenance"] = _arrays_prov_stamp

    final = qcelemental.molparse.from_arrays(
        domain="qmvz",
        speclabel=False,
        elez=[18, 18],
        elea=np.array([40, 42]),
        real=[True, True],
        fragment_multiplicities=[3, 3],
        fragment_separators=[1],
        geom_unsettled=[[], ["1", "bond"]],
        hint_types=[],
        units="Bohr",
        variables=[("bond", "3")],
    )

    assert compare_molrecs(fullans, final, tnm() + ": full")


def test_zmatfragarr_14c():
    fullans = copy.deepcopy(fullans14)
    fullans["elbl"] = ["", ""]
    fullans["fix_com"] = True
    fullans["fix_orientation"] = True
    fullans["mass"] = fullans["mass"].tolist()  # other np vs. list diffs are hidden by compare_molrecs
    fullans["real"] = fullans["real"].tolist()
    fullans["provenance"] = _arrays_prov_stamp

    final = qcelemental.molparse.from_arrays(
        np_out=False,
        domain="qmvz",
        speclabel=False,
        elez=[18, 18],
        elea=[40, None],
        mass=[None, 41.96304574],
        real=[True, True],
        fragment_multiplicities=[3, 3],
        fragment_separators=[1],
        geom_unsettled=[[], ["1", "bond"]],
        hint_types=[],
        units="Bohr",
        fix_com=True,
        fix_orientation=True,
        variables=[("bond", "3")],
    )

    assert compare_molrecs(fullans, final, tnm() + ": full")


subject14 = """
        0 3
        ar1
        --
        0 3
        42AR2 1 bond
    units a.u.
        bond =3.0
        """

fullans14 = {
    "elbl": np.array(["1", "2"]),
    "elea": np.array([40, 42]),
    "elem": np.array(["Ar", "Ar"]),
    "elez": np.array([18, 18]),
    "fix_com": False,
    "fix_orientation": False,
    "fragment_charges": [0.0, 0.0],
    "fragment_multiplicities": [3, 3],
    "fragment_separators": [1],
    "geom_unsettled": [[], ["1", "bond"]],
    "mass": np.array([39.96238312, 41.96304574]),
    "molecular_charge": 0.0,
    "molecular_multiplicity": 5,
    "real": np.array([True, True]),
    "units": "Bohr",
    "variables": [["bond", 3.0]],
}


def test_zmatfragstr_14d():
    subject = subject14
    fullans = copy.deepcopy(fullans14)
    fullans["provenance"] = _string_prov_stamp

    final, intermed = qcelemental.molparse.from_string(subject, return_processed=True, verbose=2)
    assert compare_molrecs(fullans, final["qm"], tnm() + ": full")


def test_badprov0_error():
    with pytest.raises(qcelemental.ValidationError) as e:
        qcelemental.molparse.from_arrays(geom=[1, 2, 3], elez=[4], provenance="mine")

    assert "Provenance entry is not dictionary" in str(e.value)


def test_badprov1_error():
    with pytest.raises(qcelemental.ValidationError) as e:
        qcelemental.molparse.from_arrays(
            geom=[1, 2, 3], elez=[4], provenance={"creator": ("psi", "tuple"), "routine": "buggy", "version": "0.1b"}
        )

    assert """Provenance key 'creator' should be string of creating program's name:""" in str(e.value)


def test_badprov2_error():
    with pytest.raises(qcelemental.ValidationError) as e:
        qcelemental.molparse.from_arrays(
            geom=[1, 2, 3], elez=[4], provenance={"creator": "", "routine": "buggy", "version": "my.vanity.version.13"}
        )

    assert """Provenance key 'version' should be a valid PEP 440 string:""" in str(e.value)


def test_badprov3_error():
    with pytest.raises(qcelemental.ValidationError) as e:
        qcelemental.molparse.from_arrays(
            geom=[1, 2, 3], elez=[4], provenance={"creator": "", "routine": 5, "version": "0.1b"}
        )

    assert """Provenance key 'routine' should be string of creating function's name:""" in str(e.value)


def test_badprov4_error():
    with pytest.raises(qcelemental.ValidationError) as e:
        qcelemental.molparse.from_arrays(
            geom=[1, 2, 3], elez=[4], provenance={"creators": "", "routine": "buggy", "version": "0.1b"}
        )

    assert """Provenance keys (['creator', 'routine', 'version']) incorrect:""" in str(e.value)


fullans17 = {
    "geom": np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
    "elea": np.array([1, 32, 1]),
    "elez": np.array([1, 16, 1]),
    "elem": np.array(["H", "S", "H"]),
    "mass": np.array([1.00782503, 31.9720711744, 1.00782503]),
    "real": np.array([False, True, True]),
    "elbl": np.array(["", "", ""]),
    "units": "Bohr",
    "fix_com": False,
    "fix_orientation": False,
    "fragment_separators": [],
    "fragment_charges": [-1.0],
    "fragment_multiplicities": [1],
    "molecular_charge": -1.0,
    "molecular_multiplicity": 1,
    "connectivity": [(0, 1, 1.0), (1, 2, 1.0)],
}


def test_connectivity_17a():
    fullans = copy.deepcopy(fullans17)
    fullans["provenance"] = _arrays_prov_stamp

    final = qcelemental.molparse.from_arrays(
        geom=np.arange(9),
        units="Bohr",
        elez=[1, 16, 1],
        molecular_charge=-1,
        real=[False, True, True],
        connectivity=[(0, 1, 1), (1, 2, 1)],
    )

    assert compare_molrecs(fullans, final, tnm() + ": full")


def test_connectivity_17b():
    fullans = copy.deepcopy(fullans17)
    fullans["provenance"] = _arrays_prov_stamp

    final = qcelemental.molparse.from_arrays(
        geom=np.arange(9),
        units="Bohr",
        elez=[1, 16, 1],
        molecular_charge=-1,
        real=[False, True, True],
        connectivity=[(2.0, 1, 1), (1, 0, 1)],
    )

    assert compare_molrecs(fullans, final, tnm() + ": full")


def test_connectivity_atindex_error():
    with pytest.raises(qcelemental.ValidationError) as e:
        qcelemental.molparse.from_arrays(
            geom=np.arange(9),
            units="Bohr",
            elez=[1, 16, 1],
            molecular_charge=-1,
            real=[False, True, True],
            connectivity=[(2.1, 1, 1), (1, 0, 1)],
        )

    assert "Connectivity first atom should be int" in str(e.value)


def test_connectivity_atrange_error():
    with pytest.raises(qcelemental.ValidationError) as e:
        qcelemental.molparse.from_arrays(
            geom=np.arange(9),
            units="Bohr",
            elez=[1, 16, 1],
            molecular_charge=-1,
            real=[False, True, True],
            connectivity=[(2, 1, 1), (1, -1, 1)],
        )

    assert "Connectivity second atom should be int" in str(e.value)


def test_connectivity_bondorder_error():
    with pytest.raises(qcelemental.ValidationError) as e:
        qcelemental.molparse.from_arrays(
            geom=np.arange(9),
            units="Bohr",
            elez=[1, 16, 1],
            molecular_charge=-1,
            real=[False, True, True],
            connectivity=[(2, 1, 1), (1, 0, 6)],
        )

    assert "Connectivity bond order should be float" in str(e.value)


def test_connectivity_type_error():
    with pytest.raises(qcelemental.ValidationError) as e:
        qcelemental.molparse.from_arrays(
            geom=np.arange(9),
            units="Bohr",
            elez=[1, 16, 1],
            molecular_charge=-1,
            real=[False, True, True],
            connectivity="wire",
        )

    assert "Connectivity entry is not of form" in str(e.value)


#'geom_unsettled': [[], ['1', '2.'], ['1', '2.', '2', '100.', '3', '35.']],

# final, intermed = qcelemental.molparse.from_string(subject, return_processed=True, verbose=2)
# import pprint
# pprint.pprint(final)
# pprint.pprint(intermed)

# assert False
