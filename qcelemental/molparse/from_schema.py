from typing import Dict, List, Union

import numpy as np

from ..exceptions import ValidationError
from ..util import provenance_stamp
from .from_arrays import from_arrays


def from_schema(molschema: Dict, *, nonphysical: bool = False, throw_reorder: bool = True, verbose: int = 1) -> Dict:
    r"""Construct molecule dictionary representation from non-Psi4 schema.

    Parameters
    ----------
    molschema
        Dictionary form of Molecule following known schema.
    nonphysical
        Do allow masses outside an element's natural range to pass validation?
    throw_reorder
        Do, when non-contiguous fragments detected, raise
        ValidationError (``True``) or to proceed to reorder atoms to
        contiguize fragments (``False``).
    verbose
        Amount of printing.

    Returns
    -------
    molrec : dict
        Dictionary representation of instance.

    """

    if (
        molschema.get("schema_name", "").startswith("qc_schema")
        or molschema.get("schema_name", "").startswith("qcschema")
    ) and (molschema.get("schema_version", "") == 1):
        ms = molschema["molecule"]
    elif molschema.get("schema_name", "").startswith("qcschema_molecule") and molschema.get("schema_version", "") == 2:
        ms = molschema
    else:
        raise ValidationError(
            """Schema not recognized, schema_name/schema_version: {}/{} """.format(
                molschema.get("schema_name", "(none)"), molschema.get("schema_version", "(none)")
            )
        )

    if "fragments" in ms:
        frag_pattern = ms["fragments"]
    else:
        frag_pattern = [np.arange(len(ms["symbols"]))]

    dcontig = contiguize_from_fragment_pattern(
        frag_pattern,
        geom=ms["geometry"],
        elea=ms.get("mass_numbers", None),
        elez=ms.get("atomic_numbers", None),
        elem=ms["symbols"],
        mass=ms.get("masses", None),
        real=ms.get("real", None),
        elbl=ms.get("atom_labels", None),
        throw_reorder=throw_reorder,
    )

    molrec = from_arrays(
        geom=dcontig["geom"],
        elea=dcontig["elea"],
        elez=dcontig["elez"],
        elem=dcontig["elem"],
        mass=dcontig["mass"],
        real=dcontig["real"],
        elbl=dcontig["elbl"],
        name=ms.get("name", None),
        units="Bohr",
        input_units_to_au=None,
        fix_com=ms.get("fix_com", None),
        fix_orientation=ms.get("fix_orientation", None),
        fix_symmetry=ms.get("fix_symmetry", None),
        fragment_separators=dcontig["fragment_separators"],
        fragment_charges=ms.get("fragment_charges", None),
        fragment_multiplicities=ms.get("fragment_multiplicities", None),
        molecular_charge=ms.get("molecular_charge", None),
        molecular_multiplicity=ms.get("molecular_multiplicity", None),
        comment=ms.get("comment", None),
        provenance=ms.get("provenance", None),
        connectivity=ms.get("connectivity", None),
        domain="qm",
        # missing_enabled_return=missing_enabled_return,
        speclabel=False,
        # tooclose=tooclose,
        # zero_ghost_fragments=zero_ghost_fragments,
        nonphysical=nonphysical,
        # mtol=mtol,
        verbose=verbose,
    )

    # replace from_arrays stamp with from_schema stamp
    molrec["provenance"] = provenance_stamp(__name__)

    return molrec


def contiguize_from_fragment_pattern(
    frag_pattern: List[List[int]],
    *,
    geom: Union[np.ndarray, List[List]] = None,
    verbose: int = 1,
    throw_reorder: bool = False,
    **kwargs,
):
    r"""Take (nat, ?) array-like arrays and return with atoms arranged by (nfr, ?) `frag_pattern`.

    Parameters
    ----------
    frag_pattern
        (nfr, ?) list of indices (0-indexed) grouping atoms into
        molecular fragments within the topology.
    geom
        (nat, 3) or (3 * nat, ) ndarray or list o'lists of Cartesian
        coordinates, possibly with atoms belonging to the same fragment
        being dispersed in `geom`.
    throw_reorder
        Whether, when non-contiguous fragments detected, to raise
        ValidationError (``True``) or to proceed to reorder atoms to
        contiguize fragments (``False``).
    verbose
        Quantity of printing
    kwargs : None or array-like
        Each additional array will be returned with ordering applied
        in the return dictionary.

    Returns
    -------
    fragment_separators : array-like of int
        (nfr - 1, ) list of atom indices at which to split `geom` into fragments.
    geom : ndarray of float, optional
        (3 * nat, ) Cartesian coordinates with fragments contiguous.
    kwargs : None or ndarray, optional
        (nat, ) Each `kwargs` input array reordered for contiguous fragments.

    Raises
    ------
    qcelemental.ValidationError
        When `frag_pattern` skips atoms or any array has inconsistent
        length. If `throw_reorder`, raises when non-contiguous fragments
        detected.

    """

    vsplt = np.cumsum([len(fr) for fr in frag_pattern])
    nat = vsplt[-1]
    fragment_separators = vsplt[:-1]

    # Nothing to do for len =1 and ordered
    if (len(fragment_separators) == 0) and np.all(np.diff(frag_pattern[0]) == 1):
        returns = {"fragment_separators": fragment_separators}
        if geom is not None:
            returns.update({"geom": geom.copy()})
        extras = {k: v for k, v in kwargs.items()}
        returns.update(extras)

        ncgeom = np.asarray(geom).reshape(-1, 3)
        if nat != ncgeom.shape[0]:
            raise ValidationError("""dropped atoms! nat = {} != {}""".format(nat, ncgeom.shape[0]))

        return returns

    do_reorder = False
    if not np.array_equal(np.sort(np.concatenate(frag_pattern)), np.arange(nat)):
        raise ValidationError("""Fragmentation pattern skips atoms: {}""".format(frag_pattern))

    if not np.array_equal(np.concatenate(frag_pattern), np.arange(nat)):
        print("""Warning: QCElemental is reordering atoms to accommodate non-contiguous fragments""")
        do_reorder = True

    if do_reorder and throw_reorder:
        raise ValidationError(
            """Error: QCElemental would need to reorder atoms to accommodate non-contiguous fragments"""
        )

    if geom is not None:
        ncgeom = np.asarray(geom).reshape(-1, 3)
        if nat != ncgeom.shape[0]:
            raise ValidationError("""dropped atoms! nat = {} != {}""".format(nat, ncgeom.shape[0]))
        geom = np.vstack([ncgeom[fr] for fr in frag_pattern])
        geom = geom.reshape((-1))

    def reorder(arr):
        if nat != len(arr):
            raise ValidationError("""wrong number of atoms in array: nat = {} != {}""".format(nat, len(arr)))
        return np.concatenate([np.array(arr)[fr] for fr in frag_pattern], axis=0)

    returns = {"fragment_separators": fragment_separators}
    if geom is not None:
        returns.update({"geom": geom})
    extras = {k: (None if v is None else reorder(v)) for k, v in kwargs.items()}
    returns.update(extras)

    return returns
