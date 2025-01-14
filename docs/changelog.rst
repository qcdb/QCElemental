Changelog
=========

.. X.Y.0 / 2024-MM-DD (Unreleased)
.. -------------------
..
.. Breaking Changes
.. ++++++++++++++++
..
.. New Features
.. ++++++++++++
..
.. Enhancements
.. ++++++++++++
..
.. Bug Fixes
.. +++++++++
..
.. Misc.
.. +++++


0.30.0 / 2024-MM-DD (Unreleased)
-------------------

Breaking Changes
++++++++++++++++
* The very old model names `ResultInput`, `Result`, `ResultProperties`, `Optimization` deprecated in 2019 are now only available through `qcelelemental.models.v1`
* ``models.v2`` do not support AutoDoc. The AutoDoc routines have been left at pydantic v1 syntax. Use autodoc-pydantic for Sphinx instead.
* Unlike Levi's pyd v2, this doesn't forward define dict, copy, json to v2 models. Instead it backwards-defines model_dump, model_dump_json, model_copy to v1. This will impede upgrading but be cleaner in the long run. See commented-out functions to temporarily restore this functionality. v2.Molecule retains its dict for now

New Features
++++++++++++
* Downstream code should ``from qcelemental.models.v1 import Molecule, AtomicResult`` etc. to assure medium-term availability of existing models.
* New pydantic v2 models available as ``from qcelemental.models.v2 import Molecule, AtomicResult`` etc.
- (:pr:`361`) Switch from poetry to setuptools build backend. 

Enhancements
++++++++++++
- (:pr:`364`) 
- (:pr:`364`) 
- (:pr:`364`) 
- (:pr:`364`) main storage for ``v2.TorsionDriveResult`` moved from ``optimization_history`` to ``scan_results``.
- (:pr:`364`) ``v2.TorsionDriveInput.initial_molecule`` restored from ``initial_molecules``.
- (:pr:`364`) default of OptimizationProtocol.trajectory_results changed to "none" from "all" in v1. much info can now come from properties.
- (:pr:`364`) v2.OptimizationProtocol renamed trajectory_results from trajectory in accordance with the protocol naming the controlled field. no default change yet.
- (:pr:`364`) v1/v2: import ElectronShell, BasisCenter, ECPPotential from top level models
- (:pr:`364`) molparse learns to pass through schema v3, though no new field for Mol yet.
- (:pr:`364`) ``v2.FailedOperation`` gained schema_name and schema_version=2. unversioned in v1
- (:pr:`364`) ``v2.BasisSet.schema_version`` is now 2, with no layout change.
- (:pr:`364`) ``v2.Molecule.schema_version`` is now 3. convert_v of all the models learned to handle the new schema_version.
- (:pr:`364`) v2: standardizing on In/Res get versions, Ptcl/Kw/Spec get only schema_name. At, Opt, TD
- (:pr:`364`) v1/v2: removing the version_stamps from the models: At, Opt, TD, Fail, BAsis, Mol. so it will error rather than clobber if constructed with wrong version. convert_v now handles.
- (:pr:`364`) convert_v functions learned to handle model.basis=BasisSet, not just str.
- (:pr:`364`) ``Molecule`` and ``BasisSet``  and ``WavefunctionProperties`` learned to ``convert_v`` to interconvert between v1 and v2. No layout changes. 
  ``BasisSet.schema_name`` standardized to ``qcschema_basis_set``.
  Both classes get their ``schema_name`` as Literal now
- (:pr:`360`) ``Molecule`` learned new functions ``element_composition`` and ``molecular_weight``.
  The first gives a dictionary of element symbols and counts, while the second gives the weight in amu.
  Both can access the whole molecule or per-fragment like the existing ``nelectrons`` and
  ``nuclear_repulsion_energy``. All four can now select all atoms or exclude ghosts (default).
- (:pr:`364`) separated procedures.py and renamed results.py so models are separated into atomic.py, optimization.py, torsion_drive.py, failure models moved to failed_operation.py. basis.py to basis_set.py
- (:pr:`364`) ``schema_name`` output chanded to result ``qcschema_output`` to ``qcschema_atomic_result``. also opt
- (:pr:`364`) ``TDKeywords`` renamed to ``TorsionDriveKeywords``
- (:pr:`364`) ``AtomicResultProtocols`` renamed to ``AtomicProtocols`` and ``AtomicResultProperties`` to ``AtomicProperties``
- (:pr:`364`) new ``v2.TorsionDriveProtocols`` model with field ``scan_results`` to control all/none/lowest saving of optimizationresults at each grid point. Use "all" for proper conversion to v1.
- (:pr:`363`) ``v2.TorsionDriveResult`` no longer inherits from Input and now has indep id and extras and new native_files.
- (:pr:`363`) ``v2.TorsionDriveInput.initial_molecule`` now ``initial_molecules`` as it's a list of >=1 molecules. keep change?
- (:pr:`363`) ``v2. TorsionDriveSpecification`` is a new model. instead of ``v2.TorsionDriveInput`` having a ``input_specification`` and an ``optimization_spec`` fields, it has a ``specification`` field that is a ``TorsionDriveSpecification`` which in turn hold opt info and in turn gradient/atomic info. 
- (:pr:`363`) ``v2.TDKeywords`` got a ``schema_name`` field.
- (:pr:`363`) ``native_files`` field added to ``v2.OptimizationResult`` and ``v2.TorsionDriveResult`` gained a ``native_files`` field, though not protocols for user control.
- (:pr:`363`) ``v2.AtomicResult.convert_v()`` learned external_protocols option to inject that field if known from OptIn
- (:pr:`363`) OptimizationSpecification learned a ``convert_v`` function to interconvert.
- (:pr:`363`) all the v2 models of ptcl/kw/spec/in/prop/res type have ``schema_name``.  ``qcschema_input`` and ``qcschema_output`` now are ``qcschema_atomic_input`` and ``qcschema_atomic_output``
- (:pr:`363`) whereas ``v1.AtomicInput`` and ``v1.QCInputSpecification`` shared the same schema_name, ``v2.AtomicInput`` and ``v2.AtomicSpecification`` do not. This is a step towards more explicit schema names.
- (:pr:`363`) ``v2.AtomicResult`` gets a literal schema_name and it no longer accepts the qc_schema*
- (:pr:`363`) ``v2.OptimizatonResult.energies`` becomes ``v2.OptimizationResult.trajectory_properties`` and ManyBody allowed as well as atomic. Much expands information returned
- (:pr:`363`) ``v2.OptimizatonResult.trajectory`` becomes ``v2.OptimizationResult.trajectory_results`` and ManyBody allowed as well as atomic.
- (:pr:`363`) a new basic ``v2.OptimizationProperties`` for expansion later. for now has number of opt iter. help by `OptimizationResult.properties`
- (:pr:`363`) ``v2.OptimizationResult`` gained a ``input_data`` field for the corresponding ``OptimizationInput`` and independent ``id`` and ``extras``. No longer inherits from ``OptimizationInput``.
                 Literal schema_name. Added ``native_files`` field.
- (:pr:`363`) ``v2.OptimizationInput`` got a Literal schema_name now. field ``specification`` now takes an ``OptimizationSpecification`` that itself takes an ``AtomicSpecification`` replaces field ``input_specification`` that took a ``QCInputSpecification``. ``v2.OptimizationInput`` gained a ``protocols`` field.
              fields ``keywords``, ``extras``, and ``protocols`` from Input are now in ``OptimizationSpecification``
- (:pr:`363`) ``v2.OptimizationSpecification`` now is used every optimization as ``v2.OptimizationInput.specification`` = ``OptimizationSpecification`` rather than only in torsion drives. No longer has schema_name and schema_version.
              Its. ``procedures`` field is now ``program``. Gains new field ``specification`` that is most commonly ``AtomicSpecification`` but could be ``ManyBodySpecification`` or any other E/G/H producer.
- (:pr:`363`) ``v2.OptimizationInput`` now takes consolidated ``AtomicSpecification`` rather than ``QCInputSpecification`` (now deleted)
- (:pr:`359`) ``v2.AtomicInput`` lost extras so extras belong unambiguously to the specification.
- (:pr:`359`) ``v2.AtomicSpecification``, unlike ``v1.QCInputSpecification``, doesn't have schema_name and schema version.
- (:pr:`359`) misc -- ``isort`` version bumped to 5.13 and imports and syntax take advantage of python 3.8+
- (:pr:`359`) ``v2.AtomicInput`` gained a ``specification`` field where driver, model, keywords, extras, and protocols now live. ``v2.AtomicSpecification`` and ``v1.QCInputSpecification`` (used by opt and td) learned a ``convert_v`` to interconvert.
- (:pr:`358`) ``v1.AtomicResult.convert_v`` learned a ``external_input_data`` option to inject that field (if known) rather than using incomplete reconstruction from the v1 Result. may not be the final sol'n.
- (:pr:`358`) ``v2.FailedOperation`` gained schema_name and schema_version=2.
- (:pr:`358`) ``v2.AtomicResult`` no longer inherits from ``v2.AtomicInput``. It gained a ``input_data`` field for the corresponding ``AtomicInput`` and independent ``id`` and ``molecule`` fields (the latter being equivalvent to ``v1.AtomicResult.molecule`` with the frame of the results; ``v2.AtomicResult.input_data.molecule`` is new, preserving the input frame). Gained independent ``extras``
- (:pr:`358`) Both v1/v2 ``AtomicResult.convert_v()`` learned to handle the new ``input_data`` layout.
- (:pr:`357`, :issue:`536`) ``v2.AtomicResult``, ``v2.OptimizationResult``, and ``v2.TorsionDriveResult`` have the ``success`` field enforced to ``True``. Previously it could be set T/F. Now validation errors if not T. Likewise ``v2.FailedOperation.success`` is enforced to ``False``.
- (:pr:`357`, :issue:`536`) ``v2.AtomicResult``, ``v2.OptimizationResult``, and ``v2.TorsionDriveResult`` have the ``error`` field removed. This isn't used now that ``success=True`` and failure should be routed to ``FailedOperation``.
- (:pr:`357`) ``v1.Molecule`` had its schema_version changed to a Literal[2] (remember Mol is one-ahead of general numbering scheme) so new instances will be 2 even if another value is passed in. Ditto ``v2.BasisSet.schema_version=2``. Ditto ``v1.BasisSet.schema_version=1`` Ditto ``v1.QCInputSpecification.schema_version=1`` and ``v1.OptimizationSpecification.schema_version=1``.
- (:pr:`357`) ``v2.AtomicResultProperties``, ``v2.QCInputSpecification``, ``v2.OptimizationSpecification`` lost its schema_version until we see if its really needed.
- (:pr:`357`) ``v2.OptimizationSpecification`` gained extras field
- (:pr:`357`) ``v1.FailedOperation.extras`` and ``v2.FailedOperation.extras`` default changed from None to {}
* Fix a lot of warnings originating in this project.
* `Molecule.extras` now defaults to `{}` rather than None in both v1 and v2. Input None converts to {} upon instantiation.
* ``v2.FailedOperation`` field `id` is becoming `Optional[str]` instead of plain `str` so that the default validates.
* v1.ProtoModel learned `model_copy`, `model_dump`, `model_dump_json` methods (all w/o warnings) so downstream can unify on newer syntax. Levi's work alternately/additionally taught v2 `copy`, `dict`, `json` (all w/warning) but dict has an alternate use in Pydantic v2.
* ``AtomicInput`` and ``AtomicResult`` ``OptimizationInput``, ``OptimizationResult``, ``TorsionDriveInput``, ``TorsionDriveResult``, ``FailedOperation`` (both versions) learned a ``.convert_v(ver)`` function that returns self or the other version.
* The ``models.v2`` ``AtomicInput``, ``AtomicResult``, ``AtomicResultProperties`` ``OptimizationInput``, ``OptimizationResult``, ``TorsionDriveInput``, ``TorsionDriveResult`` had their `schema_version` changed to a Literal[2] and validated so new instances will be 2, even if another value passed in.
* The ``models.v1`` ``AtomicInput``, ``AtomicResult``, ``OptimizationInput``, ``OptimizationResult``, ``TorsionDriveInput``, ``TorsionDriveResult`` had their `schema_version` changed to a Literal[1] and validated so new instances will be 1, even if another value passed in.
* The ``models.v1`` and ``models.v2`` ``OptimizationResult`` given schema_version for the first time.
* The ``models.v2`` have had their `schema_version` bumped for ``BasisSet``, ``AtomicInput``, ``OptimizationInput`` (implicit for ``AtomicResult`` and ``OptimizationResult``), ``TorsionDriveInput`` , and ``TorsionDriveResult``.
* The ``models.v2`` ``AtomicResultProperties`` has been given a ``schema_name`` and ``schema_version`` (2) for the first time.
* Note that ``models.v2`` ``QCInputSpecification`` and ``OptimizationSpecification`` have *not* had schema_version bumped.
* All of ``Datum``, ``DFTFunctional``, and ``CPUInfo`` models, none of which are mixed with QCSchema models, are translated to Pydantic v2 API syntax.
* Models ``procedures.TorsionDriveInput``, ``procedures.TorsionDriveResult``, ``common_models.Model``, ``results.AtomicResultProtocols`` are now importable from ``qcel.models`` (or its ``v1`` and ``v2`` sub) directly. For generic and v1, ``procedures.QCInputSpecification`` and ``procedures.OptimizationSpecification`` and ``procedures.TDKeywords`` are also importable from models.

Bug Fixes
+++++++++

Misc.
+++++

* added warnings to dummy files models/results.py etc. classes are rerouted to v1 so downstream can run w/o alteration with `from qcelemental.models.procedures import OptimizationInput`
* copied in pkg_resources.safe_version code as follow-up to Eric switch to packaging as both nwchem and gamess were now working. the try_harder_safe_version might be even bettter


0.28.0 / 2024-06-21
-------------------

Enhancements
++++++++++++
- (:pr:`338`) Adapts for numpy v2 (only works with pint >= v0.24). v1 still compatible.
- (:pr:`335`, :issue:`334`) Numpy >=1.26 only works with pint >=0.21. @TyBalduf


0.27.1 / 2023-10-26
-------------------

Bug Fixes
+++++++++
- (:pr:`329`) Continues :pr:`328` adding ``util.which`` workaround for only python v3.12.0 and psi4
  (can be expanded) to correctly select among cmd, cmd.bat, cmd.exe.


0.27.0 / 2023-10-24
-------------------

Breaking Changes
++++++++++++++++

New Features
++++++++++++
- (:pr:`326`, :pr:`327`) New protocol option ``occupations_and_eigenvalues`` added to
  ``WavefunctionProperties`` to store lightweight fields.

Enhancements
++++++++++++
- (:pr:`322`) Allow ``util.which`` to raise a clearer error when handling pyenv shims. Improve docs.

Bug Fixes
+++++++++
- (:pr:`325`, :issue:`324`) Ensure ``util.measure_coordinates`` isn't returning NaN angles just
   because floating-point errors are outside arccos's ``[-1, 1]`` bounds.
- (:pr:`315`) Stop resetting numpy print formatting.
- (:pr:`328`) Add workaround for only python v3.12.0 and psi4 (can be expanded) to handle
  ``util.which`` on Windows when a cmd (non-executable) and a cmd.<executable_extension> live
  side-by-side. Otherwise see ``[WinError 193] %1 is not a valid Win32 application``.

Misc.
+++++
- (:pr:`320`) Reset ``black`` formatting to 2022.
- (:pr:`327`) Enable Python v3.12 in poetry.
- (:pr:`328`) Start Windows testing and cron testing.


0.26.0 / 2023-07-31
-------------------

Breaking Changes
++++++++++++++++

- (:pr:`308`) Fix CI Pipelines. Dropped Python3.6. Bring CI pipelines into harmony with local dev experience. Lint and format entire code base. Accelerate CI pipelines. Update setup.py to correctly define extras packages. Breaking change due to dropped support for Python3.6. No code functionality was altered.
   - Dropped support for dead Python 3.6. Minimum supported Python is now 3.7.
   - Updated CONTRIBUTING.md to contain detailed instructions for developers on how to contribute.
   - Fixed broken code that failed to prepend the "v" to version numbers.
   - Updated CI to run without conda and using only packages defined in setup.py. CI is now much faster and runs the same way for local developers and GitHub Actions.
   - Added test.sh and format.sh to devtools/scripts for easy execution of formatting and testing.
   - Formatted all code with black. Sorted imports with isort.
   - Added pre-commit to repo so code formatting, linting, and testing will all run as part of regular git workflow.

Enhancements
++++++++++++
- (:pr:`310`) Modernize DevOps Tooling
   - Added `/scripts` directory to root of project that contains scripts for testing, formatting code, and building docs.
   - Updated build system from `setuptools` to modern `pyproject.toml` specification using `poetry` for the build backend.
   - Removed complicated versioning code in favor of single source of truth in `pyproject.toml`. Using standard library `importlib` for looking up package version in `__init__.py` file.
   - Added `build_docs.sh` script to `/scrips` and removed `Makefile` from `/docs`. Flattened `/docs` file structure.
   - Removed `travis-ci` code from `devtools`
   - Removed LGTM code (they no longer exist as a project).
   - Bring all package directories under `black`, `isort`, and `autoflake` control.

Bug Fixes
+++++++++
- (:pr:`305`) Initialize `Molecule.extras` as empty dictionary.
- (:pr:`311`) Update docs location from RTD to GH pages. Resolve escape char warnings. Update changelog.
- (:pr:`311`) Clear up NumPy "Conversion of an array with ndim > 0 to a scalar is deprecated" in
  `util.measure_coordinates` called by `Molecule.measure`.
- (:pr:`314`) Import `pydantic.v1` from pydantic v2 so that QCElemental can work with any >=1.8.2 pydantic
  until QCElemental is updated for v2.


0.25.1 / 2022-10-31
-------------------

Bug Fixes
+++++++++
- (:pr:`297`) Rearrange imports for compatibility with Pint v0.20. No new restrictions on pint version.


0.25.0 / 2022-06-13
-------------------

Breaking Changes
++++++++++++++++

New Features
++++++++++++

Enhancements
++++++++++++
- (:pr:`285`) Standardized default on ``AtomicResult.native_files`` to ``{}``
  from ``None``. May break strict logic.
- (:pr:`289`, :pr:`290`) Transition from some early documentation tools (class
  ``AutodocBaseSettings`` and ``qcarchive_sphinx_theme``) to externally
  maintained ones (project https://github.com/mansenfranzen/autodoc_pydantic
  and ``sphinx_rtd_theme``). Expand API docs.

Bug Fixes
+++++++++
- (:pr:`286`) Sphinx autodocumentation with typing of
  ``qcelemental.testing.compare_recursive`` no longer warns about circular
  import.


0.24.0 / 2021-11-18
-------------------

New Features
++++++++++++
- (:pr:`275`) ``AtomicResult`` gains a ``native_files`` field of a dictionary of file names (or generic ``'input'``)
  and text (not binary) contents that the CMS program may have generated but which haven't necessarily been
  harvested into QCSchema. Contents controlled by a new native_files protocol analogous to stdout protocol.

Enhancements
++++++++++++
- (:pr:`281`) ``TorsionDriveInput`` now accepts a list of ``Molecule`` objects as the ``initial_molecule`` to seed
  the torsiondrive with multiple conformations.


0.23.0 / 2021-09-23
-------------------

Breaking Changes
++++++++++++++++
- (:pr:`276`) ``AtomicResultProperties.dict()`` no longer forces arrays to JSON flat lists but now
  allows NumPy arrays. That is, ``AtomicResultProperties`` now behaves like every other QCElemental
  model. Expected to be disruptive to QCFractal.
- (:pr:`280`) Examples of QCSchema from test cases are now saved at branch
  https://github.com/MolSSI/QCElemental/tree/qcschema-examples . These have passed validation as
  models by Pydantic and as JSON by schema generated from Pydantic models.

New Features
++++++++++++
- (:pr:`277`) Documentation is now served from https://molssi.github.io/QCElemental/ and built by
  https://github.com/MolSSI/QCElemental/blob/master/.github/workflows/CI.yml .
  https://qcelemental.readthedocs.io/en/latest/ will soon be retired.

Enhancements
++++++++++++
- (:pr:`274`) The molecule ``from_string`` parser when no dtype specified learned to return the most
  specialized error message among the dtypes, not the full input string.
- (:pr:`276`) ``Molecule.to_string(..., dtype="nwchem")`` learned to handle ghosts (``real=False``)
  correctly. It also now prints the user label, which is used downstream for custom basis sets and
  shows up in a NWChem output file. QCEngine will be edited to process the label, but other uses may
  need modification.
- (:pr:`276`) ``Molecule.align`` learned a new keyword ``generic_ghosts=True`` so that it can act on
  molecules that have centers with content Gh, not Gh(He).

Bug Fixes
+++++++++
- (:pr:`276`) ``Molecule.to_string(..., dtype="gamess")`` learned to handle ghosts (``real=False``)
  correctly for ``coord=unique``. Note that QCEngine uses ``coord=prinaxis``, so actual ghosts are
  still NOT interpretable by downstream GAMESS.


0.22.0 / 2021-08-26
-------------------

New Features
++++++++++++
- (:pr:`268`) Add provisional models that store the inputs to and outputs of a torsion drive procedure. @SimonBoothroyd
- (:pr:`272`) Add SCF and return gradient and Hessian fields to ``AtomicResultProperties``.

Enhancements
++++++++++++
- (:pr:`271`) ``Molecule`` learned to create instances with geometry rounded to other than 8 decimal places through ``Molecule(..., geometry_noise=<13>)`` to optionally override ``qcel.models.molecule.GEOMETRY_NOISE = 8``. This should be used sparingly, as it will make more molecules unique in the QCA database. But it is sometimes necessary for accurate finite difference steps and to preserve intrinsic symmetry upon geometry rotation. Previous route was to reset the qcel module variable for the duration of instance creation.
- (:pr:`271`) ``Molecule.align`` and ``Molecule.scramble`` learned to return a fuller copy of self than previously. Now has aligned atom_labels, real, and mass_numbers as well as incidentals like Identifiers. Fragmentation still not addressed.
- (:pr:`271`) ``Molecule.to_string(dtype="gamess")`` learned to write symmetry information to the prinaxis output if passed in through field fix_symmetry. This is provisional, as we'd like the field to be uniform across qcprogs.

Bug Fixes
+++++++++
- (:pr:`271`) Testing function ``compare_values()`` on arrays corrected the RMS maximum o-e value displayed and added a relative value.


0.21.0 / 2021-06-30
-------------------

New Features
++++++++++++
- (:pr:`267`) Serialization learned msgpack mode that, in contrast to msgpack-ext, *doesn't* embed NumPy objects.

Enhancements
++++++++++++
- (:pr:`266`) Testing function ``compare_values()`` learned to print RMS and MAX statistics for arrays.

Bug Fixes
+++++++++
- (:pr:`265`) Fix search path construction. @sheepforce
- (:pr:`266`) Bump minimum pydantic to 1.8.2 to avoid security hole -- https://github.com/samuelcolvin/pydantic/security/advisories/GHSA-5jqp-qgf6-3pvh .


0.20.0 / 2021-05-16
-------------------

New Features
++++++++++++
- (:pr:`257`) ``PhysicalConstantsContext`` learned to write a Fortran header. @awvwgk

Enhancements
++++++++++++
- (:pr:`261`) Documentation became type-aware and grew more links.


0.19.0 / 2021-03-04
-------------------

New Features
++++++++++++

Enhancements
++++++++++++

Bug Fixes
+++++++++
- (:pr:`253`) Fixed incompatibility with Pydantic >=1.8.


0.18.0 / 2021-02-16
-------------------

New Features
++++++++++++
- (:pr:`237`) Exports models to JSON Schema with ``make schema``.
- (:pr:`237`) Build bank of JSON examples from Pydantic models defined in tests. Test that bank against exported schema with ``pytest --validate qcelemental/``.
- (:pr:`237`) Many model descriptions edited, dimensions added to array properties, ``AtomicInput.model.basis`` now takes
  ``BasisSet`` object not just string, several properties added to match QCSchema, several limitations on number and
  uniqueness imposed.

Enhancements
++++++++++++
- (:pr:`237`) Improve mypy conformance including dynamic provenance. Necessitates Pydantic >=1.5.
- (:pr:`237`) ``a0`` without underscore added as computable pint unit.
- (:pr:`246`, :pr:`250`) Removes types deprecated in NumPy v1.20.0.

Bug Fixes
+++++++++
- (:pr:`244`) Fixes where in code validation throws if ``center_data`` missing from ``BasisSet`` model.
- (:pr:`249`) Fixes web tests that weren't marked.


0.17.0 / 2020-10-01
-------------------

Enhancements
++++++++++++
- (:pr:`238`) ``molparse.to_string`` MRChem dtype developed.


0.16.0 / 2020-08-19
-------------------

New Features
++++++++++++

Enhancements
++++++++++++
- (:pr:`231`) ``compare``, ``compare_values``, and ``compare_recursive`` learned new keyword ``equal_phase`` that when
  ``True`` allows either ``computed`` or ``-computed`` to pass the comparison. For ``compare_recursive``, the leniency
  can be restricted to specific leaves of the iterable by passing a list of allowed leaves.

Bug Fixes
+++++++++
- (:pr:`229`) ``Molecule.align`` told the full truth in its documentation that the ``mol_align`` argument can take a float.


0.15.1 / 2020-06-25
-------------------

Bug Fixes
+++++++++
- (:pr:`228`) Fix testing bug for installed module, which was missing a dummy directory.


0.15.0 / 2020-06-25
-------------------

New Features
++++++++++++
- (:pr:`182`) Added experimental protocol for controlling autocorrection attemps. (That is, when a calculation throws a
  known error that QCEngine thinks it can tweak the input and rerun.) Currently in trial for NWChem.

Enhancements
++++++++++++
- (:pr:`186`, :pr:`223`) ``molparse.to_string`` Orca and MADNESS dtypes developed.
- (:pr:`226`) Allow ``which_import`` to distinguish between ordinary and namespace packages.
- (:pr:`227`) Add non-default ``strict`` argument to ``periodictable.to_Z``, ``to_symbol``, and ``to_element`` that fails when isotope information is given.
- (:pr:`227`) Allow nonphysical masses to pass validation in ``molparse.from_schema(..., nonphysical=True)``.
  Also allowed in forming ``qcel.models.Molecule(..., nonphysical=True)``.

Bug Fixes
+++++++++
- (:pr:`227`) Fixed deception described in issue 225 where ``qcel.models.Molecule(..., symbols=["O18"])`` accepted "O18"
  but did not influence the isotope, as user might have expected. That now raises ``NotAnElementError``, and an example
  of correctly setting isotope/masses has been added. This error now caught at ``qcel.molparse.from_arrays`` so general.


0.14.0 / 2020-03-06
-------------------

New Features
++++++++++++

Enhancements
++++++++++++
- (:pr:`211`) Improve testing reliability by excusing PubChem when internet flaky.
- (:pr:`216`) "CODATA2018" constants now tested.
- (:pr:`207`) Multipoles exist in ``AtomicResultProperties`` as ndarray with order-dimensional shape.
  Property ``scf_quadrupole_moment`` defined.

Bug Fixes
+++++++++
- (:pr:`216`) Fixes a bug where "CODATA2018" constants could not be used with ``conversion_factor``.
- (:pr:`217`) Can now run ``.schema()`` on pydantic classes containing ``Array`` fields (allowing ndarray in place of List).


0.13.1 / 2020-02-05
-------------------

New Features
++++++++++++
- (:pr:`209`) Added the option to Hill-order molecular formulas.

Bug Fixes
+++++++++
- (:pr:`208`) Fixes a Molecule hashing issue due to order of operations changes in ``Molecule.from_data``.
  The order of operations changed in ``Molecule.from_data`` and occasionally resulted in different hashes for Molecules
  undergoing orient operations. This issue was introduced in 0.13.0 and is unlikely to have any serious negative effects
  as this did not affect hash integrity.


0.13.0 / 2020-01-29
-------------------

New Features
++++++++++++
- (:pr:`183`, :pr:`187`) Added metadata about DFT functionals (``qcelemental.info.dftfunctionalinfo``).
- (:pr:`184`) Optional PubChem identifiers were added to molecules.
- (:pr:`187`, :pr:`192`, :pr:`195`) Added metadata about CPUs (``qcelemental.info.cpu_info``).

Enhancements
++++++++++++
- (:pr:`179`, :pr:`181`) QCElemental works with Python 3.8 at the expense of needing a new 0.10 pint (rather than generic install).
  Pint 0.10 has optional NumPy dependency of >=1.12.0, so QCElemental that requires both NumPy and pint needs this constraint.
- (:pr:`172`, :pr:`173`, :pr:`202`, :pr:`203`) Contribution improvements, including GitHub Actions checking, ``make format``
  guidance, and updated ``CONTRIBUTING.md``.
- (:pr:`189`, :pr:`196`) Constants and unit conversion based on 2018 CODATA are available (but 2014 remains the default).
- (:pr:`197`, :pr:`199`, :pr:`200`) Added more atomic units and aliases (e.g. ``au_length = bohr``).
- (:pr:`190`, :pr:`191`, :pr:`201`) Slim molecules. Many fields in ``Molecule`` objects may be optionally inferred.

Bug Fixes
+++++++++
- (:pr:`174`) Fix ``compare_recursive`` for when ``expected`` is a list but ``computed`` is not.
- (:pr:`177`) Spelling and type hint fixes.
- (:pr:`180`) Better test coverage.


0.12.0 / 2019-11-13
-------------------

New Features
++++++++++++

Enhancements
++++++++++++
- (:pr:`156`) ``Molecules`` can now be correctly compared with ``==``.
- (:pr:`157`) ``molparse.to_string`` Q-Chem dtype developed. Psi4 dtype now includes label and doesn't have extraneous info for single fragment systems.
- (:pr:`162`) New protocol ``stdout`` added to ``ResultProtocols`` controlling whether ``stdout`` field (which generally
  contains the primary logfile, whether a program writes it to file or stdout) is returned.
- (:pr:`165`) The code base is now Black formatted.

Deprecations
++++++++++++
- (:pr:`156`) ``Molecule.compare`` is deprecated and will be removed in v0.13.0.
- (:pr:`167`, :pr:`168`) ``ResultInput``, ``Result``, ``Optimization`` have been removed in favor of ``AtomicInput``, ``AtomicResult``, and ``OptimizationResult`` and will be removed in v0.13.0.

Bug Fixes
+++++++++
- (:pr:`170`) ``ProtoModel`` subclasses now correctly allow custom ``__repr__`` and ``__str__`` methods.
- (:pr:`164`, :pr:`166`) ``nglview-sdf`` molecule string format now correctly uses correct sdf format widths fixing some issues with very large molecules.


0.11.1 / 2019-10-28
-------------------

Bug Fixes
+++++++++
- (:pr:`152`) Patches ``Molecule.from_file`` and ``Molecule.from_data`` to read XYZ+ format and correctly handle keyword arguments.
   Patches ``Molecule.to_file`` to write XYZ+ format as the default for XYZ and XYZ+ files.

0.11.0 / 2019-10-24
-------------------

Enhancements
++++++++++++
- (:pr:`147`) Updates Pydantic to the 1.0 release and fixes a number of breaking changes.
- (:pr:`148`) Switches from Py3dMoljs to NGLView for molecular visualization due to Jupyter Widget integration.
- (:pr:`149`) Adds statC and Debye to the units registry.

Bug Fixes
+++++++++
- (:pr:`150`) Patches ``which_import`` to correctly handle submodules.

0.10.0 / 2019-10-16
-------------------

Enhancements
++++++++++++
- (:pr:`144`) Allows `which_import` to handle submodules.
- (:pr:`143`) Allow testing complex numbers.


0.9.0 / 2019-10-01
------------------

New Features
++++++++++++
- (:pr:`137`, :pr:`138`) Coordinates can now be output in ``Turbomole`` format in addition to all other formats.
- (:pr:`139`) A wavefunction property have been added to the ``Result`` Model. Adds the ability for Engine and other
  programs to store and fetch wavefunction data.
- (:pr:`140`) ``Protocols`` have been added to ``QCInputSpecification`` which allows data to pre-pruned by different
  specifications. Main intention is to reduce wavefunction data which may be re-computed cheaply rather than storing
  all of it. This does change the input model, so requires a minor version bump.

Enhancements
++++++++++++
- (:pr:`132`) ``BasisSet`` and ``Result``'s documentations have been brought up to the standards of other models.

0.8.0 / 2019-09-13
------------------

New Features
++++++++++++
- (:pr:`123`) QCElemental now passes MyPy!
- (:pr:`127`, :pr:`131`) Adds van der Waals radii data available through ``vdwradii.get(atom)`` function.

Enhancements
++++++++++++

Bug Fixes
+++++++++
- (:pr:`125`) Add back a consistency check that had been optimized out.

0.7.0 / 2019-08-23
------------------

Enhancements
++++++++++++

- (:pr:`118`) Model string representations should be more user friendly and descriptive without overload the
  output.
- (:pr:`119`) The ``molparse.to_string`` keyword-arg ``return_data`` now returns molecule keywords for GAMESS and
  NWChem. The ``models.Molecule.to_string`` can use ``return_data`` now, too.
- (:pr:`120`) Auto documentation tech is now built into the ``ProtoModel`` and does not need
  an external function.

0.6.1 / 2019-08-19
------------------

Bug Fixes
+++++++++

- (:pr:`114`) The Numpy einsum calls reference the top level functions and not core C functions. This fixes an issue
  which can result in NumPy version dependencies.

0.6.0 / 2019-08-14
------------------

New Features
++++++++++++

- (:pr:`85`, :pr:`87`) Msgpack is a new serialization option for Models. Serialization defaults to msgpack when
  available (``conda install msgpack-python [-c conda-forge]``), falling back to JSON otherwise. This results in
  substantial speedups for both serialization and deserialization actions and should be a transparent replacement for
  users within Elemental itself.

Enhancements
++++++++++++

- (:pr:`78`) Molecular alignments can now be aligned on the derivatives of vector components.
- (:pr:`81`) Testing is now operated both on the minimal supported and the latest released versions of dependencies.
- (:pr:`82`) Molecule fragment grouping is now disabled by default to match expected behavior.
- (:pr:`84`) Testing without internet connection should now pass since PubChem testing is skipped with no connection.
- (:pr:`85`) Molecule switches from lists to numpy arrays for internal storage of per-atom fields.
- (:pr:`86`) Molecule performance and memory enhancements through reduced validation times and LRU caching of
  common validations.
- (:pr:`88`, :pr:`109`) The ``Molecule`` Model now has its attributes documented and in an on-the-fly manner derived
  from the Pydantic Schema of those attributes.
- (:pr:`99`, :pr:`100`, :pr:`101`, :pr:`102`, :pr:`103`, :pr:`104`, :pr:`105`, :pr:`106`, :pr:`107`) Various
  documentation, type hints, and small changes.

Bug Fixes
+++++++++

- (:pr:`87`) Molecule objects built from Schema are run through validators for consistency.


0.5.0 / 2019-07-16
------------------

Enhancements
++++++++++++

- (:pr:`76`) Adds a built-in ``Molecule.to_file`` function for easy serialization into ``.numpy``, ``.json``, ``.xyz``,
  ``.psimol``, and ``.psi4`` file formats.

Bug Fixes
+++++++++

- (:pr:`74`) Atom and fragment ordering are preserved when invoking ``get_fragment``.


0.4.2 / 2019-06-13
------------------

New Features
++++++++++++

- (:pr:`70`, :pr:`72`) ``molparse.to_string`` Molpro dtype developed.


0.4.1 / 2019-05-31
------------------

New Features
++++++++++++

Enhancements
++++++++++++

- (:pr:`68`) ``molparse.to_string`` learned parameter ``return_data`` that contains aspects of the
  ``models.Molecule`` not expressible in the string. Implemented for dtypes xyz, cfour, psi4.
- (:pr:`68`) ``Datum`` gained an attribute ``numeric`` that reflects whether arithmetic on ``data``
  is valid. ``Datum``\ s that aren't numeric can now be created by initializing with ``numeric=False``.

Bug Fixes
+++++++++

- (:pr:`66`) Fix tests when `networkx` not installed.
- (:pr:`67`) Fix "unsupported format string passed to numpy.ndarray.__format__" on Mac for ``testing.compare_values``.


0.4.0 / 2019-05-13
------------------

New Features
++++++++++++

- (:pr:`51`) Changes ``models.Molecule`` connectivity to default to `None` rather than an empty list. **WARNING** this
  change alters the hashes produced from the ``Molecule.get_hash`` functionality.
- (:pr:`52`, :pr:`53`) ``models.Molecule`` learned ``nuclear_repulsion_energy``, ``nelectrons``, and
  ``to_string`` functions.
- (:pr:`54`) ``models.ResultProperties`` supports CCSD and CCSD(T) properties.
- (:pr:`56`) Algorithms Kabsch ``molutil.kabsch_align``, Hungarian ``util.linear_sum_assignment``, and Uno ``util.uno``
  added. Utilities to generate random 3D rotations ``util.random_rotation_matrix`` and reindex a NumPy array into
  smaller blocks ``util.blockwise_expand`` added.
- (:pr:`56`) Molecular alignment taking into account displacement, rotation, atom exchange, and mirror symmetry for
  superimposable and differing geometries was added in ``molutil.B787`` (basis NumPy function) and
  ``models.Molecule.align`` (far more convenient). Suitable for QM-sized molecules. Requires addition package
  ``networkx``.
- (:pr:`58`) ``utils`` learned ``which_import`` and ``which`` that provide a path or boolean result
  for locating modules or commands, respectively. These were migrated from QCEngine along with
  ``safe_version`` and ``parse_version`` to colocate the import utilities.
- (:pr:`61`) Add molecular visualization to the ``models.Molecule`` object through the optional 3dMol.js framework.
- (:pr:`65`) ``testing.compare_molrecs`` learned parameter ``relative_geoms='align'`` that lets Molecules pass if
  geometries within a translation and rotation of each other.
- (:pr:`65`) ``testing.compare_recursive`` learned parameter ``forgive`` that is a list of paths that may differ without
  failing the comparison.

Enhancements
++++++++++++

- (:pr:`52`, :pr:`53`) ``molparse.to_string`` NWChem and GAMESS dtypes developed.
- (:pr:`57`) ``molparse.to_string`` learned ``dtype='terachem'`` for writing the separate XYZ file
  required by TeraChem. Angstroms or Bohr allowed, though the latter requires extra in input file.
- (:pr:`60`) ``util.which`` added the Python interpreter path to the default search ``$PATH``.
- (:pr:`62`) Added ``*`` to parameter list of many functions requiring subsequent to be keyword only. Code relying
  heavily on positional arguments may get broken.
- (:pr:`63`) ``util.which`` learned parameter ``env`` to use an alternate search ``$PATH``.
- (:pr:`63`) ``util.which`` and ``util.which_import`` learned parameters ``raise_error`` and ``raise_msg`` which raises
  ``ModuleNotFoundError`` (for both functions) when not located. It error will have a generic error message which can
  be extended by ``raise_msg``. It is strongly encouraged to add specific remedies (like how to install) through this
  parameter. This is the third exit pattern possible from the "which" functions, of which path/None is the default,
  True/error happens when ``raise_error=True``, and True/False happens otherwise when ``return_bool=True``.
- (:pr:`65`) Testing functions ``compare``, ``compare_values``, ``compare_recursive`` learned parameter
  ``return_handler`` that lets other printing, logging, and pass/fail behavior to be interjected.

Bug Fixes
+++++++++

- (:pr:`63`) ``util.which`` uses ``os.pathsep`` rather than Linux-focused ``:``.
- (:pr:`65`) Fixed some minor printing and tolerance errors in molecule alignment.
- (:pr:`65`) ``testing.compare_recursive`` stopped doing ``atol=10**-atol`` for ``atol>=1``, bringing it in line with
  other compare functions.


0.3.3 / 2019-03-12
------------------

Enhancements
++++++++++++

- (:pr:`49`) Precompute some mass number and mass lookups and store on ``qcel.periodic_table``. Also move
  static ``re.compile`` expressions out of fns on to module. Mol validation .127s --> .005s.


0.3.2 / 2019-03-11
------------------

New Features
++++++++++++

- (:pr:`47`) ``models.DriverEnum`` now has a ``derivative_int`` function to return 1 for ``gradient``, etc.,
  for easy math. ``properties`` returns 0.
- (:pr:`47`) Optional ``fix_symmetry`` field in qcschema_molecule was missing from ``models.Molecule`` so
  Pydantic got mad at Psi4. Now calmed.

Enhancements
++++++++++++

- (:pr:`48`) If Molecule object has passed through molparse validation because it was created with a molparse
  constructor (e.g., ``from_string``), save some time by not passing it through again at ``model.Molecule``
  creation time.

Bug Fixes
+++++++++

- (:pr:`48`) Fixed a ``Molecule.get_fragment`` bug where ghosted fragments still asserted charge/multiplicity
  to the validator, which was rightly confused.


0.3.1 / 2019-03-07
------------------

Enhancements
++++++++++++

- (:pr:`37`) Documentation now pulls from the custom QC Archive Sphinx Theme, but can fall back to the standard
  RTD theme. This allows all docs across QCA to appear consistent with each other.
- (:pr:`41`) Conda-build recipe removed to avoid possible confusion for everyone who isn't a Conda-Forge
  recipe maintainer. Tests now rely on the ``conda env`` setups.
- (:pr:`44`) Molecule objects are now always validated against a more rigorous model and fragment multiplicities are
  fixed at the correct times, even when no multiplicities are provided. Molecule defaults to ``dtype=2``.


Bug Fixes
+++++++++

- (:pr:`39`) Fixed ``setup.py`` to call ``pytest`` instead of ``unittest`` when running tests on install
- (:pr:`41`) Pinned a minimum Pytest version to make sure errors are not because of too old of a pytest version


0.3.0 / 2019-02-27
------------------

New Features
++++++++++++

- (:pr:`33`) ``molparse.to_schema`` recognizes ``dtype=2`` in keeping with
  GH:MolSSI/QCSchema#60 with internal ``schema_name=qcschema_molecule`` and
  ``schema_version=2`` fields. ``molparse.from_schema`` recognizes external
  fields (existing functionality), internal fields (dtype=2), and mixed.
- (:pr:`33`) Pydantic molecule model now contains schema_name and schema_version=2 information.
- (:pr:`35`) Models now have an ``extra`` field for extra attributes, no additional base keys are allowed.


Enhancements
++++++++++++

- (:pr:`34`) Converts ``qcel.Datum`` to Pydantic model. Changes:
  (a) comment, doi, glossary fields must be accessed by keyword,
  (b) ``to_dict()`` becomes ``dict()`` and instead of only label, units,
  data fields in dict, now comment, doi, glossary present _if_ non-default,
  (c) complex values no longer list-ified by ``to_dict()``.
- (:pr:`36`) Changelog and Models documentation.

Bug Fixes
+++++++++


0.2.6 / 2019-02-18
------------------

Bug Fixes
+++++++++

- (:pr:`32`) Updates compliance with Pydantic v0.20.


0.2.5 / 2019-02-13
------------------

Enhancements
++++++++++++

- (:pr:`31`) Lints the code base preparing for a release and minor test improvements.

Bug Fixes
+++++++++

- (:pr:`30`) Fixes ``dihedral`` measurement code for incorrect phase in certain quadrants.


0.2.4 / 2019-02-08
------------------

New Features
++++++++++++

- (:pr:`27`) Adds a new ``measure`` feature to Molecule for distances, angles, and dihedrals.
- (:pr:`25`) Adds a new ``testing`` module which contains testing routines for arrays, dictionaries, and molecules.

Enhancements
++++++++++++

- (:pr:`28`) Reduces loading time from ~1 second to 200 ms by delaying ``pint`` import and ensuring git tags are only computed once.


0.2.3 / 2019-01-29
------------------

Enhancements
++++++++++++

- (:pr:`24`) Update models to be compatible with QCFractal and MongoDB objects in the QCArchive Ecosystem.
  Also enhances the ``Molecule`` model's ``json`` function to accept ``as_dict`` keyword, permitting a return as a
  dictionary of Pydantic-serialized python (primitive) objects, instead of a string.


0.2.2 / 2019-01-28
------------------

Bug Fixes
+++++++++

- (:pr:`21`) Molparse's ``from_schema`` method now correctly parses the new ``qcschema_X`` strings for schema names.
- (:pr:`23`) Pydantic model serializations now correctly handle Numpy Array objects in nested ``BaseModels``. Model serialization testing added to catch these in the future.


0.2.1 / 2019-01-27
------------------

- (:pr:`20`) Moves several Molecule parsing functions to the molparse module.


0.2.0 / 2019-01-25
------------------

- now requires Python 3.6+
- now requires Pydantic

New Features
++++++++++++

- (:pr:`14`, :pr:`16`, :pr:`17`) Added new Pydantic models for Molecules, Results, and Optimizations to make common objects used in the QCArchive project all exist in one central, always imported module.

Enhancements
++++++++++++

- (:pr:`13`) Function ``util.unnp`` that recursively list-ifies ndarray in a dict now handles lists and flattens.


0.1.3 / 2018-12-14
------------------

New Features
++++++++++++

- (:pr:`12`) Adds "connectivity" validation and storage consistent with QCSchema.

Enhancements
++++++++++++

- (:pr:`12`) Adds single dictionary provenance consistent with `QCSchema <https://github.com/MolSSI/QCSchema/blob/master/qcschema/dev/definitions.py>`_ (line 23) rather than previous list o'dicts.


0.1.2 / 2018-11-3
-----------------

New Features
++++++++++++

- (:pr:`10`) Adds covalent radii data available through ``covalentradii.get(atom)`` function.
- (:pr:`10`) Adds ``to_units(unit)`` to ``Datum`` class to access the data in non-native units.
- (:pr:`10`) Adds ``periodictable.to_period(atom)`` and ``to_group(atom)`` functions to address periodic table.


0.1.1 / 2018-10-30
------------------

New Features
++++++++++++

- (:pr:`7`, :pr:`9`) Adds "comment" and "provenance" fields to internal repr to better match QCSchema.
- (:pr:`7`) Adds provenance stamp to ``from_string``, ``from_arrays``, ``from_schema`` functions.

Enhancements
++++++++++++

- (:pr:`7`) Adds outer schema_name/schema_version to ``to_schema(..., dtype=1)`` output so is inverse to ``from_schema``.

Bug Fixes
+++++++++

- (:pr:`8`) Tests pass for installed module now that comparison tests are xfail.


0.1.0a / 2018-10-24
-------------------

This is the first alpha release of QCElemental containing the primary three components.

New Features
++++++++++++

- (:pr:`6`) Updated molparse to write new Molecule QCSchema fields in keeping with GH:MolSSI/QCSchema#44
- Periodic Table data from NIST SRD144 (c. pre-2015?) collected into ``qcelemental.periodictable`` instance,
  with accessors ``to_Z``, ``to_element``, ``to_E``, ``to_mass``, ``to_A`` (and redundant accessors ``to_mass_number``,
  ``to_atomic_number``, ``to_symbol``, ``to_name``) in ``float`` and ``Decimal`` formats. Also includes functionality
  to write a corresponding "C" header.
- Physical Constants data from NIST SRD121 (CODATA 2014) collected into ``qcelemental.constants`` instance,
  with access through ``qcelemental.constants.Faraday_constant`` (exact capitalization; ``float`` result) or
  ``get`` (free capitalization; ``float`` or ``Decimal`` result). Also includes functionality to write a
  corresponding "C" header.
- ``molparse`` submodule where ``from_string``, ``from_array``, ``from_schema`` constructors parse and rearrange
  (if necessary) and validate molecule topology inputs from the QC and EFP domains into a QCSchema-like
  data structure. Current deficiencies from QCSchema are non-contiguous fragments and "provenance" fields.
  Accessors ``to_string`` and ``to_schema`` are highly customizable.
- A `pint <https://pint.readthedocs.io/en/latest/>`_ context has been built around the NIST physical constants
  data so that ``qcelemental.constants.conversion_factor(from_unit, to_unit)`` uses the QCElemental values
  in its conversions. Resulting ``float`` is within uncertainty range of NIST constants but won't be exact
  for conversions involving multiple fundamental dimensions or ``wavelength -> energy != 1 / (energy -> wavelength)``.
