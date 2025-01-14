import pytest

from .addons import drop_qcsk, schema_versions


def test_result_properties_default_skip(request, schema_versions):
    AtomicResultProperties = (
        schema_versions.AtomicProperties if ("v2" in request.node.name) else schema_versions.AtomicResultProperties
    )

    obj = AtomicResultProperties(scf_one_electron_energy="-5.0")
    drop_qcsk(obj, request.node.name)

    assert pytest.approx(obj.scf_one_electron_energy) == -5.0

    assert obj.dict().keys() == {"scf_one_electron_energy"}


def test_result_properties_default_repr(request, schema_versions):
    AtomicResultProperties = (
        schema_versions.AtomicProperties if ("v2" in request.node.name) else schema_versions.AtomicResultProperties
    )

    obj = AtomicResultProperties(scf_one_electron_energy="-5.0")
    assert "none" not in str(obj).lower()
    assert "scf_one_electron_energy" in str(obj)
    assert len(str(obj)) < 100
    assert len(repr(obj)) < 100


def test_repr_provenance(request, schema_versions):
    Provenance = schema_versions.Provenance

    prov = Provenance(creator="qcel", version="v0.3.2")
    drop_qcsk(prov, request.node.name)

    assert "qcel" in str(prov)
    assert "qcel" in repr(prov)


def test_repr_compute_error(schema_versions):
    ComputeError = schema_versions.ComputeError

    ce = ComputeError(error_type="random_error", error_message="this is bad")

    assert "random_error" in str(ce)
    assert "random_error" in repr(ce)


def test_repr_failed_op(schema_versions):
    ComputeError = schema_versions.ComputeError
    FailedOperation = schema_versions.FailedOperation

    fail_op = FailedOperation(error=ComputeError(error_type="random_error", error_message="this is bad"))
    assert (
        str(fail_op)
        == """FailedOperation(error=ComputeError(error_type='random_error', error_message='this is bad'))"""
    )


def test_repr_result(request, schema_versions):
    AtomicInput = schema_versions.AtomicInput

    if "v2" in request.node.name:
        result = AtomicInput(
            **{
                "specification": {"driver": "gradient", "model": {"method": "UFF"}},
                "molecule": {"symbols": ["He"], "geometry": [0, 0, 0]},
            }
        )
    else:
        result = AtomicInput(
            **{"driver": "gradient", "model": {"method": "UFF"}, "molecule": {"symbols": ["He"], "geometry": [0, 0, 0]}}
        )
    drop_qcsk(result, request.node.name)
    assert "molecule_hash" in str(result)
    assert "molecule_hash" in repr(result)
    assert "'gradient'" in str(result)


def test_repr_optimization(schema_versions, request):
    OptimizationInput = schema_versions.OptimizationInput

    if "v2" in request.node.name:
        optin = {
            "specification": {"specification": {"driver": "gradient", "model": {"method": "UFF"}}},
            "initial_molecule": {"symbols": ["He"], "geometry": [0, 0, 0]},
        }
    else:
        optin = {
            "input_specification": {"driver": "gradient", "model": {"method": "UFF"}},
            "initial_molecule": {"symbols": ["He"], "geometry": [0, 0, 0]},
        }
    opt = OptimizationInput(**optin)

    assert "molecule_hash" in str(opt)
    assert "molecule_hash" in repr(opt)


def test_model_custom_repr(schema_versions):
    ProtoModel = schema_versions.ProtoModel

    class Model(ProtoModel):
        a: int

        def __repr__(self) -> str:
            return "Hello world!"

    m = Model(a=5)
    assert repr(m) == "Hello world!"
    assert "Model(" in str(m)

    class Model2(ProtoModel):
        a: int

        def __str__(self) -> str:
            return "Hello world!"

    m = Model2(a=5)
    assert "Model2(" in repr(m)
    assert str(m) == "Hello world!"
