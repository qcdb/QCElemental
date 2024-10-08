from . import types
from .align import AlignmentMill
from .basemodels import AutodocBaseSettings  # remove when QCFractal merges `next`
from .basemodels import ProtoModel
from .basis import BasisSet
from .common_models import ComputeError, DriverEnum, FailedOperation, Model, Provenance
from .molecule import Molecule
from .procedures import Optimization  # scheduled for removal
from .procedures import (
    OptimizationInput,
    OptimizationResult,
    OptimizationSpecification,
    QCInputSpecification,
    TDKeywords,
    TorsionDriveInput,
    TorsionDriveResult,
)
from .results import Result  # scheduled for removal
from .results import ResultInput  # scheduled for removal
from .results import ResultProperties  # scheduled for removal
from .results import AtomicInput, AtomicResult, AtomicResultProperties, AtomicResultProtocols


def qcschema_models():
    return [
        AtomicInput,
        AtomicResult,
        AtomicResultProperties,
        BasisSet,
        Molecule,
        Provenance,
    ]