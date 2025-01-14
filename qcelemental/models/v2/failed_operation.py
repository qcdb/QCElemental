from typing import TYPE_CHECKING, Any, Dict, Literal, Optional, Sequence, Tuple, Union

from pydantic import Field, field_validator

from .basemodels import ProtoModel, check_convertible_version

if TYPE_CHECKING:
    import qcelemental

    ReprArgs = Sequence[Tuple[Optional[str], Any]]


class ComputeError(ProtoModel):
    """Complete description of the error from an unsuccessful program execution."""

    error_type: str = Field(  # type: ignore
        ...,  # Error enumeration not yet strict
        description="The type of error which was thrown. Restrict this field to short classifiers e.g. 'input_error'. "
        "Suggested classifiers: https://github.com/MolSSI/QCEngine/blob/master/qcengine/exceptions.py",
    )
    error_message: str = Field(  # type: ignore
        ...,
        description="Text associated with the thrown error. This is often the backtrace, but it can contain additional "
        "information as well.",
    )
    extras: Optional[Dict[str, Any]] = Field(  # type: ignore
        None,
        description="Additional information to bundle with the error.",
    )

    model_config = ProtoModel._merge_config_with(repr_style=["error_type", "error_message"])

    def __repr_args__(self) -> "ReprArgs":
        return [("error_type", self.error_type), ("error_message", self.error_message)]


class FailedOperation(ProtoModel):
    """
    Record indicating that a given operation (program, procedure, etc.) has failed
    and containing the reason and input data which generated the failure.
    """

    schema_name: Literal["qcschema_failed_operation"] = Field(
        "qcschema_failed_operation",
        description=(
            f"The QCSchema specification this model conforms to. Explicitly fixed as qcschema_failed_operation."
        ),
    )
    schema_version: Literal[2] = Field(
        2,
        description="The version number of :attr:`~qcelemental.models.FailedOperation.schema_name` to which this model conforms.",
    )
    id: Optional[str] = Field(  # type: ignore
        None,
        description="A unique identifier which links this FailedOperation, often of the same Id of the operation "
        "should it have been successful. This will often be set programmatically by a database such as "
        "Fractal.",
    )
    input_data: Any = Field(  # type: ignore
        None,
        description="The input data which was passed in that generated this failure. This should be the complete "
        "input which when attempted to be run, caused the operation to fail.",
    )
    success: Literal[False] = Field(  # type: ignore
        False,
        description="A boolean indicator that the operation failed consistent with the model of successful operations. "
        "Should always be False. Allows programmatic assessment of all operations regardless of if they failed or "
        "succeeded",
    )
    error: ComputeError = Field(  # type: ignore
        ...,
        description="A container which has details of the error that failed this operation. See the "
        ":class:`ComputeError` for more details.",
    )
    extras: Optional[Dict[str, Any]] = Field(  # type: ignore
        {},
        description="Additional information to bundle with the failed operation. Details which pertain specifically "
        "to a thrown error should be contained in the `error` field. See :class:`ComputeError` for details.",
    )

    def __repr_args__(self) -> "ReprArgs":
        return [("error", self.error)]

    def convert_v(
        self, target_version: int, /
    ) -> Union["qcelemental.models.v1.FailedOperation", "qcelemental.models.v2.FailedOperation"]:
        """Convert to instance of particular QCSchema version."""
        import qcelemental as qcel

        if check_convertible_version(target_version, error="FailedOperation") == "self":
            return self

        dself = self.model_dump()
        if target_version == 1:
            dself.pop("schema_name")
            dself.pop("schema_version")

            self_vN = qcel.models.v1.FailedOperation(**dself)
        else:
            assert False, target_version

        return self_vN
