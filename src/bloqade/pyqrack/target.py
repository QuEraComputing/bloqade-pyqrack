from typing import List, TypeVar, ParamSpec, TypedDict
from dataclasses import field, dataclass

from kirin import ir
from pyqrack import QrackSimulator
from kirin.passes import Fold
from bloqade.pyqrack.base import StackMemory, DynamicMemory, PyQrackInterpreter
from bloqade.analysis.address import AnyAddress, AddressAnalysis


class PyQrackOptions(TypedDict):
    isTensorNetwork: bool
    isSchmidtDecomposeMulti: bool
    isSchmidtDecompose: bool
    isStabilizerHybrid: bool
    isBinaryDecisionTree: bool
    isPaged: bool
    isCpuGpuHybrid: bool
    isOpenCL: bool
    isHostPointer: bool


def _default_pyqrack_args() -> PyQrackOptions:
    return {
        "isTensorNetwork": False,
        "isOpenCL": False,
    }


Params = ParamSpec("Params")
RetType = TypeVar("RetType")


@dataclass
class PyQrack:
    """PyQrack target runtime for Bloqade."""

    min_qubits: int = 0
    """Minimum number of qubits required for the PyQrack simulator.
    Useful when address analysis fails to determine the number of qubits.
    """
    dynamic_qubits: bool = False
    """Whether to use dynamic qubit allocation. Cannot use with tensor network simulations."""

    pyqrack_options: PyQrackOptions = field(default_factory=_default_pyqrack_args)
    """Options to pass to the QrackSimulator object, node `qubitCount` will be overwritten."""

    def _get_interp(self, mt: ir.Method[Params, RetType]):
        if self.dynamic_qubits:
            return PyQrackInterpreter(
                mt.dialects,
                memory=DynamicMemory(
                    QrackSimulator(qubitCount=0, **self.pyqrack_options)
                ),
            )
        else:
            address_analysis = AddressAnalysis(mt.dialects)
            frame, _ = address_analysis.run_analysis(mt)
            if self.min_qubits == 0 and any(
                isinstance(a, AnyAddress) for a in frame.entries.values()
            ):
                raise ValueError(
                    "All addresses must be resolved. Or set min_qubits to a positive integer."
                )

            num_qubits = max(address_analysis.qubit_count, self.min_qubits)
            self.pyqrack_options.pop("qubitCount", None)
            memory = StackMemory(
                total=num_qubits,
                allocated=0,
                sim_reg=QrackSimulator(qubitCount=num_qubits, **self.pyqrack_options),
            )

            return PyQrackInterpreter(mt.dialects, memory=memory)

    def run(
        self,
        mt: ir.Method[Params, RetType],
        *args: Params.args,
        **kwargs: Params.kwargs,
    ) -> RetType:
        """Run the given kernel method on the PyQrack simulator.

        Args
            mt (Method):
                The kernel method to run.

        Returns
            The result of the kernel method, if any.

        """
        fold = Fold(mt.dialects)
        fold(mt)
        return self._get_interp(mt).run(mt, args, kwargs).expect()

    def multi_run(
        self,
        mt: ir.Method[Params, RetType],
        _shots: int,
        *args: Params.args,
        **kwargs: Params.kwargs,
    ) -> List[RetType]:
        """Run the given kernel method on the PyQrack `_shots` times, caching analysis results.

        Args
            mt (Method):
                The kernel method to run.
            _shots (int):
                The number of times to run the kernel method.

        Returns
            List of results of the kernel method, one for each shot.

        """
        fold = Fold(mt.dialects)
        fold(mt)

        interpreter = self._get_interp(mt)
        batched_results = []
        for _ in range(_shots):
            batched_results.append(interpreter.run(mt, args, kwargs).expect())

        return batched_results
