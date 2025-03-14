import abc
import typing
from dataclasses import field, dataclass
from unittest.mock import Mock

import numpy as np
from pyqrack import QrackSimulator
from kirin.interp import Interpreter
from typing_extensions import Self
from bloqade.pyqrack.reg import Measurement
from kirin.interp.exceptions import InterpreterError


class PyQrackOptions(typing.TypedDict):
    qubitCount: int
    isTensorNetwork: bool
    isSchmidtDecomposeMulti: bool
    isSchmidtDecompose: bool
    isStabilizerHybrid: bool
    isBinaryDecisionTree: bool
    isPaged: bool
    isCpuGpuHybrid: bool
    isOpenCL: bool


def _default_pyqrack_args() -> PyQrackOptions:
    return PyQrackOptions(
        qubitCount=-1,
        isTensorNetwork=False,
        isSchmidtDecomposeMulti=True,
        isSchmidtDecompose=True,
        isStabilizerHybrid=True,
        isBinaryDecisionTree=True,
        isPaged=True,
        isCpuGpuHybrid=True,
        isOpenCL=True,
    )


@dataclass
class MemoryABC(abc.ABC):
    pyqrack_options: PyQrackOptions = field(default_factory=_default_pyqrack_args)
    sim_reg: "QrackSimulator" = field(init=False)

    @abc.abstractmethod
    def allocate(self, n_qubits: int) -> tuple[int, ...]:
        """Allocate `n_qubits` qubits and return their ids."""
        ...

    def reset(self):
        """Reset the memory, releasing all qubits."""
        # do not reset the simulator it might be used by
        # results of the simulation
        self.sim_reg = QrackSimulator(**self.pyqrack_options)


@dataclass
class MockMemory(MemoryABC):
    """Mock memory for testing purposes."""

    allocated: int = field(init=False, default=0)

    def allocate(self, n_qubits: int):
        allocated = self.allocated + n_qubits
        result = tuple(range(self.allocated, allocated))
        self.allocated = allocated
        return result

    def reset(self):
        self.allocated = 0
        self.sim_reg = Mock()


@dataclass
class StackMemory(MemoryABC):
    total: int = field(kw_only=True)
    allocated: int = field(init=False, default=0)

    def allocate(self, n_qubits: int):
        curr_allocated = self.allocated
        self.allocated += n_qubits

        if self.allocated > self.total:
            raise InterpreterError(
                f"qubit allocation exceeds memory, "
                f"{self.total} qubits, "
                f"{self.allocated} allocated"
            )

        return tuple(range(curr_allocated, self.allocated))

    def reset(self):
        super().reset()
        self.allocated = 0


@dataclass
class DynamicMemory(MemoryABC):
    def __post_init__(self):
        self.reset()

        if self.sim_reg.is_tensor_network:
            raise ValueError("DynamicMemory does not support tensor networks")

    def allocate(self, n_qubits: int):
        start = self.sim_reg.num_qubits()
        for i in range(start, start + n_qubits):
            self.sim_reg.allocate_qubit(i)

        return tuple(range(start, start + n_qubits))


@dataclass
class PyQrackInterpreter(Interpreter):
    keys = ["pyqrack", "main"]
    memory: MemoryABC = field(kw_only=True)
    rng_state: np.random.Generator = field(
        default_factory=np.random.default_rng, kw_only=True
    )
    loss_m_result: Measurement = field(default=Measurement.One, kw_only=True)
    """The value of a measurement result when a qubit is lost."""

    def initialize(self) -> Self:
        super().initialize()
        self.memory.reset()  # reset allocated qubits
        return self
