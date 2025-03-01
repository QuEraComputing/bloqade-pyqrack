import typing
from dataclasses import field, dataclass

import numpy as np
from kirin.interp import Interpreter
from typing_extensions import Self
from bloqade.pyqrack.reg import Measurement

if typing.TYPE_CHECKING:
    from pyqrack import QrackSimulator


@dataclass
class Memory:
    total: int
    allocated: int
    sim_reg: "QrackSimulator"


@dataclass
class PyQrackInterpreter(Interpreter):
    keys = ["pyqrack", "main"]
    memory: Memory = field(kw_only=True)
    rng_state: np.random.Generator = field(
        default_factory=np.random.default_rng, kw_only=True
    )
    loss_m_result: Measurement = field(default=Measurement.One, kw_only=True)
    """The value of a measurement result when a qubit is lost."""

    def initialize(self) -> Self:
        super().initialize()
        self.memory.allocated = 0  # reset allocated qubits
        return self
