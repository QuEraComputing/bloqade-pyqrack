import enum
from typing import List, Generic, TypeVar
from dataclasses import dataclass

from bloqade.qasm2.types import QReg, Qubit


class CRegister(list[bool]):
    """Runtime representation of a classical register."""

    def __init__(self, size: int):
        super().__init__(False for _ in range(size))


@dataclass(frozen=True)
class CBitRef:
    """Object representing a reference to a classical bit."""

    ref: CRegister
    """The classical register that is holding this bit."""

    pos: int
    """The position of this bit in the classical register."""

    def set_value(self, value: bool):
        self.ref[self.pos] = value

    def get_value(self):
        return self.ref[self.pos]


class QubitState(enum.Enum):
    Active = enum.auto()
    Lost = enum.auto()


SimRegType = TypeVar("SimRegType")


@dataclass(frozen=True)
class SimQReg(QReg, Generic[SimRegType]):
    """Simulation runtime value of a quantum register."""

    size: int
    """The number of qubits in this register."""

    sim_reg: SimRegType
    """The register of the simulator."""

    addrs: tuple[int, ...]
    """The global addresses of the qubits in this register."""

    qubit_state: List[QubitState]
    """The state of each qubit in this register."""

    def drop(self, pos: int):
        """Drop the qubit at the given position in-place.

        Args
            pos (int): The position of the qubit to drop.

        """
        assert self.qubit_state[pos] is QubitState.Active, "Qubit already lost"
        self.qubit_state[pos] = QubitState.Lost

    def __getitem__(self, pos: int):
        return SimQubit(self, pos)


@dataclass(frozen=True)
class SimQubit(Qubit, Generic[SimRegType]):
    """The runtime representation of a qubit reference."""

    ref: SimQReg[SimRegType]
    """The quantum register that is holding this qubit."""

    pos: int
    """The position of this qubit in the quantum register."""

    @property
    def sim_reg(self) -> SimRegType:
        """The register of the simulator."""
        return self.ref.sim_reg

    @property
    def addr(self) -> int:
        """The global address of the qubit."""
        return self.ref.addrs[self.pos]

    def is_active(self) -> bool:
        """Check if the qubit is active.

        Returns
            True if the qubit is active, False otherwise.

        """
        return self.ref.qubit_state[self.pos] is QubitState.Active

    def drop(self):
        """Drop the qubit in-place."""
        self.ref.drop(self.pos)
