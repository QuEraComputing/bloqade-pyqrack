import enum
from typing import List, Generic, TypeVar
from dataclasses import dataclass

from bloqade.qasm2.types import QReg, Qubit


class CRegister(list[bool]):
    def __init__(self, size: int):
        super().__init__(False for _ in range(size))


@dataclass(frozen=True)
class CBitRef:
    ref: CRegister
    pos: int

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
    size: int
    sim_reg: SimRegType
    addrs: tuple[int, ...]
    qubit_state: List[QubitState]

    def drop(self, pos: int):
        assert self.qubit_state[pos] is QubitState.Active, "Qubit already lost"
        self.qubit_state[pos] = QubitState.Lost

    def __getitem__(self, pos: int):
        return SimQubit(self, pos)


@dataclass(frozen=True)
class SimQubit(Qubit, Generic[SimRegType]):
    ref: SimQReg[SimRegType]
    pos: int

    @property
    def sim_reg(self) -> SimRegType:
        return self.ref.sim_reg

    @property
    def addr(self) -> int:
        return self.ref.addrs[self.pos]

    def is_active(self) -> bool:
        return self.ref.qubit_state[self.pos] is QubitState.Active

    def drop(self):
        self.ref.drop(self.pos)
