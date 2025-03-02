from unittest.mock import Mock

from bloqade import qasm2
from bloqade.noise import native
from bloqade.pyqrack import StackMemory, PyQrackInterpreter, reg

simulation = qasm2.extended.add(native)


def test_atom_loss():

    @simulation
    def test_atom_loss(c: qasm2.CReg):
        q = qasm2.qreg(2)
        native.atom_loss_channel([q[0]], prob=0.5)
        native.atom_loss_channel([q[1]], prob=0.8)
        qasm2.measure(q[0], c[0])

        return q

    rng_state = Mock()
    rng_state.uniform.return_value = 0.7
    input = reg.CRegister(1)
    memory = StackMemory(total=2, allocated=0, sim_reg=Mock())

    result: reg.PyQrackReg[Mock] = (
        PyQrackInterpreter(simulation, memory=memory, rng_state=rng_state)
        .run(test_atom_loss, (input,))
        .expect()
    )

    assert result.qubit_state[0] is reg.QubitState.Lost
    assert result.qubit_state[1] is reg.QubitState.Active
    assert input[0] is reg.Measurement.One
