# QELM Observable Reconstruction

This repository contains `q_circuit_with_unitary_matrix.py`, a Python script that simulates a Quantum Extreme Learning Machine (QELM). It reconstructs single-qubit observables (e.g., Pauli matrices) using a linear readout and evaluates how the reconstruction error ($\sqrt{\text{MSRE}}$) scales as the number of ancilla qubits increases.

## Requirements & Usage

Requires Python 3.x. Install dependencies and run the script via terminal:

```bash
pip install numpy matplotlib tqdm
python q_circuit_with_unitary_matrix.py
```

## Configuration Options

You can customize the simulation by editing the variables in the `INPUT` block of the script:

* **`operator_choice`**: Target observable (`sigma_x`, `sigma_y`, or `sigma_z`).
* **`select_method`**: Method for generating unitaries (`"ha"` for Haar measure, `"gs"` for Gram-Schmidt).
* **`n_a_range`**: The range of ancilla qubits to iterate over (e.g., `range(2, 6)`).
* **`n_train_states` / `n_test_states`**: Number of random quantum states used for training and evaluating.
* **`finite_statistics` / `n_trials`**: Set to `"yes"` to simulate measurement shot noise using `n_trials` shots, or `"no"` for exact probabilities.
* **`new_dataset`**: Set to `0` to reuse the exact same quantum states across all ancilla counts (recommended), or `1` to regenerate them at each step.