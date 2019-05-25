Model Predictive Shielding
=====

Contains the code for the model predictive shielding algorithm proposed in [https://arxiv.org/abs/1905.10691](https://arxiv.org/abs/1905.10691).

- Works with Python 3.6. Python dependencies are `numpy`, `pytorch`, `gym`, and `sklearn`.

- The online shielding code is in the `python` folder. The main routines are `cartpole_test.py`, `cartpole_test_bl.py`, `bicycle_test.py`, and `bicycle_test_bl.py`. For example, to run the cart-pole code, run
```
    $ cd python
    $ python -m spire.main.cartpole_test
```

- The matlab directory contains the code used to perform LQR verification for the cart-pole dynamical system. This code depends on SOSTOOLS 3.03, which is available at https://www.cds.caltech.edu/sostools. We used the SeDuMi optimizer with SOSTOOLS, available at https://github.com/sqlp/sedumi.

- The baseline code depends on the Z3 theorem prover, which is available at https://github.com/Z3Prover/z3. To run the baseline code, run
```
    $ cd baseline/python
    $ python main.py
```