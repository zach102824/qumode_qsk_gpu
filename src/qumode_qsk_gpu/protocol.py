"""Frozen science and seed protocol. Identical rules to the CPU campaign."""
NS = (6, 9, 12, 15, 18, 21)
N_INSTANCES = 200
DEPTHS = (1, 2, 3, 4, 5, 6)
METHODS = ("nov3", "qaoa")
COUPLING_VARIANCE = "1/N"
HAMILTONIAN = "H = sum_{i<j} J_ij Z_i Z_j - B sum_i X_i,  J_ij ~ N(0, 1/N) for i<j, symmetrized"
FIELD = "per-instance finite-N ground-state von Neumann entropy peak on A=lowest floor(N/2) spins; not thermodynamic critical; not Gamma=1"
INIT_STATE = "|+>^{otimes N}"
SPSA_STARTS = 2
SPSA_BUDGET = "Q=1024*L inclusive"
BFGS_STARTS = 16
BFGS_MAXITER = 2000
BFGS_GTOL = 1e-10
BFGS_TIGHT = 1e-8
SEED_BASE = 831100000
PILOT_OFFSET = 800
N_PILOT_PER_N = 1
NO_SEED_REPLACEMENT = True
NO_TRAIN_TEST_SPLIT = True
NO_SPSA_WARM_ARM = True
STORE_GS_VECTORS = True
ROUND_SIZE = 10
N_ROUNDS = N_INSTANCES // ROUND_SIZE


def main_seed(n, index):
    if not (0 <= int(index) < N_INSTANCES):
        raise ValueError(index)
    return int(SEED_BASE + int(n) * 1000 + int(index))


def main_seeds(n):
    return [main_seed(n, i) for i in range(N_INSTANCES)]


def pilot_seed(n, index=0):
    if not (0 <= int(index) < 200):
        raise ValueError(index)
    return int(SEED_BASE + int(n) * 1000 + PILOT_OFFSET + int(index))


def pilot_seeds(n, k=N_PILOT_PER_N):
    return [pilot_seed(n, i) for i in range(k)]


def all_main_seeds():
    return {int(n): main_seeds(n) for n in NS}


def all_pilot_seeds():
    return {int(n): pilot_seeds(n) for n in NS}


def round_indices(round_id):
    r = int(round_id)
    if not 1 <= r <= N_ROUNDS:
        raise ValueError(round_id)
    lo = (r - 1) * ROUND_SIZE
    return list(range(lo, lo + ROUND_SIZE))


def param_count(method, L):
    L = int(L)
    if method == "nov3":
        return 3 + 3 * L
    if method == "qaoa":
        return 2 * L
    raise ValueError(method)
