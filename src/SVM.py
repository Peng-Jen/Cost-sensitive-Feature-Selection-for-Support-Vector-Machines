from gurobipy import *
import numpy as np
import pandas as pd


def P1(
    X: pd.DataFrame,
    y: np.ndarray,
    lambda_: list,
    c: list = None,
    M2: int = 100,
    M3: int = 100,
):
    """The cost-sensitive FS procedure"""
    """ The first constraint causes infeasible """
    """ Model is feasible if zeta is continuous """

    # set up
    N = X.shape[1]
    size = X.shape[0]
    x = np.array([X.iloc[i, :] for i in range(size)]).astype(np.float32)
    y = np.array(y).flatten()
    if c == None:
        c = [1 for _ in range(N)]

    # create model
    model = Model("P1")
    model.setParam(GRB.Param.OutputFlag, 0)
    model.setParam(GRB.Param.TimeLimit, 300)

    # decision variables
    w = model.addVars(N, lb=float("-inf"), vtype=GRB.CONTINUOUS)
    beta = model.addVar(lb=float("-inf"), vtype=GRB.CONTINUOUS)
    zeta = model.addVars(size, vtype=GRB.BINARY)
    z = model.addVars(N, vtype=GRB.BINARY)

    # constraints
    model.addConstrs(
        y[i] * (quicksum(w[j] * x[i, j] for j in range(N)) + beta)
        >= 1 - M2 * (1 - zeta[i])
        for i in range(size)
    )
    model.addConstr(
        quicksum(zeta[i] * (1 - y[i]) for i in range(size))
        >= lambda_[0] * quicksum(1 - y[i] for i in range(size))
    )
    model.addConstr(
        quicksum(zeta[i] * (1 + y[i]) for i in range(size))
        >= lambda_[1] * quicksum(1 + y[i] for i in range(size))
    )
    model.addConstrs(w[k] <= M3 * z[k] for k in range(N))
    model.addConstrs(w[k] >= -M3 * z[k] for k in range(N))

    # objective function
    model.setObjective(quicksum(c[k] * z[k] for k in range(N)), GRB.MINIMIZE)

    # optimization
    model.optimize()


    # result
    result = {
        "w": [w[i].x for i in range(N)],
        "beta": beta.x,
        "z": [z[i].x for i in range(N)],
        "zeta": [zeta[i].x for i in range(size)],
    }
    return result


def P2(
    X: pd.DataFrame,
    y,
    zk: list,
    lambda_: list,
    C: int = 100,
    M1: int = 100,
):
    """Cost-sensitive sparse SVMs - linear"""
    # set up
    N = X.shape[1]
    size = X.shape[0]
    x = np.array([X.iloc[i, :] for i in range(size)]).astype(np.float32)
    y = np.array(y).flatten()

    z = [int(zi) for zi in zk]
    assert len(zk) == N
    for zi in z:
        assert zi == 0 or zi == 1

    # create model
    model = Model("P2")
    model.setParam(GRB.Param.OutputFlag, 0)
    model.setParam(GRB.Param.TimeLimit, 300)

    # decision variables
    w = model.addVars(N, lb=float("-inf"), vtype=GRB.CONTINUOUS)
    beta = model.addVar(lb=float("-inf"), vtype=GRB.CONTINUOUS)
    xi = model.addVars(size, vtype=GRB.CONTINUOUS)
    zeta = model.addVars(size, vtype=GRB.BINARY)

    # constraints
    model.addConstrs(
        y[i] * (quicksum(w[j] * z[j] * x[i, j] for j in range(N)) + beta) >= 1 - xi[i]
        for i in range(size)
    )
    model.addConstrs(xi[i] <= M1 * (1 - zeta[i]) for i in range(size))
    model.addConstr(
        quicksum(zeta[i] * (1 - y[i]) for i in range(size))
        >= lambda_[0] * quicksum(1 - y[i] for i in range(size))
    )
    model.addConstr(
        quicksum(zeta[i] * (1 + y[i]) for i in range(size))
        >= lambda_[1] * quicksum(1 + y[i] for i in range(size))
    )

    # objective function

    model.setObjective(
        quicksum(w[j] ** 2 * z[j] for j in range(N))
        + C * quicksum(xi[i] for i in range(size)),
        GRB.MINIMIZE,
    )

    # optimization
    model.optimize()

    if model.Status == GRB.OPTIMAL or model.Status == GRB.SUBOPTIMAL:
        result = {
            "w": [w[i].x for i in range(N)],
            "beta": beta.x,
            "xi": [xi[i].x for i in range(size)],
        }
    else:
        print(f"No solution found: Code = {model.Status}")
        result = None

    return result


def K_(z: list, gamma: float):
    N = len(z)

    def func(x, x_prime):
        return np.exp(-gamma * sum(z[k] * (x[k] - x_prime[k]) ** 2 for k in range(N)))

    return func


def P3(
    X: pd.DataFrame,
    y,
    zk: list,
    C: int,
    gamma: float,
    lambda_: list,
    M1: int = 100,
):
    # set up
    N = X.shape[1]
    size = X.shape[0]
    x = np.array([X.iloc[i, :] for i in range(size)]).astype(np.float32)
    y = np.array(y).flatten()

    z = [int(zi) for zi in zk]
    assert len(zk) == N
    for zi in z:
        assert zi == 0 or zi == 1

    global K_
    K = K_(z, gamma)

    # create model
    model = Model("P3")
    model.setParam(GRB.Param.OutputFlag, 0)
    model.setParam(GRB.Param.TimeLimit, 300)

    # decision variables
    alpha = model.addVars(size, ub=C / 2, vtype=GRB.CONTINUOUS)
    xi = model.addVars(size, vtype=GRB.CONTINUOUS)
    zeta = model.addVars(size, vtype=GRB.BINARY)
    beta = model.addVar(lb=float("-inf"), vtype=GRB.CONTINUOUS)

    # constraints
    model.addConstrs(
        (
            y[i]
            * (quicksum(alpha[j] * y[j] * K(x[i], x[j]) for j in range(size)) + beta)
            >= 1 - xi[i]
        )
        for i in range(size)
    )
    model.addConstrs((xi[i] <= M1 * (1 - zeta[i])) for i in range(size))
    model.addConstr(quicksum(alpha[i] * y[i] for i in range(size)) == 0)
    model.addConstr(
        quicksum(zeta[i] * (1 - y[i]) for i in range(size))
        >= lambda_[0] * (quicksum(1 - y[i] for i in range(size)))
    )
    model.addConstr(
        quicksum(zeta[i] * (1 + y[i]) for i in range(size))
        >= lambda_[1] * (quicksum(1 + y[i] for i in range(size)))
    )

    # optimization
    model.optimize()

    if model.Status == GRB.OPTIMAL or model.Status == GRB.SUBOPTIMAL:
        result = {
            "alpha": [alpha[i].x for i in range(size)],
            "xi": [xi[i].x for i in range(size)],
            "zeta": [zeta[i].x for i in range(size)],
            "beta": beta.x,
        }
    else:
        print(f"No solution found: Code = {model.Status}")
        result = None

    return result
