from SVM import *


def evaluate(
    X,
    y: np.array,
    beta: float,
    z: list,
    radial_kernel: bool = False,
    w: list = None,
    X_train=None,
    y_train=None,
    alpha: list = None,
    gamma: float = None,
):
    N = X.shape[1]
    size = X.shape[0]
    x = np.array([X.iloc[i, :] for i in range(size)]).astype(np.float32)
    y = np.array(y).flatten()

    if radial_kernel:
        size_train = X_train.shape[0]
        x_train = np.array([X_train.iloc[i, :] for i in range(size_train)]).astype(np.float32)
        y_train = np.array(y_train).flatten()
        global K_
        K = K_(z, gamma)
        y_test = np.array(
            [
                1
                if np.sum(
                    [
                        alpha[j] * y_train[j] * K(x[i], x_train[j])
                        for j in range(size_train)
                    ]
                )
                + beta
                > 0
                else -1
                for i in range(size)
            ]
        )
    else:
        y_test = np.array(
            [
                1
                if np.sum([w[j] * z[j] * x[i][j] for j in range(N)]) + beta > 0
                else -1
                for i in range(size)
            ]
        )

    pos = y == 1
    neg = y == -1
    tr = y == y_test
    fa = y != y_test
    TP = np.count_nonzero(pos & tr)
    TN = np.count_nonzero(neg & tr)
    FP = np.count_nonzero(pos & fa)
    FN = np.count_nonzero(neg & fa)
    result = {
        "Acc": (TP + TN) / (TP + TN + FP + FN),
        "TPR": TP / (TP + FN),
        "TNR": TN / (TN + FP),
    }
    return result


from itertools import product
from sklearn.model_selection import KFold


def feature_selection(
    X: pd.DataFrame,
    y: pd.DataFrame,
    lambda_: list = [0.5, 0.5],
    radial_kernel: bool = False,
    folds: int = 10,
    folds2: int = 10,
    C_range: list = [2**i for i in range(-5, 6)],
    gamma_range: list = [2**i for i in range(-5, 6)],
):
    kf1 = KFold(n_splits=folds, shuffle=True, random_state=42)
    kf2 = KFold(n_splits=folds2, shuffle=True, random_state=42)
    X.reset_index(drop=True, inplace=True)
    y.reset_index(drop=True, inplace=True)
    Acc = np.zeros(folds)
    TPR = np.zeros(folds)
    TNR = np.zeros(folds)
    n_feature = np.zeros(folds)
    best_acc = 0
    best_C = None
    best_gamma = None
    for i, (train_id, test_id) in enumerate(kf1.split(X)):
        print(f"Running outer loop: Fold {i}")
        X_train = X.loc[train_id].reset_index(drop=True)
        y_train = y.loc[train_id].reset_index(drop=True)
        X_test = X.loc[test_id].reset_index(drop=True)
        y_test = y.loc[test_id].reset_index(drop=True)

        # Only tune parameter in the first fold
        if i == 0:
            if radial_kernel:
                for C, gamma in product(C_range, gamma_range):
                    print(f"Trying C={C} and gamma={gamma}")
                    total_acc = 0
                    for train_id2, test_id2 in kf2.split(X_train):
                        X_train2 = X_train.loc[train_id2].reset_index(drop=True)
                        y_train2 = y_train.loc[train_id2].reset_index(drop=True)
                        X_test2 = X_train.loc[test_id2].reset_index(drop=True)
                        y_test2 = y_train.loc[test_id2].reset_index(drop=True)
                        result = P1(X=X_train2, y=y_train2, lambda_=lambda_)
                        result2 = P3(
                            X=X_train2,
                            y=y_train2,
                            zk=result["z"],
                            C=C,
                            gamma=gamma,
                            lambda_=lambda_,
                        )
                        if result2 != None:
                            total_acc += evaluate(
                                X=X_test2,
                                y=y_test2,
                                beta=result2["beta"],
                                radial_kernel=True,
                                X_train=X_train2,
                                y_train=y_train2,
                                alpha=result2["alpha"],
                                z=result["z"],
                                gamma=gamma,
                            )["Acc"]
                    if total_acc > best_acc:
                        best_acc = total_acc
                        best_C = C
                        best_gamma = gamma
            else:
                # No gamma for linear kernel
                for C in C_range:
                    print(f"Trying C={C}")
                    total_acc = 0
                    for train_id2, test_id2 in kf2.split(X_train):
                        X_train2 = X_train.loc[train_id2].reset_index(drop=True)
                        y_train2 = y_train.loc[train_id2].reset_index(drop=True)
                        X_test2 = X_train.loc[test_id2].reset_index(drop=True)
                        y_test2 = y_train.loc[test_id2].reset_index(drop=True)
                        result = P1(X=X_train2, y=y_train2, lambda_=lambda_)
                        result2 = P2(
                            X=X_train2, y=y_train2, zk=result["z"], C=C, lambda_=lambda_
                        )
                        if result2 != None:
                            total_acc += evaluate(
                                X=X_test2,
                                y=y_test2,
                                beta=result2["beta"],
                                z=result["z"],
                                radial_kernel=False,
                                w=result2["w"],
                            )["Acc"]
                    if total_acc > best_acc:
                        best_acc = total_acc
                        best_C = C
                        print("Updating parameters")
        result = P1(X=X_train, y=y_train, lambda_=lambda_)
        if radial_kernel:
            result2 = P3(
                X=X_train,
                y=y_train,
                zk=result["z"],
                C=best_C,
                gamma=best_gamma,
                lambda_=lambda_,
            )
            res = evaluate(
                X=X_test,
                y=y_test,
                beta=result2["beta"],
                z=result["z"],
                radial_kernel=True,
                X_train=X_train,
                y_train=y_train,
                alpha=result2["alpha"],
                gamma=best_gamma,
            )
        else:
            result2 = P2(
                X=X_train, y=y_train, zk=result["z"], C=best_C, lambda_=lambda_
            )
            res = evaluate(
                X=X_test,
                y=y_test,
                beta=result2["beta"],
                z=result["z"],
                radial_kernel=False,
                w=result2["w"],
            )
        Acc[i]=res["Acc"]
        TPR[i]=res["TPR"]
        TNR[i]=res["TNR"]
        n_feature[i] = result["z"].count(1)
        # res["n_feature"] = n_feature
    
    res = {
        "Acc": Acc,
        "TPR": TPR,
        "TNR": TNR,
        "n_feature": n_feature
    }
    return res
