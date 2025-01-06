import time
from itertools import count

import jax
from jax import lax, numpy as jnp, random
from scipy.optimize import linear_sum_assignment


jax.config.update("jax_enable_x64", True)


def find_nearest_sort(array1, parity1, array2, parity2):
    # sort the image using a navie method, and don't promise the minimum distance,
    # but may be sufficient for binary lens. VBBL use the similar method.
    # To use Jax's shard_map api to get get combination of
    # jax.jit and parallel, we can't use while loop now.
    # check here for more details: https://jax.readthedocs.io/en/latest/jep/14273-shard-map.html
    cost = jnp.abs(array2 - array1[:, None]) + jnp.abs(parity2 - parity1[:, None]) * 5
    cost = jnp.where(jnp.isnan(cost), 100, cost)
    idx = jnp.argmin(cost, axis=1)

    def nan_in_array1(carry):
        array1, array2, cost, idx = carry
        idx = jnp.where(~jnp.isnan(array1), idx, -1)
        diff_idx = jnp.setdiff1d(jnp.arange(array1.shape[0]), idx, size=2)
        used = 0
        for i in range(array1.shape[0]):
            cond = jnp.isnan(array1[i])
            idx_i = jnp.where(cond, diff_idx[used], idx[i])
            idx = idx.at[i].set(idx_i)
            used = jnp.where(cond, used + 1, used)
        return (array1, array2, cost, idx)

    def nan_not_in_array1(carry):
        def nan_in_array2(carry):
            array1, array2, cost, idx = carry
            idx = jnp.argmin(cost, axis=0)
            idx = jnp.where(~jnp.isnan(array1), idx, -1)
            diff_idx = jnp.setdiff1d(jnp.arange(array2.shape[0]), idx, size=2)
            used = 0
            for i in range(array1.shape[0]):
                cond = jnp.isnan(array2[i])
                idx_i = jnp.where(cond, diff_idx[used], idx[i])
                idx = idx.at[i].set(idx_i)
                used = jnp.where(cond, used + 1, used)
            ## rearrange the idx
            row_resort = jnp.argsort(idx)
            col_idx = jnp.arange(array1.shape[0])
            return (array1, array2, cost, col_idx[row_resort])

        carry = lax.cond(
            jnp.isnan(array2).sum() == 2,
            nan_in_array2,
            lambda x: x,
            (array1, array2, cost, idx),
        )
        return carry

    carry = lax.cond(
        jnp.isnan(array1).sum() == 2,
        nan_in_array1,
        nan_not_in_array1,
        (array1, array2, cost, idx),
    )
    array1, array2, cost, idx = carry
    return idx


def find_nearest(array1, parity1, array2, parity2):
    # linear sum assignment, the theoritical complexity is O(n^3) but our relization turns out to be much fast
    # for small cost matrix. adopted from https://github.com/google/jax/issues/10403 and I make it jit-able
    cost = (
        jnp.abs(array2 - array1[:, None]) + jnp.abs(parity2 - parity1[:, None]) * 5
    )  # 系数可以指定防止出现错误，系数越大鲁棒性越好，但是速度会变慢些
    cost = jnp.where(jnp.isnan(cost), 100, cost)
    row_ind, col_idx = solve(cost)

    # col_idx=find_nearest_sort(array1, parity1, array2, parity2)

    return col_idx


@jax.jit
def augmenting_path(cost, u, v, path, row4col, i):
    minVal = 0
    remaining = jnp.arange(cost.shape[1])[::-1]
    num_remaining = cost.shape[1]
    SR = jnp.full(cost.shape[0], False)
    SC = jnp.full(cost.shape[1], False)
    shortestPathCosts = jnp.full(cost.shape[1], jnp.inf)

    sink = -1
    break_cond = False

    def cond_fun(carry):
        (
            sink,
            minVal,
            remaining,
            SR,
            SC,
            shortestPathCosts,
            path,
            break_cond,
            i,
            u,
            v,
            row4col,
            cost,
            num_remaining,
        ) = carry
        return (sink == -1) & (~break_cond)

    def while_loop_body(carry):
        (
            sink,
            minVal,
            remaining,
            SR,
            SC,
            shortestPathCosts,
            path,
            break_cond,
            i,
            u,
            v,
            row4col,
            cost,
            num_remaining,
        ) = carry
        index = -1
        lowest = jnp.inf
        SR = SR.at[i].set(True)

        def body_fun(carry):
            (
                cost,
                u,
                v,
                path,
                row4col,
                i,
                remaining,
                minVal,
                shortestPathCosts,
                lowest,
                index,
                it,
            ) = carry
            j = remaining[it]
            r = minVal + cost[i, j] - u[i] - v[j]
            path = lax.cond(
                r < shortestPathCosts[j], lambda: path.at[j].set(i), lambda: path
            )
            shortestPathCosts = shortestPathCosts.at[j].min(r)

            index = lax.cond(
                (shortestPathCosts[j] < lowest)
                | ((shortestPathCosts[j] == lowest) & (row4col[j] == -1)),
                lambda: it,
                lambda: index,
            )
            lowest = jnp.minimum(lowest, shortestPathCosts[j])
            it += 1
            return (
                cost,
                u,
                v,
                path,
                row4col,
                i,
                remaining,
                minVal,
                shortestPathCosts,
                lowest,
                index,
                it,
            )

        carry = lax.while_loop(
            lambda x: x[-1] < num_remaining,
            body_fun,
            (
                cost,
                u,
                v,
                path,
                row4col,
                i,
                remaining,
                minVal,
                shortestPathCosts,
                lowest,
                index,
                0,
            ),
        )
        (
            cost,
            u,
            v,
            path,
            row4col,
            i,
            remaining,
            minVal,
            shortestPathCosts,
            lowest,
            index,
            _,
        ) = carry
        minVal = lowest

        def True_fun(carry):
            remaining, index, row4col, sink, i, SC, num_remaining, break_cond = carry
            sink = -1
            break_cond = True
            return (remaining, index, row4col, sink, i, SC, num_remaining, break_cond)

        def False_fun(carry):
            remaining, index, row4col, sink, i, SC, num_remaining, break_cond = carry
            j = remaining[index]

            pred = row4col[j] == -1
            sink = lax.cond(pred, lambda: j, lambda: sink)
            i = lax.cond(~pred, lambda: row4col[j], lambda: i)

            SC = SC.at[j].set(True)
            num_remaining -= 1
            remaining = remaining.at[index].set(remaining[num_remaining])
            return (remaining, index, row4col, sink, i, SC, num_remaining, break_cond)

        carry = lax.cond(
            minVal == jnp.inf,
            True_fun,
            False_fun,
            (remaining, index, row4col, sink, i, SC, num_remaining, break_cond),
        )
        remaining, index, row4col, sink, i, SC, num_remaining, break_cond = carry
        return (
            sink,
            minVal,
            remaining,
            SR,
            SC,
            shortestPathCosts,
            path,
            break_cond,
            i,
            u,
            v,
            row4col,
            cost,
            num_remaining,
        )

    carry = lax.while_loop(
        cond_fun,
        while_loop_body,
        (
            sink,
            minVal,
            remaining,
            SR,
            SC,
            shortestPathCosts,
            path,
            break_cond,
            i,
            u,
            v,
            row4col,
            cost,
            num_remaining,
        ),
    )
    (
        sink,
        minVal,
        remaining,
        SR,
        SC,
        shortestPathCosts,
        path,
        break_cond,
        i,
        u,
        v,
        row4col,
        cost,
        num_remaining,
    ) = carry
    return sink, minVal, remaining, SR, SC, shortestPathCosts, path


@jax.jit
def solve(cost):
    """
    Solves the linear sum assignment problem using the Hungarian algorithm.
    adapted from https://github.com/google/jax/issues/10403

    Parameters:
        cost (ndarray): The cost matrix representing the assignment problem.

    Returns:
        row_ind (ndarray): The row indices of the assigned elements.
        col_ind (ndarray): The column indices of the assigned elements.
    """

    # transpose = cost.shape[1] < cost.shape[0]
    # if transpose:#判断矩阵是否需要转置，对于方阵不需要
    #     cost = cost.T

    u = jnp.full(cost.shape[0], 0.0)
    v = jnp.full(cost.shape[1], 0.0)
    path = jnp.full(cost.shape[1], -1)
    col4row = jnp.full(cost.shape[0], -1)
    row4col = jnp.full(cost.shape[1], -1)

    def loop_body(carry, curRow):
        u, v, path, col4row, row4col, cost = carry
        j, minVal, remaining, SR, SC, shortestPathCosts, path = augmenting_path(
            cost, u, v, path, row4col, curRow
        )

        u = u.at[curRow].add(minVal)

        mask = SR & (jnp.arange(cost.shape[0]) != curRow)
        u = jnp.where(mask, u + minVal - shortestPathCosts[col4row], u)
        # u = u.at[mask].add(minVal - shortestPathCosts[col4row][mask])
        v = jnp.where(SC, v + shortestPathCosts - minVal, v)

        # v = v.at[SC].add(shortestPathCosts[SC] - minVal)
        def while_loop_body(carry):
            path, j, row4col, col4row, break_cond = carry

            i = path[j]

            row4col = row4col.at[j].set(i)

            col4row, j = col4row.at[i].set(j), col4row[i]

            break_cond = ~(i == curRow)
            return (path, j, row4col, col4row, break_cond)

        carry = lax.while_loop(
            lambda x: x[-1], while_loop_body, (path, j, row4col, col4row, True)
        )
        path, j, row4col, col4row, break_cond = carry
        return (u, v, path, col4row, row4col, cost), curRow

    carry, _ = lax.scan(
        loop_body, (u, v, path, col4row, row4col, cost), jnp.arange(cost.shape[0])
    )
    u, v, path, col4row, row4col, cost = carry
    return jnp.arange(cost.shape[0]), col4row
    # if transpose:
    #     v = col4row.argsort()
    #     return col4row[v], v
    # else:
    #     return jnp.arange(cost.shape[0]), col4row


def main():
    key = random.PRNGKey(0)
    for t in count():
        key, subkey = random.split(key)
        key, subkey = random.split(key)
        cost = random.uniform(subkey, (5, 5))
        if t < 0:  # skip to failing case
            continue
        st = time.perf_counter()
        row_ind_1, col_ind_1 = linear_sum_assignment(cost)
        end = time.perf_counter()
        print("scipy time =", end - st)
        st = time.perf_counter()
        row_ind_2, col_ind_2 = solve(cost)
        end = time.perf_counter()
        print("jax time =", end - st)
        print(
            "{:5} {}".format(
                t, (row_ind_1 == row_ind_2).all() and (col_ind_1 == col_ind_2).all()
            )
        )
        if ~((row_ind_1 == row_ind_2).all() and (col_ind_1 == col_ind_2).all()):
            break


if __name__ == "__main__":
    main()
