import numpy as np
from numba import njit, int64
import random, time

# ------------------------------------------------------------------
# flat CSR‑like graph arrays
# ------------------------------------------------------------------
@njit
def add_edge(u, v, c,
             head, to, cap, nxt, rev, edge_ptr):
    # forward edge
    to[edge_ptr]   = v
    cap[edge_ptr]  = c
    nxt[edge_ptr]  = head[u]
    rev[edge_ptr]  = edge_ptr + 1
    head[u]        = edge_ptr
    edge_ptr      += 1
    # reverse edge
    to[edge_ptr]   = u
    cap[edge_ptr]  = 0
    nxt[edge_ptr]  = head[v]
    rev[edge_ptr]  = edge_ptr - 1
    head[v]        = edge_ptr
    edge_ptr      += 1
    return edge_ptr


# ------------------------------------------------------------------
# helpers: simple array‑backed structures Numba likes
# ------------------------------------------------------------------
@njit
def stack_push(stack, top, v):
    stack[top] = v
    return top + 1               # new top

@njit
def stack_pop(stack, top):
    top -= 1
    return stack[top], top       # value, new top

@njit
def queue_enqueue(queue, tail, v):
    queue[tail] = v
    return tail + 1

@njit
def queue_dequeue(queue, head):
    v = queue[head]
    return v, head + 1


# ------------------------------------------------------------------
# Numba‑optimised Push–Relabel with gap + global relabel
# ------------------------------------------------------------------
@njit
def maxflow_numba(head, to, cap, nxt, rev,
                  N, S, T, need):
    H    = np.zeros(N, dtype=int64)
    ex   = np.zeros(N, dtype=int64)
    cnt  = np.zeros(2*N+2, dtype=int64)
    cur  = head.copy()

    # active‑vertex LIFO stack
    stack = np.empty(N, dtype=int64)
    top   = int64(0)

    # -------- initial pre‑flow from S --------
    H[S]     = N
    cnt[N]   = 1
    e = head[S]
    while e != -1:
        v      = to[e]
        flow   = cap[e]
        cap[e] = 0
        cap[rev[e]] += flow
        ex[v]  += flow
        if v != T:                      # S never on stack
            top   = stack_push(stack, top, v)
        e = nxt[e]

    # -------- global relabel (BFS from sink) --------
    queue = np.empty(N, dtype=int64)
    head_q = tail_q = int64(0)
    H[:] = 2*N
    H[T] = 0
    tail_q = queue_enqueue(queue, tail_q, T)
    while head_q < tail_q:
        v, head_q = queue_dequeue(queue, head_q)
        e = head[v]
        dist = H[v] + 1
        while e != -1:
            u = to[e]
            if cap[rev[e]] > 0 and H[u] == 2*N:
                H[u] = dist
                tail_q = queue_enqueue(queue, tail_q, u)
            e = nxt[e]
    cnt[:] = 0
    for h in H:
        cnt[h] += 1

    # -------- main loop --------
    work            = 0
    RELABEL_INTERVAL = N // 4

    while top and ex[T] < need:
        u, top = stack_pop(stack, top)
        while ex[u]:
            pushed = False
            e = cur[u]
            while e != -1:
                v = to[e]
                if cap[e] > 0 and H[u] == H[v] + 1:
                    # push admissible edge
                    delta       = min(ex[u], cap[e])
                    cap[e]     -= delta
                    cap[rev[e]] += delta
                    ex[u]      -= delta
                    ex[v]      += delta
                    if v != S and v != T and ex[v] == delta:
                        top = stack_push(stack, top, v)
                    if ex[u] == 0:
                        break
                    pushed = True
                e = nxt[e]
            cur[u] = e      # remember where we stopped

            if not pushed:  # need relabel
                # gap heuristic
                if cnt[H[u]] == 1:
                    gap_level = H[u]
                    for v in range(N):
                        if H[v] > gap_level and H[v] < N:
                            cnt[H[v]] -= 1
                            H[v]       = N + 1
                            cnt[H[v]] += 1

                # standard relabel
                min_h = 2*N
                e2 = head[u]
                while e2 != -1:
                    if cap[e2] > 0 and H[to[e2]] < min_h:
                        min_h = H[to[e2]]
                    e2 = nxt[e2]
                cnt[H[u]] -= 1
                H[u]      = min_h + 1
                cnt[H[u]] += 1
                cur[u]    = head[u]
                work     += 1
                if work % RELABEL_INTERVAL == 0:
                    # fresh global relabel
                    head_q = tail_q = 0
                    H[:] = 2*N
                    H[T] = 0
                    tail_q = queue_enqueue(queue, tail_q, T)
                    while head_q < tail_q:
                        v, head_q = queue_dequeue(queue, head_q)
                        e2 = head[v]
                        d  = H[v] + 1
                        while e2 != -1:
                            w = to[e2]
                            if cap[rev[e2]] > 0 and H[w] == 2*N:
                                H[w] = d
                                tail_q = queue_enqueue(queue, tail_q, w)
                            e2 = nxt[e2]
                    cnt[:] = 0
                    for h in H:
                        cnt[h] += 1

        if ex[u] and u != S and u != T:
            top = stack_push(stack, top, u)

    return ex[T] == need


# ------------------------------------------------------------------
# Public helper                                                    |
# ------------------------------------------------------------------
def has_escape_numba(n, starts):
    m      = len(starts)
    total  = n*n
    IN     = lambda idx: idx<<1
    OUT    = lambda idx: (idx<<1)|1
    S      = 2*total
    T      = S+1
    V      = T+1

    # upper bound on edges (split + 4‑grid + src + sink) *2
    maxE = ( 2*total               # split edges
           + 4*total               # grid
           + m                     # from S
           + 4*n                   # to T
           ) * 2

    head = np.full(V, -1, dtype=np.int64)
    to   = np.empty(maxE, dtype=np.int64)
    cap  = np.empty(maxE, dtype=np.int64)
    nxt  = np.empty(maxE, dtype=np.int64)
    rev  = np.empty(maxE, dtype=np.int64)
    edge_ptr = 0

    vid = lambda i, j: (i-1)*n + (j-1)

    # vertex capacity 1
    for idx in range(total):
        edge_ptr = add_edge(IN(idx), OUT(idx), 1,
                            head, to, cap, nxt, rev, edge_ptr)

    # 4‑neighbour grid
    INF = m
    for i in range(1, n+1):
        for j in range(1, n+1):
            u = vid(i, j)
            if i > 1: edge_ptr = add_edge(OUT(u), IN(vid(i-1, j)), INF,
                                          head, to, cap, nxt, rev, edge_ptr)
            if i < n: edge_ptr = add_edge(OUT(u), IN(vid(i+1, j)), INF,
                                          head, to, cap, nxt, rev, edge_ptr)
            if j > 1: edge_ptr = add_edge(OUT(u), IN(vid(i, j-1)), INF,
                                          head, to, cap, nxt, rev, edge_ptr)
            if j < n: edge_ptr = add_edge(OUT(u), IN(vid(i, j+1)), INF,
                                          head, to, cap, nxt, rev, edge_ptr)

    # sources
    for x, y in starts:
        edge_ptr = add_edge(S, IN(vid(x, y)), 1,
                            head, to, cap, nxt, rev, edge_ptr)

    # boundary sinks (avoid dupes)
    for i in range(1, n+1):
        edge_ptr = add_edge(OUT(vid(i, 1)), T, 1,
                            head, to, cap, nxt, rev, edge_ptr)
        edge_ptr = add_edge(OUT(vid(i, n)), T, 1,
                            head, to, cap, nxt, rev, edge_ptr)
    for j in range(2, n):
        edge_ptr = add_edge(OUT(vid(1, j)), T, 1,
                            head, to, cap, nxt, rev, edge_ptr)
        edge_ptr = add_edge(OUT(vid(n, j)), T, 1,
                            head, to, cap, nxt, rev, edge_ptr)

    # run compiled core
    return maxflow_numba(head, to, cap, nxt, rev,
                         V, S, T, m)


# ------------------------------------------------------------------
# quick demo & timing
# ------------------------------------------------------------------
if __name__ == "__main__":
    n = 200
    random.seed(0)
    starts = random.sample([(i, j)
                            for i in range(1, n+1)
                            for j in range(1, n+1)],
                           n//2)

    # first call includes JIT compilation
    t0 = time.perf_counter()
    print("escape?", has_escape_numba(n, starts))
    t1 = time.perf_counter()
    print(f"first call (compile+run): {1000*(t1-t0):,.2f} ms")

    # second call = hot execution
    t0 = time.perf_counter()
    print("escape?", has_escape_numba(n, starts))
    t1 = time.perf_counter()
    print(f"second call (hot):       {1000*(t1-t0):,.2f} ms")
