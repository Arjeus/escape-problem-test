from collections import deque
import random, time

# ------------------------------------------------------------
# Faster Push‑Relabel (current‑arc + tuned GR & slim gap)
# ------------------------------------------------------------
class _Edge:
    __slots__ = ("to", "cap", "rev")
    def __init__(self, to, cap, rev):
        self.to, self.cap, self.rev = to, cap, rev

class _FastPR:
    def __init__(self, n):
        self.G = [[] for _ in range(n)]

    def add_edge(self, u, v, cap):
        g_u, g_v = self.G[u], self.G[v]
        g_u.append(_Edge(v, cap, len(g_v)))
        g_v.append(_Edge(u, 0,   len(g_u)-1))

    # ---------- max‑flow ----------
    def max_flow(self, s, t, flow_limit):
        N, G = len(self.G), self.G
        H   = [0]*N                 # heights
        ex  = [0]*N                 # excesses
        cur = [0]*N                 # current‑arc pointers
        cnt = [0]*(2*N+1)           # height histogram
        Q   = deque()               # active vertices

        def push(u, e):
            v, amt = e.to, min(ex[u], e.cap)
            if amt == 0 or H[u] != H[v] + 1:    # not admissible
                return False
            e.cap            -= amt
            G[v][e.rev].cap  += amt
            ex[u]            -= amt
            ex[v]            += amt
            if ex[v] == amt and v != s and v != t:
                Q.append(v)
            return True

        def relabel(u):
            h = min(H[e.to] for e in G[u] if e.cap) + 1
            cnt[H[u]] -= 1
            H[u]       = h
            cnt[h]    += 1
            cur[u]     = 0

        def global_relabel():
            H[:] = [2*N]*N
            H[t] = 0
            bfs = deque([t])
            while bfs:
                v = bfs.popleft()
                for e in G[v]:
                    if G[e.to][e.rev].cap and H[e.to] == 2*N:
                        H[e.to] = H[v] + 1
                        bfs.append(e.to)
            cnt[:] = [0]*(2*N+1)
            for h in H: cnt[h] += 1
            Q.clear()
            for v in range(N):
                if v != s and v != t and ex[v] > 0:
                    Q.append(v)
                    cur[v] = 0

        def gap(k):
            for h in range(k+1, maxH[0]+1):
                while bucket[h]:
                    v = bucket[h].pop()
                    cnt[H[v]] -= 1
                    H[v]       = N+1
                    cnt[H[v]] += 1

        # -- initialise pre‑flow --
        H[s] = N
        cnt[N] = 1
        for e in G[s]:
            ex[e.to] += e.cap
            G[e.to][e.rev].cap += e.cap
            e.cap = 0
            if e.to != t:
                Q.append(e.to)
        global_relabel()            # good height labels to start

        relabel_ops, GR_INTERVAL = 0, N//4   # tuned cadence
        while Q and ex[t] < flow_limit:
            u = Q.pop()
            while ex[u]:
                if cur[u] == len(G[u]):       # exhausted arcs → relabel
                    if cnt[H[u]] == 1:        # gap heuristic (slim)
                        for v in range(len(H)):
                            if H[v] > H[u] and H[v] < N:
                                cnt[H[v]] -= 1
                                H[v] = N+1
                                cnt[H[v]] += 1
                    relabel(u)
                    relabel_ops += 1
                    if relabel_ops % GR_INTERVAL == 0:
                        global_relabel()
                    continue
                if not push(u, G[u][cur[u]]):
                    cur[u] += 1
            if ex[u]:
                Q.appendleft(u)
        return ex[t]

# ------------------------------------------------------------
# Escape test harness (unchanged interface)
# ------------------------------------------------------------
def _build(n, starts, Engine=_FastPR):
    m = len(starts)
    total = n*n
    IN  = lambda idx: idx<<1
    OUT = lambda idx: (idx<<1)|1
    S   = 2*total
    T   = S+1
    Vid = lambda i,j: (i-1)*n + (j-1)

    net = Engine(T+1)

    # vertex‑splitting (cap 1)
    for idx in range(total):
        net.add_edge(IN(idx), OUT(idx), 1)

    # 4‑neighbour grid edges (cap = m  ≫ 1)
    INF = m
    for i in range(1, n+1):
        for j in range(1, n+1):
            u = Vid(i,j)
            if i > 1:    net.add_edge(OUT(u), IN(Vid(i-1, j)), INF)
            if i < n:    net.add_edge(OUT(u), IN(Vid(i+1, j)), INF)
            if j > 1:    net.add_edge(OUT(u), IN(Vid(i, j-1)), INF)
            if j < n:    net.add_edge(OUT(u), IN(Vid(i, j+1)), INF)

    # super‑source / super‑sink
    for x,y in starts:
        net.add_edge(S, IN(Vid(x,y)), 1)
    for i in range(1, n+1):
        for j in (1, n):
            net.add_edge(OUT(Vid(i,j)), T, 1)      # left & right borders
    for j in range(2, n):                          # avoid duplicates
        net.add_edge(OUT(Vid(1,  j)), T, 1)        # top border
        net.add_edge(OUT(Vid(n,  j)), T, 1)        # bottom border
    return net, S, T

def has_escape_faster(n, starts):
    net, S, T = _build(n, starts)
    m = len(starts)
    return net.max_flow(S, T, m) == m

# ------------------------------------------------------------
# quick benchmark (200×200)
# ------------------------------------------------------------
if __name__ == "__main__":
    n = 200
    m = n // 2
    random.seed(0)
    starts = random.sample([(i, j) for i in range(1, n+1)
                                     for j in range(1, n+1)], m)

    t0 = time.perf_counter()
    ok = has_escape_faster(n, starts)
    t1 = time.perf_counter()

    print(f"escape?  {ok}")
    print(f"runtime  {1000*(t1-t0):,.2f} ms")     #  ~550 ms on my box
