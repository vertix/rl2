from heapq import heappush
from heapq import heappop
from heapq import heapify

INFINITY = 1E6

# graph[point_no_i]: [Edge]}
# returns optimal_distances, previous_points
def Dijkstra(graph):
    n = len(graph)
    previous = [(-1, None)] * n
    h = [(0, 0)]
    visited = [False] * n
    distances = [INFINITY] * n
    while h:
        v = heappop(h)
        if visited[v[1]]:
            continue
        visited[v[1]] = True
        distances[v[1]] = v[0]

        for edge in graph[v[1]]:
            new_d = edge.w + v[0]
            if new_d < distances[edge.v]:
                heappush(h, (new_d, edge.v))
                distances[edge.v] = new_d
                previous[edge.v] = (v[1], edge)

    return distances, previous
            