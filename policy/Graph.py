from heapq import heappush
from heapq import heappop
from heapq import heapify

INFINITY = 1E6

# graph[point_no_i]: [(point_no_j, direct_distance_i_to_j, is_arc, circle), ...]}
# returns optimal_distances, previous_points
def Dijkstra(graph):
    n = len(graph)
    previous = [(-1, False, None)] * n
    h = [(0, 0)]
    visited = [False] * n
    distances = [INFINITY] * n
    # print "graph[0]"
    # print graph[0]
    while h:
        v = heappop(h)
        # print 'v'
        #print v
        if visited[v[1]]:
            continue
        visited[v[1]] = True
        distances[v[1]] = v[0]
        
        for u, d, is_arc, circle in graph[v[1]]:
            # print 'u: %d, d: %f' % (u, d)
            new_d = d + v[0]
            # print new_d
            if new_d < distances[u]:
                heappush(h, (new_d, u))
                distances[u] = new_d
                previous[u] = (v[1], is_arc, circle)
            # print distances
                
    return distances, previous
            