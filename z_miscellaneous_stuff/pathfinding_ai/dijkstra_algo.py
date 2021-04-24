graph = {
    'a': {'b': 3, 'c': 4, 'd': 7},
    'b': {'c': 1, 'f': 5},
    'c': {'d': 2, 'f': 6},
    'd': {'e': 3, 'g': 6},
    'e': {'g': 3, 'h': 4},
    'f': {'e': 1, 'h': 8},
    'g': {'h': 2},
    'h': {'g': 2, 'e': 4, 'f': 8}
}


def dijkstra(graph, start, goal):
    shortest_distance = {}
    path_predecessor = {}
    unseen_nodes = graph
    path = []
    inf = 100000000

    for node in unseen_nodes:
        shortest_distance[node] = inf
    shortest_distance[start] = 0

    while unseen_nodes:
        min_dist_node = None
        for node in unseen_nodes:
            if min_dist_node == None:
                min_dist_node = node
            elif shortest_distance[node] < shortest_distance[min_dist_node]:
                min_dist_node = node

        path_options = graph[min_dist_node].items()
        for child, weight in path_options:
            if weight + shortest_distance[min_dist_node] < shortest_distance[child]:
                shortest_distance[child] = weight + \
                    shortest_distance[min_dist_node]
                path_predecessor[child] = min_dist_node

        unseen_nodes.pop(min_dist_node)

    current_node = goal
    while current_node != start:
        try:
            path.insert(0, current_node)
            current_node = path_predecessor[current_node]
        except:
            print("goal is not reachable")

    path.insert(0, start)

    print("")
    print(f"shortest path is {str(path)}")
    print(f"shortest path's distance is {str(shortest_distance[goal])}")
    print("")


dijkstra(graph, "a", "h")
