import heapq

def dijkstra(graph, start):
    # Initialize the distance dictionary with infinity for each node
    distances = {node: float('inf') for node in graph}
    distances[start] = 0
    
    # Min-heap priority queue to keep track of the node with the smallest distance
    priority_queue = [(0, start)]  # (distance, node)
    
    while priority_queue:
        current_distance, current_node = heapq.heappop(priority_queue)
        
        # If the popped node has already a shorter distance, skip processing it
        if current_distance > distances[current_node]:
            continue
        
        # Explore neighbors
        for neighbor, weight in graph[current_node]:
            distance = current_distance + weight
            
            # Only consider this path if it's shorter
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(priority_queue, (distance, neighbor))
    
    return distances

# Example usage
graph = {
    0: [(1, 4), (2, 1)],
    1: [(2, 2), (3, 5)],
    2: [(3, 1)],
    3: []
}

start = 0
result = dijkstra(graph, start)
print(result)
