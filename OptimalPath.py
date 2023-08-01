class Graph:
    def __init__(self, roads, passengers) -> None:
        """
        Based on roads, initialize a layered directed weighted graph. The layered graph is a combination
        of two different graphs. The first graph contains edges representing non-carpool lanes, the second
        graph contains edges representing carpool lanes. Both graphs contains the same vertices. The two
        graphs are connected using edges with 0 weight for every vertex that has a passenger. The final result
        is a layered graph containing 2|L| locations and 2|R| + |P| edges, where |L| is the number of unique
        locations and |R| is the number of roads. If no passengers are present, the final graph will only contain
        |R| edges representing non-carpool lanes and |L| locations. The graph is represented using an adjacency
        list.

        Input:
            roads: Roads connecting one location to another location in the form of (a,b,c,d) where a is the
                   starting location, b is the ending location, c is the distance of non-carpool roads and
                   d is the distance of carpool roads.
            passengers: The locations that contain passengers

        Output: None since it doesn't return anything

        Postcondition: A layered graph consisting of two graphs with identical vertices and edges with possibly different
                       weights. If no passengers are present, a graph with vertices and edges representing non-carpool lanes


        Time complexity: O(|R| + |L|)

        Auxiliary space complexity: O(|R| + |L|)
        """
        self.roads = roads
        self.passengers = passengers
        self.max_vertex = 0
        for road in roads:
            if road[0] > self.max_vertex:
                self.max_vertex = road[0]
            if road[1] > self.max_vertex:
                self.max_vertex = road[1]

        # Generate a graph with edges representing non-carpool lanes. If passengers are present, generate
        # another graph with edges representing carpool lanes
        # Time complexity: O(|L|)
        self.graph = list(Vertex(num) for num in range(self.max_vertex + 1))
        if len(self.passengers) > 0:
            self.carpool_graph = list(Vertex(num2) for num2 in range(self.max_vertex + 1, 2 * (self.max_vertex + 1)))

        # Add edges to the graph(s)
        # Time complexity: O(|R|)
        for road in roads:
            self.graph[road[0]].add_edge(self.graph[road[0]], self.graph[road[1]], road[2])
            if len(self.passengers) > 0:
                self.carpool_graph[road[0]].add_edge(self.carpool_graph[road[0]], self.carpool_graph[road[1]], road[3])

        # If passengers are present, connect the two graphs by adding and edge starting from a vertex with passengers in
        # the non-carpool lanes graph to same vertex in the carpool lanes graph. Repeat for every vertex with a passenger
        # Time complexity: O(|L|), since |P| can at most be |L| - 2 and extend is O(n)
        if len(self.passengers) > 0:
            for passenger in self.passengers:
                self.graph[passenger].add_edge(self.graph[passenger], self.carpool_graph[passenger], 0)
            self.graph.extend(self.carpool_graph)

    """
    The dijkstra algorithm used to find the shortest path from the start vertex to every other vertex. 
    
    Input: 
        start: The starting vertex where the person begins his/her journey on
        
    Output: None since the function does not return anything
        
    Postcondition: All the vertex.distance is updated with the shortest distance from the start vertex
    
    Time complexity: O(|R|log|L|) where |R| is the number of roads and |L| is the number of locations
    
    Auxiliary space complexity: O(|R|log|L|) since a minimum heap is used and all the vertices are stored
                                in the heap and all the edges are distributed among all the vertices
    """
    def dijkstra(self, start):
        # Initialise the min heap
        heap = MinHeap(start, self.graph)

        # Check whether min heap is empty. If not serve the root node
        while heap.length > 1:
            current_vertex = heap.serve()
            current_vertex.mark_visited()

            # For each edge branching from the vertex in the served root node, check whether the other connected vertex
            # has a longer distance from the start the start vertex compared to the summation of the distance of the
            # current vertex from the start vertex and the weight of the edge. If so, update the distance of the other
            # vertex. Ignore if the other vertex has been visited.
            for edge in current_vertex.edges:
                adjacent_vertex = edge.vertex_two
                edge_distance = edge.weight

                if adjacent_vertex.visited:
                    pass
                elif adjacent_vertex.distance > current_vertex.distance + edge_distance:
                    adjacent_vertex.distance = current_vertex.distance + edge_distance
                    adjacent_vertex.previous = current_vertex
                    heap.rise(heap.index[adjacent_vertex.label])

    """
    Reversely generate the shortest path from start to end
    
    Input: 
        start: The starting vertex where the person begins his/her journey 
        end: The last vertex where the person ends his/her journey 
        
    Output: An array containing the shortest path from start to end in reverse order
    
    Time complexity: O(V) since the path may potentially require traversing twice the number of vertex in the graph
    
    Auxiliary space complexity: O(V), an array containing the path which may potentially be twice the number of vertex in the graph
    """
    def backtracking(self, start, end):
        # Initialise the start vertex
        start_vertex = self.graph[start]

        # Initialise the end vertex based on which end vertex has the shorter distance. Append the chosen
        # end vertex into the optimal path array
        if len(self.graph) > self.max_vertex + 1:
            if self.graph[end + len(self.carpool_graph)].distance < self.graph[end].distance:
                end_vertex = self.graph[end + len(self.carpool_graph)]
                optimal_path = [end_vertex.label - len(self.carpool_graph)]
            else:
                end_vertex = self.graph[end]
                optimal_path = [end_vertex.label]
        else:
            end_vertex = self.graph[end]
            optimal_path = [end_vertex.label]

        # Initialise the next vertex in the path
        next_vertex = end_vertex.previous

        # Keep appending vertices that are part of the path from start to end until the start vertex is reached
        # Since this backtracking is done on a layered graph consisting of two graphs with identical vertices,
        # certain checks are needed to prevent adding repeated vertices.
        while next_vertex != start_vertex:
            if next_vertex is None:
                return optimal_path
            if len(self.graph) > self.max_vertex + 1:
                if next_vertex.label > len(self.carpool_graph) - 1 and next_vertex.previous.label == next_vertex.label - len(self.carpool_graph):
                    next_vertex = next_vertex.previous
                elif next_vertex.label > len(self.carpool_graph) - 1:
                    optimal_path.append(next_vertex.label - len(self.carpool_graph))
                    next_vertex = next_vertex.previous
                else:
                    optimal_path.append(next_vertex.label)
                    next_vertex = next_vertex.previous
            else:
                optimal_path.append(next_vertex.label)
                next_vertex = next_vertex.previous

        # Append the start vertex to the optimal path array and return it
        optimal_path.append(start_vertex.label)
        return optimal_path


class Vertex:
    def __init__(self, label):
        """
        A vertex class which can contain various properties necessary for dijkstra i.e., distance,
        previous_vertex, etc.

        Input:
            label: The identity of the vertex

        Space complexity: O(|R|) since it is possible for a single vertex to be associated with every edge
        """
        self.label = label
        self.edges = []
        self.distance = math.inf
        self.previous = None
        self.visited = False

    """
    Mark a vertex as visited so that dijkstra can ignore it when deciding whether to decrease key or not
    Time complexity: O(1)
    """
    def mark_visited(self):
        self.visited = True

    """
    Add an edge associated with this vertex 
    Time complexity: O(1), since append is O(1)
    """
    def add_edge(self, vertex_one, vertex_two, weight):
        self.edges.append(Edge(vertex_one, vertex_two, weight))


class Edge:
    def __init__(self, vertex_one: Vertex, vertex_two: Vertex, weight):
        """
        An edge class
        Input:
            vertex_one: The vertex the edge starts from
            vertex_two: The vertex the edge ends at
            weight: the length of the edge

        """
        self.vertex_one = vertex_one
        self.vertex_two = vertex_two
        self.weight = weight


class MinHeap:
    def __init__(self, start, graph):
        """
        A minimum heap used by dijkstra to extract the vertex with the minimum distance after each round of edge
        relaxation. The heap is initialised by filling it with every vertex in the graph. A start is determined
        and it will be the root node of the heap before performing dijkstra. An index array is used to map each
        vertex to its position in the heap such that vertex access can be done in O(1) time complexity.

        Input:
            start: The starting vertex where the person begins his/her journey
            graph: The graph containing all the possible locations said person can visit and all possible roads
                   said person can traverse

        Output: 1) A heap array containing all the vertices in the graph
                2) A index array for mapping purposes

        Time complexity: O(|L|) since an heap array with |L| + 1 length and an index array with length |L|
                         is created

        Auxiliary space complexity: O(|R| + |L|) since each vertex is stored in the heap and each edge is stored
                                    in a vertex
        """
        # Initialise the heap and change the distance of the starting vertex to zero
        # Time complexity: O(|L|)
        self.heap = [0] * (len(graph) + 1)
        self.length = len(self.heap)
        graph[start].distance = 0

        # Replace the zeros in the heap array with vertices in the graph
        # Time complexity: O(|L|)
        for vertex_position in range(len(graph)):
            self.heap[vertex_position + 1] = graph[vertex_position]

        # Initialise the index array used for mapping
        # Time complexity: O(|L|)
        self.index = [0] * (len(graph))

        # Replace the zeros in index array with the correct position of each vertex in the heap array
        # Time complexity: O(|L|)
        for index_position in range(1, len(self.heap)):
            self.index[self.heap[index_position].label] = index_position

        # Perform the rise operation on the start vertex since it is the only vertex that has a non-infinite
        # distance. (Need to be done in the case where the starting vertex is not vertex 0)
        # Time complexity: O(log|L|)
        self.rise(start + 1)

    """
    Perform a rise operation on the selected vertex based on node_position. A rise checks whether the chosen 
    node is smaller than its parent node (comparing vertex.distance in this case). If it is smaller, swap 
    position with the parent node, if not, stop the rise operation since any node above the parent node is 
    smaller than the parent node in a minimum heap. If possible, repeat till it reaches the root node
    
    Input: 
        node_position: the position of the vertex in the heap
        
    Output: None since the function does not return anything
    
    Time complexity: O(log V) since it depends on the height of the minimum heap which is logarithm base 2 V
    """
    def rise(self, node_position):
        child_node_position = node_position
        parent_node_position = child_node_position // 2
        while parent_node_position != 0:
            if self.heap[child_node_position].distance < self.heap[parent_node_position].distance:
                self.swap(child_node_position, parent_node_position)
                child_node_position = parent_node_position
                parent_node_position = child_node_position // 2
            else:
                break

    """
    Perform a sink operation on the selected vertex based on node_position. A sink checks whether the chosen 
    node is larger than its child node (comparing vertex.distance in this case). If it is larger, swap 
    position with the child node, if not, stop the rise operation since any node below the child node is 
    larger than the child node in a minimum heap. If possible, repeat till it reaches one of leaf node
    
    Input: 
        node_position: the position of the vertex in the heap
        
    Output: None since the function does not return anything
    
    Time complexity: O(log V)
    """
    def sink(self, node_position):
        parent_node_position = node_position
        child_node_position = 2 * parent_node_position
        while child_node_position <= len(self.heap) - 1:
            if (child_node_position + 1) <= len(self.heap) - 1 and self.heap[child_node_position + 1].distance < self.heap[child_node_position].distance:
                child_node_position += 1
            if self.heap[parent_node_position].distance > self.heap[child_node_position].distance:
                self.swap(parent_node_position, child_node_position)
                parent_node_position = child_node_position
                child_node_position = 2 * parent_node_position
            else:
                break

    """
    Remove the root node from the heap, which is the vertex with smallest distance. The serve operation is done
    by swapping the root node with the rightest leaf node. Afterwards, the rightest leaf node is popped from the
    heap. A sink operation is done on the new vertex placed in the root node. Length is updated accordingly.
    
    Input: None since there is no other parameters other than self
    
    Output: The vertex that has the smallest distance away from the start vertex
    
    Time complexity: O(log V) since it is dominated by the sink operation. The other operations uses O(1) time
    """
    def serve(self):
        self.swap(1, len(self.heap) - 1)
        result_vertex = self.heap.pop()
        self.length -= 1
        self.sink(1)
        return result_vertex

    """
    Swaps the position of two vertices in the heap. Swap their position mapping in the index array as well
    
    Input:
        node_one_position: the position of the first vertex in the heap that is to be swapped with the second vertex
        node_two_position: the position of the second vertex in the heap that is to be swapped with the first vertex
        
    Time complexity: O(1)
    """
    def swap(self, node_one_position, node_two_position):
        self.heap[node_one_position], self.heap[node_two_position] = self.heap[node_two_position], self.heap[
            node_one_position]
        self.index[self.heap[node_one_position].label], self.index[self.heap[node_two_position].label] = \
            self.index[self.heap[node_two_position].label], self.index[self.heap[node_one_position].label]


def optimalRoute(start, end, passengers, road):
    """
    Find the shortest route from start to end, factoring the possibility of detouring to pick up passengers
    so that a shorter travelling time can be achieved as having passengers allows the person to take the
    possibly shorter carpool lanes. This is achieved by using a layered graph approach, combining two
    similar graphs that consists of identical vertices and edges with identical direction but possibly
    different weight. This approach is further explained in the Graph class.

    Input:
        start: The starting location where the person begins his/her journey
        end: The last location where the person ends his/her journey
        passengers: The locations that contain passengers
        road:  Roads connecting one location to another location in the form of (a,b,c,d) where a is the
               starting location, b is the ending location, c is the distance of non-carpool roads and
               d is the distance of carpool roads.

    Output: the shortest route in the form of an array of numbers representing the vertices took from start
            to end

    Time complexity: O(|R|log|L|)

    Space complexity; O(|R| + |L|)
    """
    result = []

    # Initialise the graph
    # Time complexity: O(|R| + |L|)
    g = Graph(road, passengers)

    # Perform dijkstra
    # Time complexity: O(|R|log|L|)
    g.dijkstra(start)

    # Perform backtracking
    # Time complexity: O(|L|)
    inverted_result = g.backtracking(start, end)

    # Invert the result from backtracking
    # Time complexity: O(|L|)
    for i in range(len(inverted_result)-1, -1, -1):
        result.append(inverted_result[i])
    return result
