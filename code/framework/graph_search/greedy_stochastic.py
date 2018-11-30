from .graph_problem_interface import *
from .best_first_search import BestFirstSearch
from typing import Optional
import numpy as np


class GreedyStochastic(BestFirstSearch):
    def __init__(self, heuristic_function_type: HeuristicFunctionType,
                 T_init: float = 1.0, N: int = 5, T_scale_factor: float = 0.95):
        # GreedyStochastic is a graph search algorithm. Hence, we use close set.
        super(GreedyStochastic, self).__init__(use_close=True)
        self.heuristic_function_type = heuristic_function_type
        self.T = T_init
        self.N = N
        self.T_scale_factor = T_scale_factor
        self.solver_name = 'GreedyStochastic (h={heuristic_name})'.format(
            heuristic_name=heuristic_function_type.heuristic_name)

    def _init_solver(self, problem: GraphProblem):
        super(GreedyStochastic, self)._init_solver(problem)
        self.heuristic_function = self.heuristic_function_type(problem)

    def _open_successor_node(self, problem: GraphProblem, successor_node: SearchNode):
        """
        TODO: implement this method!
        """
        if self.close.has_state(successor_node.state) and not self.open.has_state(successor_node.state):
            already_found_closed_node_with_same_state = self.close.get_node_by_state(successor_node.state)
            if already_found_closed_node_with_same_state.expanding_priority > successor_node.expanding_priority:
                self.close.remove_node(already_found_closed_node_with_same_state)
                successor_node.cost = successor_node.parent_search_node.cost + successor_node.operator_cost
                self.open.push_node(successor_node)
            return

        if self.open.has_state(successor_node.state):
            already_found_node_with_same_state = self.open.get_node_by_state(successor_node.state)
            if already_found_node_with_same_state.expanding_priority > successor_node.expanding_priority:
                self.open.extract_node(already_found_node_with_same_state)

        if not self.open.has_state(successor_node.state):
            self.open.push_node(successor_node)

    def _calc_node_expanding_priority(self, search_node: SearchNode) -> float:
        """
        TODO: implement this method!
        Remember: `GreedyStochastic` is greedy.
        """
        return self.heuristic_function.estimate(search_node.state)  # taking the heuristic value only into account

    def _extract_next_search_node_to_expand(self) -> Optional[SearchNode]:
        """
        Extracts the next node to expand from the open queue,
         using the stochastic method to choose out of the N
         best items from open.
        TODO: implement this method!
        Use `np.random.choice(...)` whenever you need to randomly choose
         an item from an array of items given a probabilities array `p`.
        You can read the documentation of `np.random.choice(...)` and
         see usage examples by searching it in Google.
        Notice: You might want to pop min(N, len(open) items from the
                `open` priority queue, and then choose an item out
                of these popped items. The other items have to be
                pushed again into that queue.
        """
        if self.open.is_empty():
            return None
        best_open_nodes = []
        # print('size of open: ',len(self.open))
        best_open_nodes_expanding_priorities = np.array([])
        num_best_open_nodes = min(len(self.open), self.N)
        for i in range(num_best_open_nodes):
            best_open_nodes.append(self.open.pop_next_node())
            # print(best_open_nodes[-1].expanding_priority)
            best_open_nodes_expanding_priorities = np.append(best_open_nodes_expanding_priorities,
                                                             best_open_nodes[-1].expanding_priority)
        alpha = np.min(best_open_nodes_expanding_priorities)
        node_to_expand = None  # temp value
        if alpha == 0.0:
            # The [0] at the end of the following line is due to the fact that list
            # comprehension returns a new list and not an element of the original list.
            node_to_expand = [node for node in best_open_nodes if node.expanding_priority == 0.0][0]
        else:
            numerator = (best_open_nodes_expanding_priorities / alpha) ** (-1 / self.T)
            denominator = numerator.sum()
            p = numerator / denominator
            node_to_expand = np.random.choice(best_open_nodes, None, True, p)

            # The following code pushes back the nodes we extracted from self.open except for the node
            # that was randomly chosen
            for node_to_push_back_to_open in best_open_nodes:
                if node_to_push_back_to_open == node_to_expand:
                    continue
                self.open.push_node(node_to_push_back_to_open)

        if self.use_close:
            self.close.add_node(node_to_expand)
        self.T *= self.T_scale_factor
        return node_to_expand
