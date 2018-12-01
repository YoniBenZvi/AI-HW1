from framework.graph_search import *
from framework.ways import *
from .map_problem import MapProblem
from .deliveries_problem_input import DeliveriesProblemInput
from .relaxed_deliveries_problem import RelaxedDeliveriesState, RelaxedDeliveriesProblem

from typing import Set, FrozenSet, Optional, Iterator, Tuple, Union


class StrictDeliveriesState(RelaxedDeliveriesState):
    """
    An instance of this class represents a state of the strict
     deliveries problem.
    This state is basically similar to the state of the relaxed
     problem. Hence, this class inherits from `RelaxedDeliveriesState`.

    TODO:
        If you believe you need to modify the state for the strict
         problem in some sense, please go ahead and do so.
    """
    pass


class StrictDeliveriesProblem(RelaxedDeliveriesProblem):
    """
    An instance of this class represents a strict deliveries problem.
    """

    name = 'StrictDeliveries'

    def __init__(self, problem_input: DeliveriesProblemInput, roads: Roads,
                 inner_problem_solver: GraphProblemSolver, use_cache: bool = True):
        super(StrictDeliveriesProblem, self).__init__(problem_input)
        self.initial_state = StrictDeliveriesState(
            problem_input.start_point, frozenset(), problem_input.gas_tank_init_fuel)
        self.inner_problem_solver = inner_problem_solver
        self.roads = roads
        self.use_cache = use_cache
        self._init_cache()

    def _init_cache(self):
        self._cache = {}
        self.nr_cache_hits = 0
        self.nr_cache_misses = 0

    def _insert_to_cache(self, key, val):
        if self.use_cache:
            self._cache[key] = val

    def _get_from_cache(self, key):
        if not self.use_cache:
            return None
        if key in self._cache:
            self.nr_cache_hits += 1
        else:
            self.nr_cache_misses += 1
        return self._cache.get(key)

    def expand_state_with_costs(self, state_to_expand: GraphProblemState) -> Iterator[Tuple[GraphProblemState, float]]:
        from deliveries.map_heuristics import AirDistHeuristic
        """
        TODO: implement this method!
        This method represents the `Succ: S -> P(S)` function of the strict deliveries problem.
        The `Succ` function is defined by the problem operators as shown in class.
        The relaxed problem operators are defined in the assignment instructions.
        It receives a state and iterates over the successor states.
        Notice that this is an *Iterator*. Hence it should be implemented using the `yield` keyword.
        For each successor, a pair of the successor state and the operator cost is yielded.
        """
        assert isinstance(state_to_expand, StrictDeliveriesState)
        # Iterate over all the problem's remaining possible stop points and check whether they're close enough to reach
        for junction in self.possible_stop_points.difference(state_to_expand.dropped_so_far):
            # skip if junction is the current junction we're at
            if junction.index == state_to_expand.current_location.index:
                continue
            cache_source = state_to_expand.current_location.index
            cache_destination = junction.index
            map_problem_a_star_instance = AStar(AirDistHeuristic, 0.5)
            operator_cost = self._get_from_cache((cache_source, cache_destination)) or \
                            map_problem_a_star_instance.solve_problem(
                                MapProblem(self.roads, cache_source, cache_destination)).final_search_node.cost
            self._insert_to_cache((cache_source, cache_destination), operator_cost)

            # The link has been found, therefore we're calculating the fuel that'll be left
            # after traveling via this link.
            fuel_left = state_to_expand.fuel - operator_cost

            # if not enough fuel to reach there, continue to next possible junction
            if fuel_left < 0:
                continue

            # Create the successor state (it should be an instance of class `StrictDeliveriesState`).
            if junction not in self.gas_stations:
                successor_state = StrictDeliveriesState(junction,
                                                        state_to_expand.dropped_so_far.union([junction]),
                                                        fuel_left)

            else:
                successor_state = StrictDeliveriesState(junction, state_to_expand.dropped_so_far,
                                                        self.gas_tank_capacity)

            # Yield the successor state and the cost of the operator we used to get this successor.
            yield (successor_state, operator_cost)

    def is_goal(self, state: GraphProblemState) -> bool:
        """
        This method receives a state and returns whether this state is a goal.
        TODO: implement this method!
        """
        assert isinstance(state, StrictDeliveriesState)

        return len(self.drop_points.difference(state.dropped_so_far)) == 0
