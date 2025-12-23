# -*- coding: utf-8 -*-
"""
NoCrash Benchmark Implementation for CARLA

Based on: "Exploring the Limitations of Behavior Cloning for Autonomous Driving"
Codevilla et al., ICCV 2019
https://arxiv.org/abs/1904.08980

Original benchmark specifications:
- 25 predefined routes per town
- 3 traffic conditions: Empty, Regular, Dense
- 6 weather conditions: 4 training + 2 new (test)
- Towns: Town01 (training), Town02 (testing)
"""

import random
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple
from enum import Enum

import numpy as np
import gym
import carla

from carla_env.carla_env import CarlaEnv


# ============================================================================
# NoCrash Weather Definitions (CARLA 0.9.x mapping)
# Original CARLA 0.8.4 IDs: Training [1,3,6,8], New [10,14]
# ============================================================================

class WeatherGroup(Enum):
    TRAINING = "training"
    NEW = "new"  # Test weathers (unseen during training)


NOCRASH_WEATHERS = {
    # Training weathers (ID 1, 3, 6, 8 in CARLA 0.8.4)
    "ClearNoon": {
        "params": carla.WeatherParameters.ClearNoon,
        "group": WeatherGroup.TRAINING,
    },
    "WetNoon": {
        "params": carla.WeatherParameters.WetNoon,
        "group": WeatherGroup.TRAINING,
    },
    "HardRainNoon": {
        "params": carla.WeatherParameters.HardRainNoon,
        "group": WeatherGroup.TRAINING,
    },
    "ClearSunset": {
        "params": carla.WeatherParameters.ClearSunset,
        "group": WeatherGroup.TRAINING,
    },
    # New weathers for testing (ID 10, 14 in CARLA 0.8.4)
    "WetSunset": {
        "params": carla.WeatherParameters.WetSunset,
        "group": WeatherGroup.NEW,
    },
    "SoftRainSunset": {
        "params": carla.WeatherParameters.SoftRainSunset,
        "group": WeatherGroup.NEW,
    },
}


def get_weathers_by_group(group: WeatherGroup) -> List[str]:
    """Get weather names by group (training or new)."""
    return [name for name, info in NOCRASH_WEATHERS.items() if info["group"] == group]


# ============================================================================
# NoCrash Traffic Conditions
# Original values from carla-simulator/driving-benchmarks/carla100.py
# ============================================================================

@dataclass
class TrafficCondition:
    """Traffic condition with vehicles and pedestrians count."""
    name: str
    vehicles: int
    pedestrians: int


# Town01 traffic conditions (training town)
TOWN01_TRAFFIC = {
    "empty": TrafficCondition("empty", 0, 0),
    "regular": TrafficCondition("regular", 20, 50),
    "dense": TrafficCondition("dense", 100, 250),
}

# Town02 traffic conditions (testing town)
TOWN02_TRAFFIC = {
    "empty": TrafficCondition("empty", 0, 0),
    "regular": TrafficCondition("regular", 15, 50),
    "dense": TrafficCondition("dense", 70, 150),
}


def get_traffic_for_town(town: str) -> dict:
    """Get traffic conditions for a specific town."""
    if "Town01" in town:
        return TOWN01_TRAFFIC
    elif "Town02" in town:
        return TOWN02_TRAFFIC
    else:
        # Default to Town01 traffic for other towns
        return TOWN01_TRAFFIC


# ============================================================================
# NoCrash Route Definitions
# Original 25 poses from NoCrash benchmark (spawn point index pairs)
# These are indices into CARLA's spawn point list
# ============================================================================

# Town01: 25 navigation poses [start_idx, end_idx]
# From carla-simulator/driving-benchmarks/carla100/carla100.py
TOWN01_POSES = [
    [105, 29], [27, 130], [102, 87], [132, 27], [25, 44],
    [4, 64], [34, 67], [54, 30], [140, 134], [105, 9],
    [148, 129], [65, 18], [21, 16], [147, 97], [134, 49],
    [30, 41], [81, 89], [69, 45], [102, 95], [18, 145],
    [111, 64], [79, 45], [84, 69], [73, 31], [37, 81],
]

# Town02: 25 navigation poses [start_idx, end_idx]
# From carla-simulator/driving-benchmarks/carla100/carla100.py
TOWN02_POSES = [
    [19, 66], [79, 14], [19, 57], [39, 53], [60, 26],
    [53, 76], [42, 13], [31, 71], [59, 35], [47, 16],
    [10, 61], [66, 3], [20, 79], [14, 56], [26, 69],
    [79, 19], [2, 29], [16, 14], [5, 57], [77, 68],
    [70, 73], [46, 67], [34, 77], [61, 49], [21, 12],
]


def get_poses_for_town(town: str) -> List[List[int]]:
    """Get route poses for a specific town."""
    if "Town01" in town:
        return TOWN01_POSES
    elif "Town02" in town:
        return TOWN02_POSES
    else:
        # For other towns, we'll use random routes
        return None


# ============================================================================
# NoCrash Scenario Definition
# ============================================================================

@dataclass
class NoCrashScenario:
    """A complete NoCrash scenario configuration."""
    name: str
    town: str
    traffic: TrafficCondition
    weather_name: str
    weather_params: carla.WeatherParameters
    weather_group: WeatherGroup
    route_idx: int  # Index into the poses list
    start_idx: int  # Spawn point index for start
    end_idx: int    # Spawn point index for end


def make_nocrash_scenarios(
    town: str = "Town01",
    traffic_levels: Sequence[str] = ("empty", "regular", "dense"),
    weather_group: WeatherGroup = WeatherGroup.TRAINING,
    routes_per_condition: int = 25,
) -> List[NoCrashScenario]:
    """
    Build NoCrash scenarios following the original benchmark.

    Args:
        town: CARLA town name (Town01 for training, Town02 for testing)
        traffic_levels: Traffic conditions to include
        weather_group: TRAINING or NEW weathers
        routes_per_condition: Number of routes per traffic/weather combination

    Returns:
        List of NoCrashScenario objects
    """
    traffic_map = get_traffic_for_town(town)
    poses = get_poses_for_town(town)
    weathers = get_weathers_by_group(weather_group)

    scenarios = []

    for traffic_name in traffic_levels:
        if traffic_name not in traffic_map:
            continue
        traffic = traffic_map[traffic_name]

        for weather_name in weathers:
            weather_info = NOCRASH_WEATHERS[weather_name]

            # Use predefined poses if available, otherwise we'll use random
            num_routes = min(routes_per_condition, len(poses)) if poses else routes_per_condition

            for route_idx in range(num_routes):
                if poses and route_idx < len(poses):
                    start_idx, end_idx = poses[route_idx]
                else:
                    start_idx, end_idx = -1, -1  # Will be assigned randomly

                scenario = NoCrashScenario(
                    name=f"{traffic_name}_{weather_name}_route{route_idx:02d}",
                    town=town,
                    traffic=traffic,
                    weather_name=weather_name,
                    weather_params=weather_info["params"],
                    weather_group=weather_info["group"],
                    route_idx=route_idx,
                    start_idx=start_idx,
                    end_idx=end_idx,
                )
                scenarios.append(scenario)

    return scenarios


# ============================================================================
# Legacy API compatibility
# ============================================================================

# For backward compatibility with simpler API
WEATHER_PRESETS = {name: info["params"] for name, info in NOCRASH_WEATHERS.items()}

# Add additional weather presets not in original NoCrash
WEATHER_PRESETS.update({
    "CloudyNoon": carla.WeatherParameters.CloudyNoon,
    "WetCloudyNoon": carla.WeatherParameters.WetCloudyNoon,
    "SoftRainNoon": carla.WeatherParameters.SoftRainNoon,
    "MidRainyNoon": carla.WeatherParameters.MidRainyNoon,
    "CloudySunset": carla.WeatherParameters.CloudySunset,
    "WetCloudySunset": carla.WeatherParameters.WetCloudySunset,
    "MidRainSunset": carla.WeatherParameters.MidRainSunset,
    "HardRainSunset": carla.WeatherParameters.HardRainSunset,
})


def make_default_scenarios(
    vehicle_levels: Sequence[str],
    weather_names: Sequence[str],
) -> List[NoCrashScenario]:
    """
    Build scenarios from simple names (backward compatible API).

    Note: This is a simplified version. For strict NoCrash benchmark,
    use make_nocrash_scenarios() instead.
    """
    # Use Town01 traffic by default
    veh_map = {
        "empty": TrafficCondition("empty", 0, 0),
        "regular": TrafficCondition("regular", 20, 50),
        "dense": TrafficCondition("dense", 100, 250),
    }

    scenarios = []
    route_idx = 0

    for veh_key in vehicle_levels:
        traffic = veh_map.get(veh_key, TrafficCondition(veh_key, 0, 0))

        for w_name in weather_names:
            weather_params = WEATHER_PRESETS.get(w_name, carla.WeatherParameters.ClearNoon)
            weather_info = NOCRASH_WEATHERS.get(w_name, {
                "params": weather_params,
                "group": WeatherGroup.TRAINING,
            })

            scenario = NoCrashScenario(
                name=f"{veh_key}_{w_name}",
                town="Town01",
                traffic=traffic,
                weather_name=w_name,
                weather_params=weather_params,
                weather_group=weather_info.get("group", WeatherGroup.TRAINING),
                route_idx=route_idx,
                start_idx=-1,  # Random
                end_idx=-1,    # Random
            )
            scenarios.append(scenario)
            route_idx += 1

    return scenarios


# ============================================================================
# NoCrash Environment Wrapper
# ============================================================================

class NoCrashEnv(gym.Env):
    """
    Wrapper around CarlaEnv implementing the NoCrash benchmark.

    The NoCrash benchmark evaluates autonomous driving policies on:
    - 25 predefined routes per town
    - 3 traffic conditions (empty, regular, dense)
    - 6 weather conditions (4 training + 2 new)

    Success: Reach goal without collision
    Failure: Collision with any object
    """

    def __init__(
        self,
        base_params: dict,
        scenarios: List[NoCrashScenario],
        max_steps: int = 1000,
        goal_threshold: float = 2.0,
        min_goal_distance: float = 50.0,
        use_predefined_routes: bool = True,
    ):
        """
        Initialize NoCrash environment.

        Args:
            base_params: Parameters for CarlaEnv
            scenarios: List of NoCrashScenario to evaluate
            max_steps: Maximum steps before timeout (based on route distance in original)
            goal_threshold: Distance to goal for success (meters)
            min_goal_distance: Minimum distance for random goals (meters)
            use_predefined_routes: Whether to use predefined spawn points
        """
        super().__init__()
        self.env = CarlaEnv(base_params)
        self.scenarios = scenarios
        self.max_steps = max_steps
        self.goal_threshold = goal_threshold
        self.min_goal_distance = min_goal_distance
        self.use_predefined_routes = use_predefined_routes

        self.scenario_idx = 0
        self.goal_location: Optional[carla.Location] = None
        self.start_location: Optional[carla.Location] = None
        self.cur_scenario: Optional[NoCrashScenario] = None
        self.step_count = 0
        self.route_distance = 0.0

        # Statistics tracking
        self.episode_stats = {
            "success": 0,
            "collision": 0,
            "timeout": 0,
            "off_road": 0,
        }

        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    def _get_spawn_point_by_index(self, idx: int) -> Optional[carla.Transform]:
        """Get spawn point by index, handling out of bounds."""
        spawn_points = self.env.vehicle_spawn_points
        if idx < 0 or idx >= len(spawn_points):
            return None
        return spawn_points[idx]

    def _pick_goal(self, scenario: NoCrashScenario) -> Tuple[carla.Transform, carla.Transform]:
        """
        Pick start and goal transforms based on scenario.

        Returns:
            Tuple of (start_transform, goal_transform)
        """
        spawn_points = self.env.vehicle_spawn_points

        # Try to use predefined routes first
        if self.use_predefined_routes and scenario.start_idx >= 0 and scenario.end_idx >= 0:
            start_tf = self._get_spawn_point_by_index(scenario.start_idx)
            goal_tf = self._get_spawn_point_by_index(scenario.end_idx)

            if start_tf is not None and goal_tf is not None:
                return start_tf, goal_tf

        # Fallback to random selection
        candidates = list(spawn_points)
        random.shuffle(candidates)

        # Pick start randomly
        start_tf = candidates[0] if candidates else None

        # Pick goal that's far enough from start
        goal_tf = None
        if start_tf:
            for sp in candidates[1:]:
                if sp.location.distance(start_tf.location) >= self.min_goal_distance:
                    goal_tf = sp
                    break

            # If no suitable goal found, pick farthest point
            if goal_tf is None and len(candidates) > 1:
                max_dist = 0
                for sp in candidates[1:]:
                    dist = sp.location.distance(start_tf.location)
                    if dist > max_dist:
                        max_dist = dist
                        goal_tf = sp

        return start_tf, goal_tf

    def _calculate_timeout(self, distance: float) -> int:
        """
        Calculate timeout based on route distance.

        Original NoCrash formula: based on route length / average speed.
        For strict mode, we don't cap at max_steps to allow long routes.
        """
        # Assume 5 m/s average speed (18 km/h, reasonable for urban driving)
        estimated_time = distance / 5.0  # seconds
        # Add 20% buffer + fixed offset for traffic/turns
        timeout_seconds = estimated_time * 1.2 + 20.0
        # Convert to steps
        timeout_steps = int(timeout_seconds / self.env.dt)
        # For strict NoCrash mode, use higher upper bound to allow long routes
        # Original benchmark had no hard cap, timeout was purely distance-based
        upper_bound = max(self.max_steps, 5000) if self.use_predefined_routes else self.max_steps
        return max(100, min(timeout_steps, upper_bound))

    def reset(self):
        """Reset environment with next scenario."""
        self.cur_scenario = self.scenarios[self.scenario_idx % len(self.scenarios)]
        self.scenario_idx += 1

        # Set traffic and weather
        self.env.number_of_vehicles = self.cur_scenario.traffic.vehicles
        self.env.number_of_walkers = self.cur_scenario.traffic.pedestrians
        self.env.world.set_weather(self.cur_scenario.weather_params)

        # Reset base environment (spawns ego at random location first)
        obs = self.env.reset()

        # Get start and goal for this scenario
        start_tf, goal_tf = self._pick_goal(self.cur_scenario)

        # [FIX #1] Teleport ego to predefined start position
        if start_tf is not None and self.use_predefined_routes:
            self.env.teleport_ego(start_tf)
            self.start_location = start_tf.location
        else:
            self.start_location = self.env.ego.get_transform().location

        if goal_tf:
            self.goal_location = goal_tf.location
            # [FIX #3] Use actual route distance, not straight-line
            self.route_distance = self.env.get_route_distance(
                self.start_location, self.goal_location
            )
        else:
            self.goal_location = None
            self.route_distance = 100.0  # Default

        self.step_count = 0

        # Calculate dynamic timeout based on actual route distance
        self.current_timeout = self._calculate_timeout(self.route_distance)

        # Re-fetch observation after teleport
        if start_tf is not None and self.use_predefined_routes:
            obs = self.env._get_obs()

        return obs

    def step(self, action):
        """Execute action and check NoCrash termination conditions."""
        obs, reward, cost, done, info = self.env.step(action)
        self.step_count += 1

        # Calculate distance to goal
        ego_loc = self.env.ego.get_transform().location
        dist_to_goal = ego_loc.distance(self.goal_location) if self.goal_location else 1e9

        # Check termination conditions
        collision = bool(self.env._is_collision)
        off_road = bool(self.env._is_off_road)
        timeout = self.step_count >= self.current_timeout
        reached_goal = dist_to_goal <= self.goal_threshold

        # [FIX #2] NoCrash: collision is immediate failure, prevents success
        # Success ONLY if reached goal WITHOUT collision
        success = reached_goal and not collision

        # Determine termination reason (mutually exclusive for stats)
        # Priority: collision > success > timeout
        done = done or collision or success or timeout

        # Reward shaping and statistics (mutually exclusive counting)
        if collision:
            reward -= 100.0
            self.episode_stats["collision"] += 1
        elif success:
            reward += 100.0
            self.episode_stats["success"] += 1
        elif timeout:
            reward -= 10.0
            self.episode_stats["timeout"] += 1

        if off_road:
            self.episode_stats["off_road"] += 1

        # Calculate route completion
        if self.route_distance > 0:
            distance_traveled = self.start_location.distance(ego_loc) if self.start_location else 0
            route_completion = min(1.0, distance_traveled / self.route_distance)
        else:
            route_completion = 0.0

        # Build info dict
        info = info or {}
        info.update({
            "scenario": self.cur_scenario.name if self.cur_scenario else "unknown",
            "scenario_idx": self.scenario_idx - 1,
            "route_idx": self.cur_scenario.route_idx if self.cur_scenario else -1,
            "traffic": self.cur_scenario.traffic.name if self.cur_scenario else "unknown",
            "weather": self.cur_scenario.weather_name if self.cur_scenario else "unknown",
            "weather_group": self.cur_scenario.weather_group.value if self.cur_scenario else "unknown",
            "dist_to_goal": dist_to_goal,
            "route_distance": self.route_distance,
            "route_completion": route_completion,
            "success": success,
            "collision": collision,
            "off_road": off_road,
            "timeout": timeout,
            "step_count": self.step_count,
            "timeout_limit": self.current_timeout,
        })

        return obs, reward, cost, done, info

    def get_statistics(self) -> dict:
        """Get cumulative episode statistics."""
        total = sum(self.episode_stats.values())
        if total == 0:
            return self.episode_stats

        return {
            **self.episode_stats,
            "success_rate": self.episode_stats["success"] / max(1, self.scenario_idx),
            "collision_rate": self.episode_stats["collision"] / max(1, self.scenario_idx),
            "total_episodes": self.scenario_idx,
        }

    def clear_all_actors(self, filters):
        """Clear all actors (delegate to base env)."""
        return self.env.clear_all_actors(filters)


# ============================================================================
# Convenience Functions for Paper Reproduction
# ============================================================================

def create_nocrash_training_scenarios(
    town: str = "Town01",
    traffic_levels: Sequence[str] = ("empty", "regular", "dense"),
) -> List[NoCrashScenario]:
    """
    Create scenarios for NoCrash TRAINING evaluation.

    Uses 4 training weathers: ClearNoon, WetNoon, HardRainNoon, ClearSunset
    """
    return make_nocrash_scenarios(
        town=town,
        traffic_levels=traffic_levels,
        weather_group=WeatherGroup.TRAINING,
        routes_per_condition=25,
    )


def create_nocrash_newweather_scenarios(
    town: str = "Town01",
    traffic_levels: Sequence[str] = ("empty", "regular", "dense"),
) -> List[NoCrashScenario]:
    """
    Create scenarios for NoCrash NEW WEATHER evaluation.

    Uses 2 new weathers: WetSunset, SoftRainSunset
    """
    return make_nocrash_scenarios(
        town=town,
        traffic_levels=traffic_levels,
        weather_group=WeatherGroup.NEW,
        routes_per_condition=25,
    )


def create_nocrash_newtown_scenarios(
    traffic_levels: Sequence[str] = ("empty", "regular", "dense"),
) -> List[NoCrashScenario]:
    """
    Create scenarios for NoCrash NEW TOWN evaluation (generalization test).

    Uses Town02 with training weathers.
    """
    return make_nocrash_scenarios(
        town="Town02",
        traffic_levels=traffic_levels,
        weather_group=WeatherGroup.TRAINING,
        routes_per_condition=25,
    )
