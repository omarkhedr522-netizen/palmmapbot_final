from backend.robot_state import RobotState
from backend.navigation_manager import NavigationManager
from backend.mission_manager import MissionManager
from backend.tree_mapper import TreeMapper
from backend.coverage_planner import CoveragePlanner


class MissionController:
    def __init__(self):
        self.robot_state = RobotState()
        self.navigation_manager = NavigationManager(self.robot_state)
        self.mission_manager = MissionManager()
        self.tree_mapper = TreeMapper(assumed_tree_distance_m=2.0)
        self.coverage_planner = CoveragePlanner(lane_spacing=2.0)

    def start_mission(self, mission_name: str, area_name: str | None = None, notes: str | None = None):
        """
        Start a mission and save current robot pose as home pose.
        """
        self.robot_state.set_status("starting")

        current = self.robot_state.current_pose
        self.robot_state.set_home_pose(
            current["x"],
            current["y"],
            current["yaw"]
        )

        mission_id = self.mission_manager.create_mission(
            mission_name=mission_name,
            area_name=area_name,
            notes=notes
        )

        self.robot_state.assign_mission(mission_id)
        self.robot_state.set_status("surveying")

        print(f"Mission started: {mission_id}")
        print(f"Home pose saved: {self.robot_state.home_pose}")

        return mission_id

    def survey_farm(self, waypoints: list[dict], gps_lat: float = 29.203451, gps_lon: float = 25.519833):
        """
        Follow waypoints only.
        Real tree detections should be added separately from the camera detector.
        """
        if self.robot_state.current_mission_id is None:
            raise ValueError("No active mission.")

        print("Starting farm survey...")

        for i, wp in enumerate(waypoints, start=1):
            print(f"Survey waypoint {i}/{len(waypoints)}")
            self.navigation_manager.go_to_pose(wp["x"], wp["y"], wp["yaw"])

            # Do NOT create fake detections here.
            # Real detections should only be logged when the external camera
            # and detector actually detect a palm tree.

        print("Survey completed.")

    def survey_rectangular_farm(
        self,
        width: float,
        height: float,
        start_x: float = 0.0,
        start_y: float = 0.0,
        gps_lat: float = 29.203451,
        gps_lon: float = 25.519833
    ):
        """
        Automatically generate a rectangular coverage path and survey it.
        """
        waypoints = self.coverage_planner.generate_rectangular_coverage(
            width=width,
            height=height,
            start_x=start_x,
            start_y=start_y
        )

        print(f"Generated {len(waypoints)} automatic waypoints.")
        self.survey_farm(waypoints, gps_lat=gps_lat, gps_lon=gps_lon)

    def return_home(self):
        """
        Return robot to stored home position.
        """
        self.robot_state.set_status("returning_home")
        self.navigation_manager.return_home()

    def complete_mission(self):
        """
        End the current mission and return robot to idle state.
        """
        mission_id = self.robot_state.current_mission_id
        if mission_id is None:
            raise ValueError("No active mission to complete.")

        self.mission_manager.end_mission(mission_id)
        self.robot_state.set_status("completed")

        print(f"Mission completed: {mission_id}")

        self.robot_state.clear_mission()
        self.robot_state.set_status("idle")

    def abort_mission(self):
        """
        Emergency stop / abort mission.
        """
        self.robot_state.set_status("stopped")
        print("Mission aborted.")

    def get_state(self):
        return self.robot_state.to_dict()

    def record_tree_detection(
        self,
        robot_x: float,
        robot_y: float,
        robot_yaw_rad: float,
        gps_lat: float,
        gps_lon: float,
        confidence: float
    ):
        """
        Record a real palm-tree detection during an active mission.
        """
        if self.robot_state.current_mission_id is None:
            print("Ignoring detection: no active mission.")
            return None

        result = self.tree_mapper.process_tree_detection(
            robot_x=robot_x,
            robot_y=robot_y,
            robot_yaw_rad=robot_yaw_rad,
            gps_lat=gps_lat,
            gps_lon=gps_lon,
            mission_id=self.robot_state.current_mission_id,
            confidence=confidence
        )

        print("Mapped real tree:", result["backend_result"])
        return result
