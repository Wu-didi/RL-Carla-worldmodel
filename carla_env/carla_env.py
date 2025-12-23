# -*- coding: utf-8 -*-


from __future__ import division
import numpy as np
import random
import time
import gym
from gym import spaces
from gym.utils import seeding
import carla
import sys, os; sys.path.append(os.path.dirname(__file__))
# from utils.DynamicPF import pf_base 
# from utils.DynamicPF import DynamicPF
import numpy as np, cv2, io, requests


class CarlaEnv(gym.Env):
    def __init__(self, params):
        self.params = params 
        self.collision_sensor = None
        self.lidar_sensor = None
        self._is_collision = False
        self._is_off_road = False
        self.off_road_counter = 0
        self.number_of_vehicles = params['number_of_vehicles']
        self.number_of_walkers = params['number_of_walkers']
        self.dt = params['dt']
        self.max_time_episode = params['max_time_episode']
        self.max_waypoints = params['max_waypoints']
        self.visualize_waypoints = params['visualize_waypoints']
        self.desired_speed = params['desired_speed']
        self.max_ego_spawn_times = params['max_ego_spawn_times']
        self.view_mode = params['view_mode']
        self.traffic = params['traffic']
        self.lidar_max_range = params['lidar_max_range']
        self.max_nearby_vehicles = params['max_nearby_vehicles']
        self.surrounding_vehicle_spawned_randomly = params['surrounding_vehicle_spawned_randomly']
        self.use_vlm = self.params.get('use_vlm', False)
        self.use_camera = self.params.get('use_camera', False)
        self.use_lidar = self.params.get('use_lidar', False)
        self.use_rgb_seg = self.params.get('use_rgb_seg', False)

        self.rgb_camera = None
        self.segmentation_camera = None
        self.depth_camera = None
        self.rgb_frame = None
        self.rgb_step = 0
        self.camera_width = 0
        self.camera_height = 0

        # 连接到Carla服务器并设置世界
        print('Connecting to Carla server...')
        client = carla.Client('localhost', params['port'])
        client.set_timeout(10.0)
        self.world = client.load_world(params['town'])
        self.world.set_weather(carla.WeatherParameters.ClearNoon)
        print('Connection established!')
    
        # Get all predefined vehicle spawn points from the map
        self.vehicle_spawn_points = list(self.world.get_map().get_spawn_points())
        # Prepare a list to hold spawn points for pedestrians (walkers)
        self.walker_spawn_points = []
        # Randomly generate spawn points for the specified number of pedestrians
        for i in range(self.number_of_walkers):
            spawn_point = carla.Transform()  # Create an empty transform object
            # Try to get a random navigable location in the environment
            loc = self.world.get_random_location_from_navigation()
            # If a valid location is found, use it as a spawn point for a pedestrian
            if loc is not None:
                spawn_point.location = loc
                self.walker_spawn_points.append(spawn_point)
    
    
        self.ego_bp = self._create_vehicle_bluepprint(params['ego_vehicle_filter'], color='255,0,0')
        

    
        self.collision_hist = []
        self.collision_hist_l = 1
        self.collision_bp = self.world.get_blueprint_library().find('sensor.other.collision')
    
        self.lidar_data = None  # Placeholder to store incoming LiDAR data
        self.lidar_height = 0.8  # Height at which the LiDAR is mounted on the vehicle (in meters)
        # Set the position of the LiDAR sensor using a transform (translation only in Z direction)
        self.lidar_trans = carla.Transform(carla.Location(x=0.0, z=self.lidar_height))
        # Get the LiDAR blueprint from Carla's sensor library
        self.lidar_bp = self.world.get_blueprint_library().find('sensor.lidar.ray_cast')
        # Set LiDAR attributes
        self.lidar_bp.set_attribute('channels', '1')  # Use 1 channel to perform a flat 360° horizontal scan
        self.lidar_bp.set_attribute('range', '50')  # Maximum LiDAR range in meters
        self.lidar_bp.set_attribute('rotation_frequency', '10')  # How many full 360° rotations per second
        self.lidar_bp.set_attribute('points_per_second', '10000')  # Total number of points generated per second
        self.lidar_bp.set_attribute('upper_fov', '0')  # upper and lower FOV are both 0 for a flat horizontal scan
        self.lidar_bp.set_attribute('lower_fov', '0')  
    
    
        self.settings = self.world.get_settings()  # Get the current world settings
        self.settings.fixed_delta_seconds = self.dt  # Set the physics simulation step size (in seconds)
                                                      # This ensures consistent time intervals for simulation updates
    
    
        self.reset_step = 0
        self.total_step = 0
        self.last_speed = {}


        # vlm
        if self.use_vlm:
            print("Initializing VLM connection...")
            self._VLM_URL = "http://127.0.0.1:18000/generate"
            self._PROMPT = ("Does the ego vehicle need to brake? only output 0 or 1, 0=no, 1=yes.")  # only output 0 or 1, 0=no, 1=yes.
            self._vlm_sess = requests.Session()

        # Define action and observation spaces for gym compatibility
        # Action: [throttle, steer, brake] - throttle/brake in [0,1], steer in [-1,1]
        self.action_space = spaces.Box(
            low=np.array([0.0, -1.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )

        # Observation space is a Dict space
        obs_spaces = {
            'ego_state': spaces.Box(low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32),
            'waypoints': spaces.Box(low=-np.inf, high=np.inf, shape=(self.max_waypoints * 3,), dtype=np.float32),
            'lane_info': spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32),
        }
        if self.use_camera:
            # RGB image: default 640x384x3, but we use a flexible shape
            obs_spaces['rgb'] = spaces.Box(low=0, high=255, shape=(384, 640, 3), dtype=np.uint8)
        else:
            obs_spaces['rgb'] = spaces.Box(low=0, high=255, shape=(1, 1, 3), dtype=np.uint8)
        self.observation_space = spaces.Dict(obs_spaces)

        print("Finish CarlaEnv initialized.")

    def _resize_keep_ar_cv(self, bgr: np.ndarray, target_max=384) -> np.ndarray:
        h, w = bgr.shape[:2]
        mx = max(h, w)
        if mx <= target_max:
            return bgr
        scale = target_max / mx
        new_w, new_h = int(round(w * scale)), int(round(h * scale))
        return cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

    def _send_frame_bgr(self, bgr: np.ndarray, *, jpeg_quality=80, resize_to=384, timeout=5.0):
        # 可选：先缩到较短边 384，减少编码/传输时间
        if resize_to and resize_to > 0:
            bgr = self._resize_keep_ar_cv(bgr, resize_to)

        # OpenCV 直接编码 JPEG（比 PNG 快，体积也更小）
        ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
        if not ok:
            return None

        bio = io.BytesIO(buf.tobytes())
        files = {"image_file": ("frame.jpg", bio, "image/jpeg")}
        data  = {"prompt": self._PROMPT, "max_tokens": "64"}
        r = self._vlm_sess.post(self._VLM_URL, files=files, data=data, timeout=timeout)
        r.raise_for_status()
        return r.json()

    def _get_qwen_result(self, image: carla.Image, save_stride: int = 5):
        """最快路径：不落盘，按 stride 直接编码JPEG并发送到 Flask。"""
        if self.rgb_step % save_stride == 0:
            
            # image.save_to_disk('0000.png')
            
            h, w = image.height, image.width
            # CARLA 原始是 BGRA 连续字节
            bgra = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(h, w, 4)
            bgr  = bgra[:, :, :3]  # 丢 alpha，保持 BGR 以便 OpenCV 编码更快

            try:
                resp = self._send_frame_bgr(bgr, jpeg_quality=80, resize_to=384, timeout=5.0)
                if resp is not None:
                    self.llm_record = resp['answer']
                    # print(resp['answer'])  # 或者解析 reward 用于你的逻辑 {'answer': '0', 'latency_ms': 69.95}
                    self.qwen_coach_brake_action = int(resp.get('answer', '0'))
                    if "adjust" in resp('answer'):
                        self.qwen_coach_brake_action =  2
                    print(self.qwen_coach_brake_action)
            except Exception as e:
                # 失败就忽略，避免阻塞仿真主循环
                a_temp=0
                # print("[warn] VLM send failed:", e)

        self.rgb_step += 1

    def _save_rgb_image(self, image: carla.Image,
                         save_stride: int = 50):
        """保存RGB图像"""
        # save image every 200 steps
        if self.rgb_step % save_stride == 0:
            image.save_to_disk(f'{self.rgb_dir}/{self.rgb_step:06d}.png')
        self.rgb_step += 1

    def _on_rgb_raw(self, image: carla.Image):
        """Store the latest RGB frame as a numpy array."""
        h, w = image.height, image.width
        arr = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(h, w, 4)
        # CARLA gives BGRA, convert to RGB
        self.rgb_frame = arr[:, :, :3][:, :, ::-1].copy()
        self.rgb_step += 1

    def _on_depth_raw(self, image: carla.Image,
                    near: float = 0.1, far: float = 80.0,
                    save_stride: int = 50,
                    save_16bit_png: bool = False,
                    save_npz: bool = False):
        """
        - 可视化：每 png_stride 帧保存一张 8-bit（可选再存 16-bit）灰度深度图，黑=近、白=远
        - 训练：每帧保存一份 .npy（float32, 米制，裁剪到 [near, far]）
        """
        import os, json
        import numpy as np
        from PIL import Image

        # --- 目录准备 ---
        os.makedirs(self.depth_dir, exist_ok=True)
        os.makedirs(self.depth_dir_vis, exist_ok=True)
        
        h, w = image.height, image.width
        # CARLA raw depth: BGRA (uint8)
        arr = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(h, w, 4)
        B = arr[..., 0].astype(np.float32)
        G = arr[..., 1].astype(np.float32)
        R = arr[..., 2].astype(np.float32)

        # 1) 归一化到 [0,1]
        normalized = (R + G * 256.0 + B * (256.0 ** 2)) / (256.0 ** 3 - 1.0)

        # 2) 转换为米制深度（CARLA depth far plane ≈ 1000m）
        depth_m = normalized * 1000.0

        # 3) 训练使用的“裁剪后米制深度”（float32，无量化）
        depth_m_clip = np.clip(depth_m, near, far).astype(np.float32)

        # 4) 保存 .npy（每帧）
        if self.depth_step % save_stride == 0:
            npy_path = f"{self.depth_dir}/{self.depth_step:06d}.npy"
            np.save(npy_path, depth_m_clip)
        if save_npz:
            # 可选：无损压缩版本，省磁盘（CPU 解压略慢）
            np.savez_compressed(npy_path.replace(".npy", ".npz"), depth=depth_m_clip)

        # 5) 可视化：把 [near, far] 映射到 [0,255]（近=黑，远=白）
        d01 = (depth_m_clip - near) / (far - near)  # 已经在 [0,1]
        if self.depth_step % save_stride == 0:
            gray8 = np.round(d01 * 255.0).astype(np.uint8)
            Image.fromarray(gray8, mode="L").save(
                f"{self.depth_dir_vis}/{self.depth_step:06d}.png"
            )
            if save_16bit_png:
                gray16 = np.round(d01 * 65535.0).astype(np.uint16)
                Image.fromarray(gray16, mode="I;16").save(
                    f"{self.depth_dir}/depth_gray16_{self.depth_step:06d}.png"
                )

        # 6) 首帧写入元数据，避免 near/far 混淆
        if self.depth_step == 0:
            meta = {
                "units": "meters",
                "source": "carla.RawDepth(BGRA)->float",
                "decode": "norm=(R+256G+256^2B)/(256^3-1); depth_m=norm*1000",
                "near": float(near),
                "far": float(far),
                "save_stride": int(save_stride),
                "png_mapping": "[near,far]->[0,255] (8-bit), black=near, white=far",
                "train_file_pattern": "depth_m_XXXXXX.npy (float32, clipped)"
            }
            with open(os.path.join(self.depth_dir, "depth_meta.json"), "w") as f:
                json.dump(meta, f, indent=2)

        self.depth_step += 1

    def _on_seg_raw_5cls(self, image: carla.Image, 
                                                            save_stride: int = 50):
        """
        保存为 5 类单通道 8-bit 灰度语义图：
        0=背景, 1=道路, 2=车道线, 3=车辆, 4=行人
        """
        import numpy as np
        from PIL import Image

        h, w = image.height, image.width

        # 确保是 Raw（像素值=语义ID，而非调色板色彩）
        image.convert(carla.ColorConverter.Raw)

        # BGRA -> 取 R 通道作为类别ID（uint8）
        arr = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(h, w, 4)
        seg_id = arr[:, :, 2]  # R

        # ------- 映射到 5 类 -------
        # CARLA 常见ID：ped=4, road_lines=6, road=7, vehicles=10
        ROAD_ID        = 7
        LANE_ID        = 6
        VEHICLES_ID    = 10
        PEDESTRIANS_ID = 4

        out = np.zeros_like(seg_id, dtype=np.uint8)   # 先全背景=0
        out[seg_id == ROAD_ID]        = 1
        out[seg_id == LANE_ID]        = 2
        out[seg_id == VEHICLES_ID]    = 3
        out[seg_id == PEDESTRIANS_ID] = 4
        # 其他保持0（背景）

        # -------- 保存 8-bit 灰度PNG --------
        if self.segmentation_step % save_stride == 0:
            Image.fromarray(out, mode='L').save(
                f'{self.segmentation_dir}/{self.segmentation_step:06d}.png'
            )
        self.segmentation_step += 1

    def _on_seg_raw(self, image: carla.Image, save_stride: int = 50, save_preview: bool = False):
        """
        保存全部类别（不做重新映射）为 8-bit 灰度 PNG；可选另存彩色可视化 PNG。
        - 灰度 PNG：每个像素值 = CARLA 语义类别 ID（0..255，常用 0..22）
        - 预览 PNG：CityScapesPalette 颜色可视化，仅用于查看
        """
        import os
        import numpy as np
        from PIL import Image

        h, w = image.height, image.width

        # 1) 转换为 Raw：像素值就是语义 ID，存放在 BGRA 的 R 通道
        image.convert(carla.ColorConverter.Raw)
        arr = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(h, w, 4)
        seg_id = arr[:, :, 2].copy()  # 取 R 通道（索引2），复制一份避免后续 convert 覆盖

        if self.segmentation_step % save_stride == 0:
            os.makedirs(self.segmentation_dir, exist_ok=True)

            # 2) 保存“全类别 ID 图”（8-bit 灰度，mode='L'）
            id_path = f"{self.segmentation_dir}/{self.segmentation_step:06d}.png"
            Image.fromarray(seg_id, mode='L').save(id_path)

            # 3) （可选）再转为 CityScapesPalette 做彩色可视化，便于人眼检查
            if save_preview:
                image.convert(carla.ColorConverter.CityScapesPalette)
                arr_vis = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(h, w, 4)
                # CARLA 给的是 BGRA，转成 RGB 方便保存
                rgb = arr_vis[:, :, :3][:, :, ::-1].copy()
                vis_path = f"{self.segmentation_dir}/{self.segmentation_step:06d}_vis.png"
                Image.fromarray(rgb, mode='RGB').save(vis_path)

        self.segmentation_step += 1


    def reset(self):
        # Stop and destroy the collision sensor if it exists
        if self.collision_sensor is not None:
            try:
                self.collision_sensor.stop()
                self.collision_sensor.destroy()
            except:
                pass
            self.collision_sensor = None
    
        # Stop and destroy the LiDAR sensor if it exists
        if self.lidar_sensor is not None:
            try:
                self.lidar_sensor.stop()
                self.lidar_sensor.destroy()
            except:
                pass
            self.lidar_sensor = None
    
            # 清除并销毁现有传感器
        if self.rgb_camera:
            self.rgb_camera.stop()
            self.rgb_camera.destroy()
        if self.segmentation_camera:
            self.segmentation_camera.stop()
            self.segmentation_camera.destroy()
        if self.depth_camera:
            self.depth_camera.stop()
            self.depth_camera.destroy()
        # 
    
    
        # Reset collision and off-road status flags
        self._is_collision = False
        self._is_off_road = False
    
        self._set_synchronous_mode(False)  # Switch back to asynchronous mode
        self._clear_all_actors([
            'sensor.other.collision',
            'sensor.lidar.ray_cast',
            'sensor.camera.rgb',
            'vehicle.*',
            'controller.ai.walker',
            'walker.*'
        ])  # Remove all specified actors from the world

        # Spawn surrounding vehicles
        random.shuffle(self.vehicle_spawn_points)
        count = self.number_of_vehicles
        self.spawned_vehicles = []
        self.used_spawn_points = []
        
        if count > 0:
            for spawn_point in self.vehicle_spawn_points:
                vehicle = self._try_spawn_random_vehicle_at(spawn_point, number_of_wheels=[4])
                if vehicle:
                    self.spawned_vehicles.append(vehicle)  # Record the spawned vehicle
                    self.used_spawn_points.append(spawn_point)  # Mark spawn point as used
                    count -= 1
                if count <= 0:
                    break
        # print(f"Surrounding vehicles number is {len(self.spawned_vehicles)}")

        # Spawn pedestrians
        count = self.number_of_walkers

        # Dynamically generate walker spawn points if needed
        if count > 0:
            self.walker_spawn_points = []
            for _ in range(count * 2):  # Generate extra points for fallback
                spawn_point = carla.Transform()
                loc = self.world.get_random_location_from_navigation()
                if loc is not None:
                    spawn_point.location = loc
                    self.walker_spawn_points.append(spawn_point)

        if count > 0 and len(self.walker_spawn_points) > 0:
            random.shuffle(self.walker_spawn_points)
            for spawn_point in self.walker_spawn_points:
                if self._try_spawn_random_walker_at(spawn_point):
                    count -= 1
                if count <= 0:
                    break

            # Try random spawn points until all pedestrians are spawned
            max_attempts = count * 10  # Prevent infinite loop
            attempts = 0
            while count > 0 and attempts < max_attempts and len(self.walker_spawn_points) > 0:
                if self._try_spawn_random_walker_at(random.choice(self.walker_spawn_points)):
                    count -= 1
                attempts += 1

        # Get actors' polygon list
        # Calculate and collect the bounding polygons (e.g., four corners) of surrounding vehicles and pedestrians
        self.vehicle_polygons = []
        vehicle_poly_dict = self._get_actor_polygons('vehicle.*')
        self.vehicle_polygons.append(vehicle_poly_dict)
        
        self.walker_polygons = []
        walker_poly_dict = self._get_actor_polygons('walker.*')
        self.walker_polygons.append(walker_poly_dict)

        # Spawn the ego vehicle
        ego_spawn_times = 0
        max_total_attempts = self.max_ego_spawn_times * 3  # Hard limit to prevent infinite loop

        while True:
            if ego_spawn_times > self.max_ego_spawn_times:
                # Too many failed attempts, clear actors and retry from scratch
                print(f"[WARN] Failed to spawn ego after {ego_spawn_times} attempts, clearing actors...")
                self._clear_all_actors(['vehicle.*', 'walker.*', 'controller.ai.walker'])
                self.spawned_vehicles = []
                self.used_spawn_points = []
                ego_spawn_times = 0  # Reset counter after clearing
                time.sleep(1.0)  # Give CARLA time to clean up

            if ego_spawn_times > max_total_attempts:
                raise RuntimeError(f"Failed to spawn ego vehicle after {max_total_attempts} total attempts")

            # Select a spawn point for the ego vehicle by excluding locations used by nearby vehicles
            available_spawn_points = [
                sp for sp in self.vehicle_spawn_points if sp not in self.used_spawn_points
            ]

            if len(available_spawn_points) > 0:
                transform = random.choice(available_spawn_points)
            else:
                transform = random.choice(self.vehicle_spawn_points)

            # Try to spawn the ego vehicle at the selected location
            if self._try_spawn_ego_vehicle_at(transform):
                break  # Successfully spawned the ego vehicle
            else:
                ego_spawn_times += 1
                time.sleep(0.1)

        self.stationary_penalty_timer = 0.0   # 前方无车累计静止时长（奖励函数中）
        self.stationary_reward_timer = 0.0   # 前方有车累计静止时长（奖励函数中）
        self.outoflane_timer = 0.0    # 累计偏离车道时长（奖励函数中）
        # self.pf =  DynamicPF(alpha=0.1336)
        self.llm_record ='0'


        # ============================================================
        # 让 CARLA 的自动驾驶系统接管 ego
        # self.tm = carla.Client('localhost', self.params['port']).get_trafficmanager()
        # self.tm.set_synchronous_mode(True)
        # self.tm.set_global_distance_to_leading_vehicle(4.0)          # 跟车距离大
        # self.tm.keep_right_rule_percentage(self.ego, 50)
        # self.tm.vehicle_percentage_speed_difference(self.ego, 50)
        # self.ego.set_autopilot(True, self.tm.get_port())
        # ============================================================



        if self.traffic == 'off':
            # Set all traffic lights to green and freeze them
            for actor in self.world.get_actors().filter('traffic.traffic_light*'):
                actor.set_state(carla.TrafficLightState.Green)
                actor.freeze(True)
        elif self.traffic == 'on':
            # Allow traffic lights to work normally
            for actor in self.world.get_actors().filter('traffic.traffic_light*'):
                actor.freeze(False)

        # Add collision sensor
        self.collision_sensor = self.world.spawn_actor(
            self.collision_bp,
            carla.Transform(),  # Attach at the center of the ego vehicle (no offset)
            attach_to=self.ego
        )
        
        # Start listening for collision events
        self.collision_sensor.listen(
            lambda event: get_collision_hist(event)  # When a collision event happens, pass the event to get_collision_hist()
        )

        def get_collision_hist(event):
            impulse = event.normal_impulse  # Get the collision impulse (a 3D vector)
            intensity = np.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2)  # Calculate collision intensity (vector norm)
            self.collision_hist.append(intensity)  # Record the collision intensity
            if len(self.collision_hist) > self.collision_hist_l:
                self.collision_hist.pop(0)  # Keep only the latest collision records (FIFO)
        
        # Initialize collision history list
        # Clear collision history after each episode because in gym-carla setup,
        # a collision typically triggers episode termination and reset.
        self.collision_hist = []

        # Add lidar sensor (optional)
        if self.use_lidar:
            self.lidar_data = []
            self.lidar_sensor = self.world.spawn_actor(self.lidar_bp, self.lidar_trans, attach_to=self.ego)
            self.lidar_sensor.listen(lambda data: get_lidar_data(data))
        def get_lidar_data(data):
            self.lidar_data = data
            
                # 重新初始化相机传感器
        # Define the cameras separately
        if self.use_camera:
            self.rgb_frame = None
            self.rgb_step = 0
            self.rgb_camera_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
            self.rgb_camera_bp.set_attribute('image_size_x', '640')
            self.rgb_camera_bp.set_attribute('image_size_y', '384')
            self.rgb_camera_bp.set_attribute('fov', '120')
            self.rgb_camera_transform = carla.Transform(carla.Location(x=1.5, z=2.5))
            self.camera_width = int(self.rgb_camera_bp.get_attribute('image_size_x').as_int())
            self.camera_height = int(self.rgb_camera_bp.get_attribute('image_size_y').as_int())
            self.rgb_camera = self.world.spawn_actor(self.rgb_camera_bp, self.rgb_camera_transform, attach_to=self.ego)
            self.rgb_camera.listen(lambda image: self._on_rgb_raw(image))
            
            if self.use_rgb_seg:
                # segmentation camera
                self.segmentation_camera_bp = self.world.get_blueprint_library().find('sensor.camera.semantic_segmentation')
                self.segmentation_camera_bp.set_attribute('image_size_x', '640')
                self.segmentation_camera_bp.set_attribute('image_size_y', '384')
                self.segmentation_camera_bp.set_attribute('fov', '120')
                self.segmentation_camera_transform = carla.Transform(carla.Location(x=1.5, z=2.5))
                self.segmentation_camera = self.world.spawn_actor(self.segmentation_camera_bp, self.segmentation_camera_transform, attach_to=self.ego)

                # depth camera
                self.depth_camera_bp = self.world.get_blueprint_library().find('sensor.camera.depth')
                self.depth_camera_bp.set_attribute('image_size_x', '640')
                self.depth_camera_bp.set_attribute('image_size_y', '384')
                self.depth_camera_bp.set_attribute('fov', '120')
                self.depth_camera_transform = carla.Transform(carla.Location(x=1.5, z=2.5))
                self.depth_camera = self.world.spawn_actor(self.depth_camera_bp, self.depth_camera_transform, attach_to=self.ego)
            


    

        # Update timesteps
        self.time_step = 1  # Indicates a new episode has started
        self.reset_step += 1  # Count how many resets have occurred
        
        # Enable autopilot for all surrounding vehicles
        for vehicle in self.spawned_vehicles:
            vehicle.set_autopilot()
        
        self._set_synchronous_mode(True)  # Switch to synchronous mode for simulation
        self.world.tick()  # Advance the simulation by one tick     

        self.lane_yaw_list = []
        return self._get_obs()  # Return the initial observation after reset

    def teleport_ego(self, transform: carla.Transform):
        """
        Teleport the ego vehicle to a specific transform.
        Used by NoCrashEnv to set predefined start positions.
        """
        if self.ego is None:
            return False
        # Disable physics temporarily for clean teleport
        self.ego.set_simulate_physics(False)
        self.ego.set_transform(transform)
        self.world.tick()
        self.ego.set_simulate_physics(True)
        # Reset velocity to zero
        self.ego.set_target_velocity(carla.Vector3D(0, 0, 0))
        self.world.tick()
        return True

    def get_route_distance(self, start_loc: carla.Location, end_loc: carla.Location) -> float:
        """
        Calculate actual route distance using CARLA's GlobalRoutePlanner.
        Returns distance in meters along the road network.
        Falls back to heuristic (1.4x straight-line) if GlobalRoutePlanner unavailable.
        """
        # Use cached flag to avoid repeated import attempts and warnings
        if not hasattr(self, '_grp_available'):
            self._grp_available = None  # None = not checked yet

        if self._grp_available is False:
            # Already know GlobalRoutePlanner is unavailable, use heuristic
            return start_loc.distance(end_loc) * 1.4

        try:
            # Try multiple import paths for CARLA's GlobalRoutePlanner
            try:
                from agents.navigation.global_route_planner import GlobalRoutePlanner
            except ImportError:
                from carla.agents.navigation.global_route_planner import GlobalRoutePlanner

            grp = GlobalRoutePlanner(self.world.get_map(), sampling_resolution=1.0)
            route = grp.trace_route(start_loc, end_loc)
            if not route:
                return start_loc.distance(end_loc) * 1.4
            # Sum up segment distances
            total_dist = 0.0
            for i in range(len(route) - 1):
                wp1 = route[i][0].transform.location
                wp2 = route[i + 1][0].transform.location
                total_dist += wp1.distance(wp2)
            self._grp_available = True
            return total_dist
        except ImportError:
            # GlobalRoutePlanner not available - warn once, then use heuristic
            if self._grp_available is None:
                print("[WARN] GlobalRoutePlanner not found, using heuristic (1.4x straight-line). "
                      "For accurate route distance, add CARLA PythonAPI/carla/agents to PYTHONPATH.")
            self._grp_available = False
            return start_loc.distance(end_loc) * 1.4
        except Exception as e:
            # Other errors - use heuristic
            if self._grp_available is None:
                print(f"[WARN] Route planning failed ({e}), using heuristic")
            self._grp_available = False
            return start_loc.distance(end_loc) * 1.4

    #expert step
    def step_sample(self):

        # 1. 让仿真推进一帧
        self.world.tick()

        # 2. 读取 CARLA 自动驾驶系统此刻真实下发的控制量
        control = self.ego.get_control()
        action = np.array([
            control.throttle,
            control.steer,
            control.brake
        ], dtype=np.float32)

        # 3. 其余逻辑保持不变
        if self.view_mode == 'top':
            spectator = self.world.get_spectator()
            spectator.set_transform(
                carla.Transform(
                    self.ego.get_transform().location + carla.Location(z=40),
                    carla.Rotation(pitch=-90)
                )
            )
        elif self.view_mode == 'follow':
            spectator = self.world.get_spectator()
            t = self.ego.get_transform()
            cam = t.transform(carla.Location(x=-6, z=3))
            spectator.set_transform(carla.Transform(cam, carla.Rotation(pitch=-10, yaw=t.rotation.yaw)))

        self.time_step += 1
        self.total_step += 1

        obs = self._get_obs()
        done = self._terminal()
        reward = self._get_reward(obs, done)
        cost = self._get_cost(obs)
        info = {'is_collision': self._is_collision, 'is_off_road': self._is_off_road}

        return obs, reward, cost, done, info, action   # 多返回一个 action

    #mormal train step
    def step(self, action):
        throttle = float(np.clip(action[0], 0.0, 1.0))
        steer    = float(np.clip(action[1], -1.0, 1.0))
        brake    = float(np.clip(action[2], 0.0, 1.0))

        # Apply control
        control = carla.VehicleControl(throttle=throttle, steer=steer, brake=brake)
        self.ego.apply_control(control)

        self.world.tick()

        # Set spectator (camera) view
        spectator = self.world.get_spectator()
        transform = self.ego.get_transform()
        if self.view_mode == 'top':
            # Top-down view (bird's eye)
            spectator.set_transform(
                carla.Transform(
                    transform.location + carla.Location(z=40),
                    carla.Rotation(pitch=-90)
                )
            )
        elif self.view_mode == 'follow':
            # Follow view (behind and above the ego vehicle)
            cam_location = transform.transform(carla.Location(x=-6.0, z=3.0))  # 6 meters behind, 3 meters above
            cam_rotation = carla.Rotation(pitch=-10, yaw=transform.rotation.yaw, roll=0)
            spectator.set_transform(carla.Transform(cam_location, cam_rotation))

        # Update timesteps
        self.time_step += 1
        self.total_step += 1

        obs = self._get_obs()
        done = self._terminal()
        reward = self._get_reward(obs, done)
        cost = self._get_cost(obs)

        # state information
        info = {
          'is_collision': self._is_collision,
          'is_off_road': self._is_off_road
        }
        return (obs, reward, cost, done, info)

    def _create_vehicle_bluepprint(self, actor_filter, color=None, number_of_wheels=[4]):
        """Create a vehicle blueprint based on the given filter and wheel number.

        Args:
            actor_filter (str): Filter string to select vehicle types, e.g., 'vehicle.lincoln*' 
                                ('*' matches a series of models).
            color (str, optional): Desired vehicle color. Randomly chosen if None.
            number_of_wheels (list): A list of acceptable wheel numbers (default is [4]).

        Returns:
            bp (carla.ActorBlueprint): A randomly selected blueprint matching the criteria.
        """
        # Get all blueprints matching the actor filter
        blueprints = self.world.get_blueprint_library().filter(actor_filter)
        blueprint_library = []

        # Further filter blueprints based on the number of wheels
        # Keeping number_of_wheels as a list makes it flexible to match multiple types (e.g., cars, trucks)
        for nw in number_of_wheels:
            blueprint_library += [x for x in blueprints if int(x.get_attribute('number_of_wheels')) == nw]

        # Randomly select one blueprint from the filtered list
        bp = random.choice(blueprint_library)

        # Set the vehicle color
        if bp.has_attribute('color'):
            if not color:
                color = random.choice(bp.get_attribute('color').recommended_values)
            bp.set_attribute('color', color)

        return bp

    def _set_synchronous_mode(self, synchronous=True):

        """Enable or disable synchronous mode for the simulation.
        Args:
            synchronous (bool):
                True to enable synchronous mode (server waits for client each frame),
                False to disable and run in asynchronous mode (default is True).
        """
        self.settings.synchronous_mode = synchronous  # Set the synchronous mode
        self.world.apply_settings(self.settings)  # Apply the updated settings to the world

    def _try_spawn_random_vehicle_at(self, transform, number_of_wheels=[4]):
        """Try to spawn a surrounding vehicle at a specific transform.
    
        Args:
            transform (carla.Transform): Location and orientation where the vehicle should be spawned.
            number_of_wheels (list): Acceptable number(s) of wheels for the vehicle blueprint.
            random_vehicle (bool): 
                False to use Tesla Model 3 with a blue color,
                True to randomly select a vehicle model and color (default).
    
        Returns:
            carla.Actor or None: Spawned vehicle actor if successful, otherwise None.
        """
        if self.surrounding_vehicle_spawned_randomly:
            # Randomly choose any vehicle blueprint
            blueprint = self._create_vehicle_bluepprint('vehicle.*', number_of_wheels=number_of_wheels)
            if blueprint.has_attribute('color'):
                color = random.choice(blueprint.get_attribute('color').recommended_values)
                blueprint.set_attribute('color', color)
        else:
            # Fixed: Tesla Model 3 with blue color
            blueprint = self._create_vehicle_bluepprint('vehicle.tesla.model3', color='0,0,255', number_of_wheels=number_of_wheels)
        
        blueprint.set_attribute('role_name', 'autopilot')  # Set the vehicle to autopilot mode
    
        # Try to spawn the vehicle
        vehicle = self.world.try_spawn_actor(blueprint, transform)
    
        return vehicle if vehicle is not None else None

    def _try_spawn_random_walker_at(self, transform):
        """Try to spawn a walker at a specific transform with a random blueprint.
    
        Args:
            transform (carla.Transform): Location and orientation where the walker should be spawned.
    
        Returns:
            Bool: True if spawn is successful, False otherwise.
        """
        # Randomly select a walker blueprint
        walker_bp = random.choice(self.world.get_blueprint_library().filter('walker.*'))
    
        # Make the walker vulnerable (can be affected by collisions)
        if walker_bp.has_attribute('is_invincible'):
            walker_bp.set_attribute('is_invincible', 'false')
    
        # Try to spawn the walker actor
        walker_actor = self.world.try_spawn_actor(walker_bp, transform)
    
        if walker_actor is not None:
            # Spawn a controller for the walker
            walker_controller_bp = self.world.get_blueprint_library().find('controller.ai.walker')
            walker_controller_actor = self.world.spawn_actor(walker_controller_bp, carla.Transform(), walker_actor)
    
            # Start the controller to control the walker
            walker_controller_actor.start()
    
            # Move the walker to a random location
            walker_controller_actor.go_to_location(self.world.get_random_location_from_navigation())
    
            # Set a random walking speed between 1 m/s and 2 m/s (default is 1.4 m/s)
            walker_controller_actor.set_max_speed(1 + random.random())
    
            return True  # Spawn and initialization successful
    
        return False  # Failed to spawn

    def _try_spawn_ego_vehicle_at(self, transform):
        """Try to spawn the ego vehicle at a specific transform.
    
        Args:
            transform (carla.Transform): Target location and orientation.
    
        Returns:
            Bool: True if spawn is successful, False otherwise.
        """
        vehicle = None
        overlap = False
    
        # Check if ego position overlaps with surrounding vehicles
        for idx, poly in self.vehicle_polygons[-1].items():  # Use .items() to iterate over dict keys and values
            poly_center = np.mean(poly, axis=0)
            ego_center = np.array([transform.location.x, transform.location.y])
            dis = np.linalg.norm(poly_center - ego_center)
    
            if dis > 8:
                continue
            else:
                overlap = True
                break
    
        # If no overlap, try to spawn the ego vehicle
        if not overlap:
            vehicle = self.world.try_spawn_actor(self.ego_bp, transform)
    
        if vehicle is not None:
            self.ego = vehicle
            return True
    
        return False

    def _get_actor_polygons(self, filt):
        """Get the bounding box polygon of actors.
    
        Args:
            filt: the filter indicating what type of actors we'll look at.
    
        Returns:
            actor_poly_dict: a dictionary containing the bounding boxes of specific actors.
        """
        actor_poly_dict = {}
        for actor in self.world.get_actors().filter(filt): 
            # Get all actors in the current world that meet the filt condition, such as vehicle.* or walker.*
            # Note that self.world.get_actors() retrieves all objects in the current simulation environment (vehicles, pedestrians, traffic lights, etc.).
    
            # Get x, y and yaw of the actor
            trans = actor.get_transform() 
            # Get the actor's global position (location) and heading angle (rotation).
    
            x = trans.location.x 
            # x, y are the actor's global coordinates.
    
            y = trans.location.y
            yaw = trans.rotation.yaw / 180 * np.pi 
            # yaw is the heading angle, whose unit is degrees, needs to be converted to radians (multiply by pi/180) to facilitate subsequent matrix calculations.
    
            # Get length and width
            bb = actor.bounding_box 
            # Get the "half-length" boundary.
    
            l = bb.extent.x 
            # "Half-length" in the x-direction (the distance from the center to the edge).
    
            w = bb.extent.y
            # "Half-width" in the y-direction (the distance from the center to the edge).
    
            # Get bounding box polygon in the actor's local coordinate
            # Take the vehicle center as the origin, build a local coordinate system, and list four corner points:
            # (l, w): front right corner, (l, -w): rear right corner, (-l, -w): rear left corner, (-l, w): front left corner
            poly_local = np.array([
                [l, w], [l, -w], [-l, -w], [-l, w]
            ]).transpose() 
            # Transpose() here is to facilitate subsequent matrix operations,
            # changing the matrix from (4,2) to (2,4) format.
    
            # Get rotation matrix to transform to global coordinate
            # This is a standard 2D rotation matrix: used to transform points from the local coordinate system to the global coordinate system.
            # Rotation matrix R = [cosθ  -sinθ]
            #                     [sinθ   cosθ]
            R = np.array([
                [np.cos(yaw), -np.sin(yaw)],
                [np.sin(yaw), np.cos(yaw)]
            ])
    
            # Get global bounding box polygon
            poly = np.matmul(R, poly_local).transpose() + np.repeat([[x, y]], 4, axis=0) 
            # np.matmul(R, poly_local):
            # Transform the four corners (in the local coordinate system) into the global direction through the rotation matrix.
            # After .transpose(), it becomes (4,2) format (one point per row).
            # + np.repeat([[x,y]],4,axis=0):
            # Add the global position offset of the vehicle/pedestrian to each point
            # to obtain the final polygon coordinates in the global coordinate system.
    
            actor_poly_dict[actor.id] = poly 
            # Store the calculated poly (a 4×2 array, four corner points in global coordinates)
            # with actor.id as the key into actor_poly_dict.
            # After returning, the entire dictionary structure:
            # {
            # actor_id_1: np.array([[x1,y1],[x2,y2],[x3,y3],[x4,y4]]),
            # actor_id_2: np.array([[x1,y1],[x2,y2],[x3,y3],[x4,y4]]),
            # ...
            # }
    
        return actor_poly_dict

    def _get_obs(self):
        obs = {}
        ego_transform = self.ego.get_transform()
        ego_x = ego_transform.location.x
        ego_y = ego_transform.location.y
        ego_yaw = np.deg2rad(ego_transform.rotation.yaw)
        
        # ========================== Ego vehicle state extraction =======================================
        velocity = self.ego.get_velocity()
        speed = np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
        angular_velocity = self.ego.get_angular_velocity()
        acceleration = self.ego.get_acceleration()
        
        front_vehicle_distance = 30.0
        relative_speed = speed
        
        min_front_distance = 20.0  # Search range threshold
        vehicle_list = self.world.get_actors().filter('vehicle.*')
        
        for vehicle in vehicle_list:
            if vehicle.id == self.ego.id:
                continue
        
            transform = vehicle.get_transform()
            rel_x = transform.location.x - ego_x
            rel_y = transform.location.y - ego_y
        
            local_x = np.cos(-ego_yaw) * rel_x - np.sin(-ego_yaw) * rel_y
            local_y = np.sin(-ego_yaw) * rel_x + np.cos(-ego_yaw) * rel_y
        
            if 0 < local_x < min_front_distance and abs(local_y) < 2.5:
                d = np.sqrt(local_x**2 + local_y**2)
                if front_vehicle_distance == 30.0 or d < front_vehicle_distance:
                    front_vehicle_distance = d
                    front_speed = vehicle.get_velocity()
                    front_speed_mag = np.sqrt(front_speed.x**2 + front_speed.y**2 + front_speed.z**2)
                    relative_speed = speed - front_speed_mag
        
        ego_state = np.array([
            ego_x,
            ego_y,
            ego_yaw,
            speed,
            angular_velocity.z,
            acceleration.x,
            acceleration.y,
            front_vehicle_distance,
            relative_speed
        ], dtype=np.float32)
        
        obs['ego_state'] = ego_state

# ========================== Current reference waypoints (up to N waypoints) ==========================
        max_waypoints = self.max_waypoints
        world_map = self.world.get_map()
        waypoint = world_map.get_waypoint(self.ego.get_location())
        waypoints_array = np.zeros((max_waypoints, 3), dtype=np.float32)
        
        for i in range(max_waypoints):
            if waypoint is None:
                break
        
            loc = waypoint.transform.location
            yaw = waypoint.transform.rotation.yaw
        
            # Transform waypoint location into ego-centric local coordinates
            local_x = np.cos(-ego_yaw) * (loc.x - ego_x) - np.sin(-ego_yaw) * (loc.y - ego_y)
            local_y = np.sin(-ego_yaw) * (loc.x - ego_x) + np.cos(-ego_yaw) * (loc.y - ego_y)
            yaw_relative = np.deg2rad(yaw) - ego_yaw  # Relative heading
        
            waypoints_array[i] = [local_x, local_y, yaw_relative]
        
            # Move to the next waypoint 2.0 meters ahead
            next_waypoints = waypoint.next(2.0)
            waypoint = next_waypoints[0] if next_waypoints else None
        
        obs['waypoints'] = waypoints_array.flatten()

# ============================= Lane boundary information =========================================
        ego_tf = self.ego.get_transform()
        forward_vec = ego_tf.get_forward_vector()
        feel = 1.0 # 车头前0米
        look_ahead_pos = ego_tf.location + forward_vec * feel 
        
        waypoint_center = world_map.get_waypoint(
            look_ahead_pos, project_to_road=True, lane_type=carla.LaneType.Driving
        )
        
        if waypoint_center is None:
            # If no valid driving lane is found
            obs['lane_info'] = np.array([0.0, 0.0], dtype=np.float32)
        else:
            lane_width = waypoint_center.lane_width
            center_location = waypoint_center.transform.location
        
            # Calculate lateral offset between ego position and lane centerline
            lateral_offset = np.linalg.norm(
                np.array([
                    look_ahead_pos.x - center_location.x,
                    look_ahead_pos.y - center_location.y
                ])
            )
        
            obs['lane_info'] = np.array([lane_width, lateral_offset], dtype=np.float32)

        # =============================== RGB observation for latent encoder ===============================
        if self.use_camera:
            if self.rgb_frame is not None:
                obs['rgb'] = self.rgb_frame.copy()
            else:
                h = self.camera_height if self.camera_height > 0 else 1
                w = self.camera_width if self.camera_width > 0 else 1
                obs['rgb'] = np.zeros((h, w, 3), dtype=np.uint8)
        else:
            obs['rgb'] = np.zeros((1, 1, 3), dtype=np.uint8)

# =============================== Visualize current reference waypoints ===============================
        if self.visualize_waypoints:
            for i in range(max_waypoints):
                wx, wy, _ = waypoints_array[i]
        
                # Transform from ego-centric local coordinates to global coordinates
                gx = np.cos(ego_yaw) * wx - np.sin(ego_yaw) * wy + ego_x
                gy = np.sin(ego_yaw) * wx + np.cos(ego_yaw) * wy + ego_y
        
                self.world.debug.draw_point(
                    carla.Location(x=gx, y=gy, z=ego_transform.location.z + 1.0),
                    size=0.1,
                    life_time=0.5,
                    color=carla.Color(0, 255, 0)  # Green points
                )
        return obs

    def _get_reward(self, obs, done):
        reward = 0.0
    
        # 1. Forward driving reward (within speed limit and along lane direction)
        speed = obs['ego_state'][3]
        if 4.0 <= speed <= 6.0:
            reward += 16 - 4 * np.abs(speed - 5)
        elif speed <= 4.0: 
            reward += 3.0 * speed
        elif 6.0 <= speed <= self.desired_speed:
            reward = 12.0
        else:
            reward += -3.0 * (speed - self.desired_speed)

        a_lon = abs(obs['ego_state'][5])
        reward += -0.5 * max(a_lon - 0.5, 0.0)
        # print('急刹车加速惩罚',-0.5 * max(a_lon - 0.5, 0.0))   

        # 2. Lane deviation penalty (penalize offset from lane center)
        lane_width, lateral_offset = obs['lane_info']
        reward += -1.0 * max(lateral_offset - 0.25 , 0.0)
        # print('不按中心线惩罚',-1.2 * max(lateral_offset - 0.25 , 0.0))
    

        # 3. Smooth driving penalty (lateral acceleration penalty)
        a_lat = obs['ego_state'][6]
        waypoint = self.world.get_map().get_waypoint(
        self.ego.get_location(), project_to_road=True,
        lane_type=carla.LaneType.Driving)
        if waypoint is not None and not waypoint.is_intersection:
            reward += -0.5 * max(abs(a_lat) - 1.0, 0.0)
            # print('瞬时横向偏移惩罚',-1 * max(abs(a_lat) - 1.0, 0.0) )
            # print('前瞻横向偏移量',abs(a_lat))

            if max(abs(a_lat) - 1.0, 0.0) != 0:
                self.outoflane_timer += self.dt
            else:
                self.outoflane_timer = 0
            if self.outoflane_timer >= 1.5:      # 超过 1.5 秒大幅度偏离车道
                reward += -1 
                # print('长期横向偏移惩罚',-1)

        # 4. Stationary penalty (if no vehicle ahead but ego is barely moving)
        front_distance = obs['ego_state'][7]
        #4.1 前方无车不动
        if front_distance > 10.0 and speed < 0.1 :
            self.stationary_penalty_timer += self.dt
        else:
            self.stationary_penalty_timer = 0.0

        if self.stationary_penalty_timer >= 5.0:      # 超过 5 秒静止
            reward += -1.0  
            # print('前方无车静止惩罚')

        #4.2 前方塞车不动
        if 2.0 < front_distance < 8.0 and speed < 0.1 :
            self.stationary_reward_timer += self.dt
        else:
            self.stationary_reward_timer = 0.0

        if self.stationary_reward_timer >= 5.0:      # 超过 5 秒静止
            reward += 10.0  
            # print('前方有车静止奖励')

        #4.3 前方要紧急刹车 
        if front_distance < 2.0 and speed > 0.1 :
            reward += -1.0  
        if front_distance < 1.0 and speed != 0 :
            reward += -3.0 
            
        #4.4 大语言模型
        if self.llm_record =='1' and speed  > 2.0:
            reward += -2.0
            # print('大语言模型惩罚')


        # 5. Collision penalty
        if self._is_collision:
            reward += -200.0
    
        # 6. Off-road penalty
        if self._is_off_road:
            reward += -200.0
    
        # 7. Unstable angle penalty
        waypoint = self.world.get_map().get_waypoint(
        self.ego.get_location(), project_to_road=True,
        lane_type=carla.LaneType.Driving)

        if waypoint is not None and not waypoint.is_intersection:
            lane_yaw = np.degrees(obs['waypoints'][2])% 360
            lane_yaw = (lane_yaw + 180) % 360 - 180
            self.lane_yaw_list.append(lane_yaw)
            # print('lane_yaw',lane_yaw)

            if len(self.lane_yaw_list) >= 2:
                last_two = self.lane_yaw_list[-2:]
                delta2 = abs(last_two[-1] - last_two[-2])
                if delta2 > 2:
                    reward += -1.0
                    # print('一阶导过大')
            else:
                delta2 = None  

            if len(self.lane_yaw_list) >= 3:
                last_three = self.lane_yaw_list[-3:]
                delta21 = abs(last_three[1] - last_three[0])
                delta32 = abs(last_three[2] - last_three[1])
                delta_delta = abs(delta32 - delta21)
                if delta_delta > 1:
                    reward += -1.0
                    # print('二阶导过大')
            else:
                delta_delta = None  
            
        # 8.potential field penalty
        # total_potential = []
        # distance = []
        # reward_p = 0.0
        # nv = self.max_nearby_vehicles
        # nearby = obs['nearby_vehicles'].reshape(nv, 6) 
        # for i in range(nv):
        #     lx, ly, rel_yaw, spd, acc, dapf = nearby[i]
        #     if lx==0 and ly==0:
        #         continue
        #     total_potential.append(dapf)
        #     distance.append(np.sqrt(lx**2+ly**2))
        # if total_potential:
        #     potential = max(total_potential)
        #     distance_write = min(distance)
        # else:
        #     potential = 0.0
        #     distance_write = 100
        # if potential >=0.60:
        #     reward_p = -0.1 
        #     # print('potential',potential)
        #     # print('distance_write',distance_write)
        # reward += -reward_p
        return reward

    """
    #def _get_reward_for_mulititrain(self, obs, done):
    """

    def _get_cost(self, obs):
        """
        Calculate the constraint cost for safe reinforcement learning.
    
        This cost is only used in safe RL settings and does not affect the reward function.
        It penalizes collisions, off-road events, and overspeeding behavior.
        
        Args:
            obs: The current observation dictionary.
    
        Returns:
            cost (float): The accumulated constraint cost.
        """
        cost = 0.0
    
        # 1. Collision cost
        if self._is_collision:
            cost += 20.0
    
        # 2. Off-road cost
        if self._is_off_road:
            cost += 20.0
    
        # 3. Overspeed cost
        speed = obs['ego_state'][3]
        if speed > self.desired_speed:
            cost += (speed - self.desired_speed) / self.desired_speed  # Cost proportional to overspeed percentage
    
        return cost

    def _terminal(self):
        ego_transform = self.ego.get_transform()
        ego_x = ego_transform.location.x
        ego_y = ego_transform.location.y
    
        # 1. Collision termination
        if len(self.collision_hist) > 0:
            self._is_collision = True
            print('Collision occurred')
            return True
    
        # 2. Exceeding maximum allowed timesteps
        if self.time_step > self.max_time_episode:
            print('Exceeded maximum timesteps')
            return True
    
        # # 3. Goal reaching termination (optional)
        # if self.dests is not None:
        #     for dest in self.dests:
        #         if np.sqrt((ego_x - dest[0])**2 + (ego_y - dest[1])**2) < 4:
        #             return True
    
        # 4. Check if the current lane is a drivable lane
        waypoint = self.world.get_map().get_waypoint(
            self.ego.get_location(),
            project_to_road=True,
            lane_type=carla.LaneType.Driving
        )
        if waypoint is None:
            self._is_off_road = True
            print('Non-drivable lane detected')
            return True
    
        # 5. Check if the vehicle's heading deviates too much from lane direction (> ±90°)
        ego_yaw = self.ego.get_transform().rotation.yaw
        lane_yaw = waypoint.transform.rotation.yaw
        yaw_diff = np.deg2rad(ego_yaw - lane_yaw)
        yaw_diff = np.arctan2(np.sin(yaw_diff), np.cos(yaw_diff))  # Normalize to [-π, π]
        if not waypoint.is_intersection:
            if abs(yaw_diff) > np.pi / 2:  # More than 90 degrees deviation (wrong-way driving)
                self._is_off_road = True
                print('Wrong-way driving detected')
                return True
    
        # 6. Deviation too far from lane center
        lane_width, lateral_offset = self._get_obs()['lane_info']
        if not waypoint.is_intersection:
            if lateral_offset > lane_width / 2 + 1.0:
                self._is_off_road = True
                print('Deviated from lane')
                return True
    
        return False

    def _clear_all_actors(self, actor_filters):
        """Clear (destroy) all actors matching the given filter patterns.

        Args:
            actor_filters (list): A list of filter strings, e.g., ['vehicle.*', 'walker.*', 'sensor.*'].
        """
        import sys
        import os

        # Collect all actors to destroy
        actors_to_destroy = []
        sensors_to_stop = []
        controllers_to_stop = []

        for actor_filter in actor_filters:
            for actor in self.world.get_actors().filter(actor_filter):
                if 'sensor' in actor.type_id:
                    sensors_to_stop.append(actor)
                elif 'controller' in actor.type_id:
                    controllers_to_stop.append(actor)
                actors_to_destroy.append(actor)

        # Stop sensors first
        for sensor in sensors_to_stop:
            try:
                sensor.stop()
            except:
                pass

        # Stop walker controllers
        for controller in controllers_to_stop:
            try:
                controller.stop()
            except:
                pass

        # Suppress CARLA's C++ error messages by redirecting stderr
        stderr_fd = sys.stderr.fileno()
        with open(os.devnull, 'w') as devnull:
            old_stderr = os.dup(stderr_fd)
            os.dup2(devnull.fileno(), stderr_fd)
            try:
                for actor in actors_to_destroy:
                    if actor.is_alive:
                        try:
                            actor.destroy()
                        except:
                            pass
            finally:
                os.dup2(old_stderr, stderr_fd)
                os.close(old_stderr)

    def clear_all_actors(self, actor_filters):
        """Clear (destroy) all actors matching the given filter patterns.

        Args:
            actor_filters (list): A list of filter strings, e.g., ['vehicle.*', 'walker.*', 'sensor.*'].
        """
        import sys
        import os

        # Collect all actors to destroy
        actors_to_destroy = []
        sensors_to_stop = []
        controllers_to_stop = []

        for actor_filter in actor_filters:
            for actor in self.world.get_actors().filter(actor_filter):
                if 'sensor' in actor.type_id:
                    sensors_to_stop.append(actor)
                elif 'controller' in actor.type_id:
                    controllers_to_stop.append(actor)
                actors_to_destroy.append(actor)

        # Stop sensors first
        for sensor in sensors_to_stop:
            try:
                sensor.stop()
            except:
                pass

        # Stop walker controllers before destroying
        for controller in controllers_to_stop:
            try:
                controller.stop()
            except:
                pass

        # Suppress CARLA's C++ error messages by redirecting stderr
        stderr_fd = sys.stderr.fileno()
        with open(os.devnull, 'w') as devnull:
            old_stderr = os.dup(stderr_fd)
            os.dup2(devnull.fileno(), stderr_fd)
            try:
                for actor in actors_to_destroy:
                    if actor.is_alive:
                        try:
                            actor.destroy()
                        except:
                            pass
            finally:
                os.dup2(old_stderr, stderr_fd)
                os.close(old_stderr)
