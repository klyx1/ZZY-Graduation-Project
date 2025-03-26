import unittest
import numpy as np
import cv2
import os
import sys
import time
import pyrealsense2 as rs
import open3d as o3d
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.sensors.vision.camera import RealSenseCamera
from src.sensors.vision.processor import VisionProcessor

class TestVisionSensor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """测试类初始化"""
        cls.config_path = "config/config.yaml"
        cls.camera = RealSenseCamera(cls.config_path)
        cls.processor = VisionProcessor(cls.config_path)
        # 确保在setUpClass中就连接相机
        try:
            cls.camera.connect()
            # 等待相机预热
            print("等待相机预热...")
            time.sleep(3)
        except Exception as e:
            print(f"相机连接失败: {e}")
            raise
        
    @classmethod
    def tearDownClass(cls):
        """测试类清理"""
        cls.camera.disconnect()
        
    def test_camera_initialization(self):
        """测试相机初始化"""
        self.assertIsNotNone(self.camera.config)
        self.assertIsNotNone(self.camera.pipeline)  # 现在pipeline在setUpClass中已经初始化
        self.assertEqual(self.camera.width, 640)  # 检查分辨率
        self.assertEqual(self.camera.height, 480)
        
    def test_camera_connection(self):
        """测试相机连接"""
        try:
            self.assertTrue(self.camera.is_connected)
            self.assertIsNotNone(self.camera.pipeline)
            
            # 测试深度单位设置
            depth_sensor = self.camera.pipeline.get_active_profile().get_device().first_depth_sensor()
            depth_scale = depth_sensor.get_option(rs.option.depth_units)
            self.assertAlmostEqual(depth_scale, 0.001, places=4)  # 检查深度单位是否为1mm
            
        except Exception as e:
            self.fail(f"相机连接测试失败: {e}")
        
    def test_get_frames(self):
        """测试获取图像帧"""
        try:
            # 连续获取多帧以确保稳定性
            for _ in range(5):
                depth_image, color_image = self.camera.get_frames()
                
                self.assertIsNotNone(depth_image)
                self.assertIsNotNone(color_image)
                self.assertEqual(depth_image.shape, (480, 640))  # 检查深度图分辨率
                self.assertEqual(color_image.shape, (480, 640, 3))  # 检查彩色图分辨率
                
                # 检查深度范围
                depth_scale = 0.001
                depth_meters = depth_image * depth_scale
                self.assertTrue(np.all(depth_meters[depth_meters > 0] <= 4.0))  # 最大深度4米
                
                time.sleep(0.1)  # 等待下一帧
                
        except Exception as e:
            self.fail(f"获取图像帧测试失败: {e}")
        
    def test_get_pointcloud(self):
        """测试获取点云数据"""
        try:
            # 获取深度图和彩色图
            depth_frame = self.camera.pipeline.wait_for_frames().get_depth_frame()
            color_frame = self.camera.pipeline.wait_for_frames().get_color_frame()
            
            # 获取点云数据
            pointcloud = self.camera.get_pointcloud(depth_frame)
            
            self.assertIsNotNone(pointcloud)
            self.assertEqual(pointcloud.shape[1], 3)  # XYZ坐标
            
            # 检查点云范围
            self.assertTrue(np.all(pointcloud[:, 2] <= 4.0))  # Z轴最大4米
            
            # 创建点云对象
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pointcloud)
            
            # 计算点云法向量
            pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=50))
            
            # 添加颜色信息
            color_image = np.asanyarray(color_frame.get_data())
            color_image = cv2.resize(color_image, (640, 480))
            colors = color_image.reshape(-1,3) / 255.0
            
            
            pcd.colors = o3d.utility.Vector3dVector(colors)
            
            # 创建可视化窗口
            vis = o3d.visualization.Visualizer()
            vis.create_window(width=1280, height=720)
            
            # 设置渲染选项
            opt = vis.get_render_option()
            opt.background_color = np.asarray([0.1, 0.1, 0.1])  # 深灰色背景
            opt.point_size = 1.5  # 减小点大小以获得更好的细节
            opt.light_on = True  # 启用光照
            opt.mesh_show_back_face = True
            opt.mesh_show_wireframe = False
            
            # 添加点云到可视化器
            vis.add_geometry(pcd)
            
            # 设置视角
            ctr = vis.get_view_control()
            ctr.set_zoom(0.7)  # 稍微放大一点
            ctr.set_front([0, 0, 1])  # 正Z轴方向
            ctr.set_lookat([0, 0, 0.3])  # 降低观察点以更好地看到物体
            ctr.set_up([0, -1, 0])  # 负Y轴方向
            
            # 渲染并保存图像
            vis.poll_events()
            vis.update_renderer()
            image = vis.capture_screen_float_buffer(do_render=True)
            image = np.asarray(image)
            
            # 保存图像
            output_dir = "data/pointcloud_images"
            os.makedirs(output_dir, exist_ok=True)
            cv2.imwrite(os.path.join(output_dir, "pointcloud.png"), (image * 255).astype(np.uint8))
            
            # 关闭可视化窗口
            vis.destroy_window()
            
        except Exception as e:
            self.fail(f"获取点云测试失败: {e}")
        
    def test_vision_processor(self):
        """测试视觉处理器"""
        try:
            depth_image, color_image = self.camera.get_frames()
            
            # 测试深度图处理
            processed_depth = self.processor.process_depth(depth_image)
            self.assertIsNotNone(processed_depth)
            self.assertEqual(processed_depth.shape, (480, 640))
            
            # 测试彩色图处理
            processed_color = self.processor.process_color(color_image)
            self.assertIsNotNone(processed_color)
            self.assertEqual(processed_color.shape, (720, 1280, 3))  # 处理后放大到1280x720
            
            # 测试图像分割
            segmentation = self.processor.segment_object(color_image, depth_image)
            self.assertIsNotNone(segmentation)
            self.assertIn('mask', segmentation)
            self.assertIn('segmented', segmentation)
            self.assertIn('bbox', segmentation)
            
        except Exception as e:
            self.fail(f"视觉处理器测试失败: {e}")
        
    def test_camera_intrinsics(self):
        """测试相机内参"""
        try:
            intrinsics = self.camera.get_intrinsics()
            
            self.assertIsNotNone(intrinsics)
            self.assertIn('depth', intrinsics)
            self.assertIn('color', intrinsics)
            
            for stream in ['depth', 'color']:
                self.assertEqual(intrinsics[stream]['width'], 640)
                self.assertEqual(intrinsics[stream]['height'], 480)
                self.assertIn('ppx', intrinsics[stream])
                self.assertIn('ppy', intrinsics[stream])
                self.assertIn('fx', intrinsics[stream])
                self.assertIn('fy', intrinsics[stream])
                self.assertIn('model', intrinsics[stream])
                self.assertIn('coeffs', intrinsics[stream])
                
        except Exception as e:
            self.fail(f"相机内参测试失败: {e}")
        
    def test_camera_extrinsics(self):
        """测试相机外参"""
        try:
            extrinsics = self.camera.get_extrinsics()
            
            self.assertIsNotNone(extrinsics)
            self.assertIn('rotation', extrinsics)
            self.assertIn('translation', extrinsics)
            
            # 检查旋转矩阵
            rotation = np.array(extrinsics['rotation']).reshape(3, 3)
            self.assertTrue(np.allclose(np.dot(rotation, rotation.T), np.eye(3), atol=1e-6))
            
            # 检查平移向量
            translation = np.array(extrinsics['translation'])
            self.assertEqual(translation.shape, (3,))
            
        except Exception as e:
            self.fail(f"相机外参测试失败: {e}")

if __name__ == '__main__':
    unittest.main() 