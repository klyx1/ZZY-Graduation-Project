import cv2
import numpy as np
import threading
import time
from queue import Queue
from sensors.gelsight_driver import GelSightCamera
from processing.tactile_processor import TactileProcessor
from models.grasp_controller import AGGripperController
from utils.visualization import TactileVisualizer

# --------------------- 初始化硬件 ---------------------
tactile_cam = GelSightCamera()
gripper = AGGripperController(port='/dev/ttyUSB0')
processor = TactileProcessor(E=0.3, nu=0.48)
visualizer = TactileVisualizer()

# --------------------- 多线程数据队列 ---------------------
tactile_queue = Queue(maxsize=10)  # 触觉数据队列
force_threshold = 2.0  # 安全力阈值（单位：N）


def tactile_thread():
    """ 触觉数据采集线程（100Hz） """
    while True:
        try:
            frame = tactile_cam.get_calibrated_frame()
            if tactile_queue.full():
                tactile_queue.get()  # 丢弃旧数据
            tactile_queue.put(frame)
        except Exception as e:
            print("触觉线程错误:", e)
        time.sleep(0.01)


# --------------------- 主处理循环 ---------------------
def main():
    threading.Thread(target=tactile_thread, daemon=True).start()

    while True:
        # 1. 获取最新触觉数据
        if tactile_queue.empty():
            continue
        tactile_img = tactile_queue.get()

        # 2. 计算力场
        gx, gy = processor.compute_gradients(tactile_img)
        depth_map = processor.poisson_reconstruct(gx, gy)
        force_map = processor.depth_to_force(depth_map)

        # 3. 安全力检测
        if np.max(force_map) > force_threshold:
            gripper.set_force(force_threshold)  # 力控模式
            print(f"警告：接触力超过阈值 {force_threshold}N")

        # 4. 可视化
        blended = visualizer.overlay_heatmap(tactile_img, force_map)
        cv2.imshow("Tactile Force", blended)
        visualizer.update_3d_mesh(depth_map)

        if cv2.waitKey(1) == 27:  # ESC退出
            break

    tactile_cam.close()
    gripper.client.close()


if __name__ == "__main__":
    main()