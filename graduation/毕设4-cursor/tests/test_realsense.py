import pyrealsense2 as rs
import numpy as np
import cv2
import time

def test_realsense():
    # 创建上下文
    ctx = rs.context()
    
    # 获取所有设备
    devices = ctx.query_devices()
    print(f"Found {len(devices)} device(s)")
    
    # 遍历所有设备
    for dev in devices:
        print("Device info:")
        print("  Name:", dev.get_info(rs.camera_info.name))
        print("  Serial number:", dev.get_info(rs.camera_info.serial_number))
        print("  USB type:", dev.get_info(rs.camera_info.usb_type_descriptor))
        print("  Firmware version:", dev.get_info(rs.camera_info.firmware_version))
        print("  Product ID:", dev.get_info(rs.camera_info.product_id))
        print("  Product line:", dev.get_info(rs.camera_info.product_line))
        
        # 获取传感器
        depth_sensor = dev.first_depth_sensor()
        if depth_sensor:
            print("\nDepth sensor found:")
            # 获取支持的分辨率
            print("  Supported depth modes:")
            for sensor in dev.query_sensors():
                if sensor.is_depth_sensor():
                    for profile in sensor.get_stream_profiles():
                        if profile.stream_type() == rs.stream.depth:
                            video_profile = profile.as_video_stream_profile()
                            print(f"    {video_profile.width()}x{video_profile.height()} @ {video_profile.fps()}fps")
                
        color_sensor = dev.first_color_sensor()
        if color_sensor:
            print("\nColor sensor found:")
            # 获取支持的分辨率
            print("  Supported color modes:")
            for sensor in dev.query_sensors():
                if sensor.is_color_sensor():
                    for profile in sensor.get_stream_profiles():
                        if profile.stream_type() == rs.stream.color:
                            video_profile = profile.as_video_stream_profile()
                            print(f"    {video_profile.width()}x{video_profile.height()} @ {video_profile.fps()}fps")
    
    # 尝试获取图像
    print("\nTrying to get frames...")
    pipeline = rs.pipeline()
    config = rs.config()
    
    # 配置流
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    
    try:
        # 启动管道
        pipeline_profile = pipeline.start(config)
        print("Pipeline started successfully")
        
        # 获取深度传感器
        depth_sensor = pipeline_profile.get_device().first_depth_sensor()
        
        # 设置深度单位为毫米
        depth_sensor.set_option(rs.option.depth_units, 0.001)
        
        # 创建对齐对象
        align = rs.align(rs.stream.color)
        
        # 等待相机预热
        print("Waiting for camera to warm up...")
        time.sleep(3)
        
        # 获取30帧数据进行测试
        for i in range(30):
            # 等待一帧数据
            frames = pipeline.wait_for_frames()
            
            # 对齐深度帧到彩色帧
            aligned_frames = align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            
            if depth_frame and color_frame:
                # 转换为numpy数组
                depth_image = np.asanyarray(depth_frame.get_data())
                color_image = np.asanyarray(color_frame.get_data())
                
                # 创建深度图的可视化
                depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
                
                # 显示图像
                cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
                images = np.hstack((color_image, depth_colormap))
                cv2.imshow('RealSense', images)
                
                # 每5帧保存一次图像
                if i % 5 == 0:
                    cv2.imwrite(f"depth_{i}.png", depth_colormap)
                    cv2.imwrite(f"color_{i}.png", color_image)
                    print(f"Frame {i} saved")
                
                # 等待按键
                key = cv2.waitKey(1)
                if key == 27:  # ESC键退出
                    break
                    
            else:
                print("Failed to get valid frames")
                
    except Exception as e:
        print(f"Error: {str(e)}")
        
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print("Pipeline stopped")

if __name__ == "__main__":
    test_realsense()