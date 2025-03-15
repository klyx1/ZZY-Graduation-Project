'''# RealSense相机内参
realsense:
  depth:
    fx: 610.0   # 焦距x
    fy: 610.0   # 焦距y
    cx: 420.0   # 主点x
    cy: 240.0   # 主点y
  color:
    fx: 900.0
    fy: 900.0
    cx: 640.0
    cy: 360.0

# GelSight标定参数
gelsight:
  dark_frame: "calib/dark_frame.npy"  # 暗场校准文件路径
  mmpp: 0.1  # 毫米/像素'''