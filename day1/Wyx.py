import rasterio
import numpy as np

def shuchu(tif_file):
    """
    从 GeoTIFF 文件中读取波段数据，并生成归一化的真彩色图像（RGB）。

    参数:
        tif_file (str): GeoTIFF 文件的路径。

    返回:
        rgb_normalized (numpy.ndarray): 归一化后的 RGB 图像，数据类型为 uint8。
    """
    # 打开 GeoTIFF 文件
    with rasterio.open(tif_file) as src:
        # 读取所有波段数据
        bands = src.read()  # 形状为 (波段数, 高度, 宽度)
        profile = src.profile  # 获取元数据

    # 检查波段数量是否满足要求
    if bands.shape[0] < 3:
        raise ValueError("TIFF 文件中波段数量不足，无法生成 RGB 图像。")

    # 分配波段（假设波段顺序为 B02, B03, B04）
    blue = bands[0].astype(float)   # B02 - 蓝
    green = bands[1].astype(float)  # B03 - 绿
    red = bands[2].astype(float)    # B04 - 红

    # 真彩色正则化
    rgb_orign = np.dstack((red, green, blue))
    array_min, array_max = rgb_orign.min(), rgb_orign.max()
    rgb_normalized = ((rgb_orign - array_min) / (array_max - array_min)) * 255
    rgb_normalized = rgb_normalized.astype(np.uint8)

    # 转换为 (波段数, 高度, 宽度) 的格式
    rgb_normalized = np.moveaxis(rgb_normalized, -1, 0)

    return rgb_normalized, profile

# 调用函数
tif_file_path = "E:\ShiXunData\EXAMPLE.tif"  # 替换为你的 GeoTIFF 文件路径
normalized_rgb, profile = shuchu(tif_file_path)

# 修改元数据以保存 RGB 图像
profile.update(
    driver='GTiff',
    dtype=rasterio.uint8,
    count=3,  # RGB 图像有 3 个波段
    compress='lzw'
)

# 保存为 GeoTIFF 文件
output_image_path = "E:\ShiXunData"  # 输出图像的路径
with rasterio.open(output_image_path, 'w', **profile) as dst:
    dst.write(normalized_rgb)

# 打印结果
print(f"归一化的 RGB 图像已保存到 {output_image_path}")
