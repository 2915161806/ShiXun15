# AlexNet CIFAR-10 图像分类项目  

## 遇到的问题及解决方案

### site-packages 文件的 DLL 文件版本问题（已解决）  


#### 解决方案：重新安装 Python 虚拟环境并手动下载  

##### 1. 创建并激活新的虚拟环境  
```bash  
# 使用 venv 创建虚拟环境（推荐 Python 3.8+）  
python -m venv new_env  
# Windows 激活命令  
new_env\Scripts\activate  
# Linux/Mac 激活命令  
source new_env/bin/activate  
```  

##### 2. 手动下载 PyTorch 及依赖（以 CPU 版本为例）  
- **步骤 1：访问 PyTorch 官网**  
  进入 [PyTorch 下载页面](https://pytorch.org/get-started/locally/)，根据系统配置选择合适的安装命令。例如：  
  ```bash  
  # CPU 版本（适用于无 GPU 或未安装 CUDA 的环境）  
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu  
  ```  
  - 若使用 CUDA，需确保下载的版本与显卡驱动、CUDA 版本匹配（如 `cu118` 对应 CUDA 11.8）。  

- **步骤 2：手动下载缺失的 DLL 文件（可选）**  
  若安装后仍提示 DLL 缺失，可从以下渠道手动获取：  
  - **PyTorch 官方安装包**：在 `site-packages/torch/bin/` 目录下找到 `cudart64_xx.dll`、`cublas64_xx.dll` 等文件（xx 为版本号）。  
  - **NVIDIA 开发者官网**：根据 CUDA 版本下载对应 DLL（如 [CUDA Toolkit 11.8](https://developer.nvidia.com/cuda-11-8-0-download-archive)）。  


#### 操作示例：完整流程  
```bash  
# 1. 删除旧环境（可选）  
rm -rf old_env  # Linux/Mac  
rd /s/q old_env  # Windows  

# 2. 创建新虚拟环境  
python -m venv alexnet_env  

# 3. 激活环境  
alexnet_env\Scripts\activate  # Windows  
source alexnet_env/bin/activate  # Linux/Mac  

# 4. 安装 PyTorch（CPU 版本）  
pip install torch torchvision tensorboard pillow numpy --index-url https://download.pytorch.org/whl/cpu  

# 5. 验证安装  
python -c "import torch; print(torch.__version__)"  # 输出版本号即安装成功  
```  


#### 注意事项  
- **虚拟环境隔离**：避免在系统 Python 环境中直接安装依赖，虚拟环境可防止不同项目的依赖冲突。  
- **DLL 路径配置**：若手动复制 DLL，建议将文件放入虚拟环境的 `site-packages/torch/bin/` 目录，而非系统目录（避免全局冲突）。  
- **版本一致性**：安装前确认 PyTorch 与 CUDA 的 [兼容性列表](https://pytorch.org/get-started/previous-versions/)，例如：  
  | PyTorch 版本 | 支持的 CUDA 版本 |  
  |--------------|------------------|  
  | 1.12.x       | 10.2, 11.3, 11.6 |  


