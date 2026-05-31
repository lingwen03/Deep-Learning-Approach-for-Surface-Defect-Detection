# 本地复现说明

## 当前工作目录

项目目录已经更新为：

```text
D:\python\Deep-Learning-Approach-for-Surface-Defect-Detection
```

本目录保留了作者提供的 `checkpoint/ckp-486` 权重、`Log/` 日志和 `visualization/test/` 可视化结果。

## 已做兼容修复

- 将字符串比较从 `is` 改为 `==`。
- 将 TensorFlow placeholder 中的 `/8` 改为 `//8`，避免 Python 3 下 shape 变成浮点数。
- 删除未使用且新版 SciPy 已移除的 `scipy.misc.imread/imresize/imsave` 导入。
- 保存 checkpoint 前自动创建 `checkpoint/` 目录。

## 推荐环境

原作者环境：

```text
Python 3.6
CUDA 9.0
cuDNN 7.1.4
TensorFlow 1.12
```

当前机器命令行检测到的是 Python 3.14，不能安装或运行 TensorFlow 1.12。建议使用 Python 3.6 隔离环境、WSL/conda，或 Docker。

## 数据集

下载 KolektorSDD 后，建议解压到：

```text
D:\python\Deep-Learning-Approach-for-Surface-Defect-Detection\data\KolektorSDD
```

数据目录内应包含 `kos01` 到 `kos50` 这类子目录，每个子目录包含原图和对应的 `_label.bmp` 文件。

## Python 3.6 运行

在 Python 3.6 环境中执行：

```powershell
cd D:\python\Deep-Learning-Approach-for-Surface-Defect-Detection
python -m pip install -r requirements-py36.txt
python run.py --test -dd data\KolektorSDD
```

训练分三步：

```powershell
python run.py --train_segment -dd data\KolektorSDD
python run.py --train_decision -dd data\KolektorSDD
python run.py --train_total -dd data\KolektorSDD
```

## Docker 运行

Docker Desktop 启动后，在本目录执行：

```powershell
docker build -t surface-defect-tf112 .
docker run --rm -v "${PWD}\data\KolektorSDD:/workspace/data/KolektorSDD" surface-defect-tf112
```

如果需要保存新的输出结果，可以再挂载日志和可视化目录。

## 输出位置

- 测试可视化：`visualization/test/`
- 日志：`Log/*.txt`
- 模型权重：`checkpoint/`
