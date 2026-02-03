import os
import pandas as pd

# 获取环境变量
data_dir = os.getenv("AMLT_BLOB_ROOT_DIR")

# 如果环境变量为空，使用默认路径
if not data_dir:
    print("环境变量 'AMLT_BLOB_ROOT_DIR' 未设置，使用默认路径。")
    data_dir = "/data/lewwang/shared/data"  # 你可以根据需要修改默认路径

# 构建文件路径
file_path = os.path.join(data_dir, "lewwang/test_data.csv")

# 读取并打印文件内容
try:
    df = pd.read_csv(file_path)
    print("读取的CSV内容如下：")
    print(df)
except FileNotFoundError:
    print(f"文件未找到: {file_path}")
except Exception as e:
    print(f"读取文件时出错: {e}")

