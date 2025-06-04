
import dvc.api
import os
import pandas as pd
from datetime import datetime

def init_dvc(data_path="data/raw/sample_data.csv"):
    """Khởi tạo DVC và thêm dữ liệu mẫu."""
    if not os.path.exists(data_path):
        os.makedirs("data/raw", exist_ok=True)
        sample_data = pd.DataFrame({
            "feature": [1, 2, 3, 4, 5],
            "label": [0, 1, 0, 1, 0]
        })
        sample_data.to_csv(data_path, index=False)
    
    os.system(f"dvc add {data_path}")
    os.system(f"git add {data_path}.dvc data/.gitignore")
    os.system(f"git commit -m 'Thêm dữ liệu mẫu vào DVC'")

def ai_agent_data_update(data_path="data/raw/sample_data.csv"):
    """AI Agent kiểm tra và cập nhật dữ liệu mới vào DVC."""
    if os.path.exists(data_path):
        current_data = pd.read_csv(data_path)
        # Mô phỏng dữ liệu mới (thêm dòng)
        new_data = pd.DataFrame({
            "feature": [6, 7, 8],
            "label": [1, 0, 1]
        })
        updated_data = pd.concat([current_data, new_data], ignore_index=True)
        updated_data.to_csv(data_path, index=False)
        
        # Cập nhật DVC
        os.system(f"dvc add {data_path}")
        os.system(f"git add {data_path}.dvc")
        os.system(f"git commit -m 'AI Agent: Cập nhật dữ liệu {datetime.now()}'")
        print(f"AI Agent: Đã cập nhật {data_path} vào DVC")
    else:
        print("AI Agent: Không phát hiện dữ liệu mới")

if __name__ == "__main__":
    init_dvc()
    ai_agent_data_update()
