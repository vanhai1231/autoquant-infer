import sys
import os

# Thêm thư mục gốc của dự án vào sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.dvc_management import ai_agent_data_update
from scripts.mlflow_management import train_and_log_model, ai_agent_model_registry
from scripts.quantization import ai_agent_quantization_optimization

def run_ai_agent():
    """AI Agent chính điều phối toàn bộ quy trình."""
    print("AI Agent: Bắt đầu quản lý dữ liệu và mô hình...")
    
    # Bước 1: Cập nhật dữ liệu
    print("AI Agent: Kiểm tra và cập nhật dữ liệu...")
    ai_agent_data_update()
    
    # Bước 2: Huấn luyện và đăng ký mô hình
    print("AI Agent: Huấn luyện và đăng ký mô hình...")
    run_id, accuracy = train_and_log_model()
    ai_agent_model_registry(run_id, accuracy)
    
    # Bước 3: Tối ưu hóa quantization
    print("AI Agent: Tối ưu hóa quantization...")
    best_quant_type = ai_agent_quantization_optimization()
    
    print(f"AI Agent: Hoàn tất! Mức quantization tối ưu: {best_quant_type}")

if __name__ == "__main__":
    run_ai_agent()