import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pandas as pd

def train_and_log_model(data_path="data/raw/sample_data.csv", model_name="logistic_model"):
    """Huấn luyện và lưu mô hình vào MLflow."""
    data = pd.read_csv(data_path)
    X = data[["feature"]]
    y = data["label"]
    
    model = LogisticRegression()
    model.fit(X, y)
    predictions = model.predict(X)
    accuracy = accuracy_score(y, predictions)
    
    with mlflow.start_run():
        mlflow.sklearn.log_model(model, "model")
        mlflow.log_metric("accuracy", accuracy)
        run_id = mlflow.active_run().info.run_id
        print(f"Đã lưu mô hình với run_id: {run_id}, Độ chính xác: {accuracy}")
    return run_id, accuracy

def ai_agent_model_registry(run_id, accuracy, model_name="logistic_model"):
    """AI Agent so sánh và đăng ký mô hình tốt nhất vào MLflow Registry."""
    client = mlflow.tracking.MlflowClient()
    try:
        client.create_registered_model(model_name)
    except:
        pass
    
    best_accuracy = 0.0
    try:
        versions = client.get_latest_versions(model_name)
        if versions:
            best_accuracy = client.get_run(versions[0].run_id).data.metrics.get("accuracy", 0.0)
    except:
        pass
    
    if accuracy > best_accuracy:
        model_uri = f"runs:/{run_id}/model"
        client.create_model_version(model_name, model_uri, run_id)
        print(f"AI Agent: Đã đăng ký phiên bản mô hình mới với độ chính xác {accuracy}")
    else:
        print(f"AI Agent: Mô hình hiện tại (độ chính xác {accuracy}) không tốt hơn phiên bản tốt nhất (độ chính xác {best_accuracy})")

if __name__ == "__main__":
    run_id, accuracy = train_and_log_model()
    ai_agent_model_registry(run_id, accuracy)
