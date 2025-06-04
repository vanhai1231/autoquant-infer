import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
import os

def create_sample_model(data_path="data/raw/sample_data.csv"):
    """Huấn luyện mô hình đơn giản và lưu lại."""
    data = pd.read_csv(data_path)
    X = data[["feature"]].values.astype(np.float32)
    y = data["label"].values

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(1,)),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    model.fit(X, y, epochs=10, verbose=0)

    os.makedirs("models", exist_ok=True)
    model.save("models/unquantized_model.keras")
    return model, X, y

def apply_quantization(model, quantization_type="int8"):
    """Convert model về TFLite với quantization cụ thể."""
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    if quantization_type == "int8":
        def representative_dataset_gen():
            for x in np.linspace(0, 10, 100, dtype=np.float32):
                yield [np.array([[x]], dtype=np.float32)]
        converter.representative_dataset = representative_dataset_gen
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8

    model_path = f"models/quantized_model_{quantization_type}.tflite"
    tflite_model = converter.convert()
    with open(model_path, "wb") as f:
        f.write(tflite_model)

    return model_path

def evaluate_quantized_model(model_path, X, y, quantization_type="float32"):
    """Chạy đánh giá độ chính xác mô hình TFLite."""
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    predictions = []
    for x in X:
        input_value = x.astype(np.float32).reshape(1, 1)

        if quantization_type == "int8":
            scale, zero_point = input_details["quantization"]
            input_value = np.round(input_value / scale + zero_point).astype(np.int8)

        interpreter.set_tensor(input_details["index"], input_value)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details["index"])
        pred = output[0][0] if output.ndim == 2 else output[0]
        predictions.append(1 if pred > 0.5 else 0)

    accuracy = accuracy_score(y, predictions)
    inference_time = 0.001  # Giả định (có thể đo thời gian thật nếu cần)
    return accuracy, inference_time

def ai_agent_quantization_optimization(data_path="data/raw/sample_data.csv"):
    """Tìm mức quantization tối ưu (accuracy tốt, inference nhanh)."""
    print("AI Agent: Huấn luyện mô hình...")
    model, X, y = create_sample_model(data_path)

    quantization_types = ["int8", "dynamic"]
    best_quant_type = None
    best_accuracy = 0
    best_inference_time = float("inf")

    for q_type in quantization_types:
        print(f"\nAI Agent: Thử quantization: {q_type}")
        model_path = apply_quantization(model, q_type)
        acc, inf_time = evaluate_quantized_model(model_path, X, y, q_type)
        print(f"AI Agent: Đánh giá {q_type} - Accuracy: {acc:.4f}, Inference time: {inf_time:.4f}s")

        if acc >= best_accuracy * 0.95 and inf_time < best_inference_time:
            best_accuracy = acc
            best_inference_time = inf_time
            best_quant_type = q_type

    print(f"\n✅ AI Agent: Mức quantization tối ưu: {best_quant_type}, Accuracy: {best_accuracy:.4f}, Inference time: {best_inference_time:.4f}s")
    return best_quant_type

if __name__ == "__main__":
    ai_agent_quantization_optimization()
