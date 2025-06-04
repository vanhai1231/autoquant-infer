import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

def create_sample_model(data_path="data/raw/sample_data.csv"):
    """Tạo mô hình TensorFlow đơn giản."""
    data = pd.read_csv(data_path)
    X = data[["feature"]].values
    y = data["label"].values
    
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(16, activation="relu", input_shape=(1,)),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    model.fit(X, y, epochs=10, verbose=0)
    model.save("models/unquantized_model")
    return model, X, y

def apply_quantization(model, quantization_type="int8"):
    """Áp dụng Post-Training Quantization."""
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    if quantization_type == "int8":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.int8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        converter.representative_dataset = lambda: [np.array([[x]], dtype=np.float32) for x in np.linspace(0, 10, 100)]
    elif quantization_type == "dynamic":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    tflite_model = converter.convert()
    model_path = f"models/quantized_model_{quantization_type}.tflite"
    with open(model_path, "wb") as f:
        f.write(tflite_model)
    return model_path

def evaluate_quantized_model(model_path, X, y):
    """Đánh giá mô hình quantized."""
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]
    
    predictions = []
    for x in X:
        interpreter.set_tensor(input_index, x.astype(np.float32).reshape(1, 1))
        interpreter.invoke()
        pred = interpreter.get_tensor(output_index)[0][0]
        predictions.append(1 if pred > 0.5 else 0)
    
    accuracy = accuracy_score(y, predictions)
    inference_time = 0.001  # Giả lập thời gian inference (thực tế cần đo)
    return accuracy, inference_time

def ai_agent_quantization_optimization(data_path="data/raw/sample_data.csv"):
    """AI Agent chọn mức quantization tối ưu."""
    model, X, y = create_sample_model(data_path)
    
    quantization_types = ["int8", "dynamic"]
    best_accuracy = 0.0
    best_inference_time = float("inf")
    best_quant_type = None
    
    for q_type in quantization_types:
        model_path = apply_quantization(model, q_type)
        accuracy, inference_time = evaluate_quantized_model(model_path, X, y)
        print(f"AI Agent: Đánh giá {q_type} - Độ chính xác: {accuracy}, Thời gian inference: {inference_time}")
        
        # Tiêu chí: tối ưu inference time, độ chính xác không giảm quá 5%
        if accuracy >= best_accuracy * 0.95 and inference_time < best_inference_time:
            best_accuracy = accuracy
            best_inference_time = inference_time
            best_quant_type = q_type
    
    print(f"AI Agent: Mức quantization tối ưu: {best_quant_type}, Độ chính xác: {best_accuracy}, Thời gian inference: {best_inference_time}")
    return best_quant_type

if __name__ == "__main__":
    best_quant_type = ai_agent_quantization_optimization()
