import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

# Định nghĩa lại kiến trúc ResNet
def residual_block(x, filters, kernel_size=3):
    y = layers.Conv2D(filters, kernel_size, padding='same', activation='relu',
                      kernel_regularizer=tf.keras.regularizers.l2(0.002))(x)
    y = layers.Conv2D(filters, kernel_size, padding='same',
                      kernel_regularizer=tf.keras.regularizers.l2(0.002))(y)
    shortcut = layers.Conv2D(filters, 1, padding='same')(x) if x.shape[-1] != filters else x
    y = layers.Add()([shortcut, y])
    y = layers.Activation('relu')(y)
    return y

def build_resnet(input_shape=(128, 32, 1), num_classes=11):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(32, (3, 3), padding='same', activation='relu',
                      kernel_regularizer=tf.keras.regularizers.l2(0.002))(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    
    x = residual_block(x, 32)
    x = residual_block(x, 64)
    x = layers.MaxPooling2D((2, 2))(x)
    x = residual_block(x, 128)
    
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu',
                     kernel_regularizer=tf.keras.regularizers.l2(0.002))(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    return models.Model(inputs, outputs)

# Hàm tải mô hình với xử lý lỗi batch_shape
def load_model_with_fallback(model_path):
    try:
        model = tf.keras.models.load_model(model_path, compile=False)
        print("Đã tải mô hình thành công từ file!")
    except Exception as e:
        print(f"Không thể tải trực tiếp mô hình: {e}")
        print("Đang thử định nghĩa lại mô hình và tải trọng số...")
        model = build_resnet(input_shape=(128, 32, 1), num_classes=11)
        try:
            model.load_weights(model_path)
            print("Đã tải trọng số thành công!")
        except Exception as weight_error:
            raise ValueError(f"Không thể tải trọng số: {weight_error}")
    return model

# Hàm phân tích các lớp và ma trận trọng số
def analyze_layers_and_weights(model):
    print("\n=== Phân tích Các Lớp và Ma Trận Trọng Số ===")
    
    # Tổng quan mô hình
    total_params = model.count_params()
    print(f"Tổng số tham số: {total_params:,}")
    print(f"Số lớp: {len(model.layers)}")
    
    # Duyệt qua từng lớp
    print("\nDanh sách các lớp:")
    for i, layer in enumerate(model.layers):
        layer_name = layer.name
        layer_type = layer.__class__.__name__
        output_shape = layer.output_shape
        
        print(f"\nLớp {i}: {layer_name} ({layer_type})")
        print(f"  Kích thước đầu ra: {output_shape}")
        
        # Kiểm tra và in trọng số (nếu có)
        if layer.trainable_weights:
            weights = layer.get_weights()
            print(f"  Số lượng ma trận trọng số: {len(weights)}")
            for j, w in enumerate(weights):
                print(f"    Ma trận {j}:")
                print(f"      Kích thước: {w.shape}")
                # In một phần nhỏ của ma trận (giới hạn để tránh output quá dài)
                if w.size > 0:
                    sample_size = min(5, w.size)  # In tối đa 5 giá trị
                    flat_weights = w.flatten()[:sample_size]
                    print(f"      Giá trị mẫu: {flat_weights}")
                else:
                    print("      (Không có giá trị)")
        else:
            print("  Không có trọng số (non-trainable hoặc không áp dụng)")

# Hàm chính
def main():
    print("TensorFlow Version:", tf.__version__)
    
    # Đường dẫn đến file mô hình
    model_path = r'D:\Python\AI-Instrument-Classifier\models\resnet_mel_instrument_classifier_89.h5'
    
    # Tải mô hình
    try:
        model = load_model_with_fallback(model_path)
        analyze_layers_and_weights(model)  # Phân tích các lớp và trọng số
    except Exception as e:
        print(f"Không thể tải mô hình: {e}")

if __name__ == "__main__":
    main()