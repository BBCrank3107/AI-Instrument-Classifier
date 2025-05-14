import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import pandas as pd

# Định nghĩa residual_block
def residual_block(x, filters, kernel_size=3):
    y = layers.Conv2D(filters, kernel_size, padding='same', activation='relu',
                      kernel_regularizer=tf.keras.regularizers.l2(0.002))(x)
    y = layers.Conv2D(filters, kernel_size, padding='same',
                      kernel_regularizer=tf.keras.regularizers.l2(0.002))(y)
    shortcut = layers.Conv2D(filters, 1, padding='same')(x) if x.shape[-1] != filters else x
    y = layers.Add()([shortcut, y])
    y = layers.Activation('relu')(y)
    return y

# Định nghĩa build_resnet với kernel_size và filters linh hoạt
def build_resnet(input_shape=(128, 32, 1), num_classes=11, kernel_size=3, filters=[32, 64, 128]):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(filters[0], (kernel_size, kernel_size), padding='same', activation='relu',
                      kernel_regularizer=tf.keras.regularizers.l2(0.002))(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    
    x = residual_block(x, filters[0], kernel_size=kernel_size)
    x = residual_block(x, filters[1], kernel_size=kernel_size)
    x = layers.MaxPooling2D((2, 2))(x)
    x = residual_block(x, filters[2], kernel_size=kernel_size)
    
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu',
                     kernel_regularizer=tf.keras.regularizers.l2(0.002))(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    return models.Model(inputs, outputs)

# Hàm tải mô hình
def load_model_with_fallback(model_path, kernel_size, filters):
    try:
        model = build_resnet(input_shape=(128, 32, 1), num_classes=11, kernel_size=kernel_size, filters=filters)
        model.load_weights(model_path)
        print(f"Đã tải mô hình thành công từ file {model_path}!")
    except Exception as e:
        raise ValueError(f"Không thể tải mô hình từ {model_path}: {e}")
    return model

# Hàm phân tích các lớp và ma trận trọng số
def analyze_layers_and_weights(model, model_name):
    print(f"\n=== Phân tích Các Lớp và Ma Trận Trọng Số của {model_name} ===")
    
    # Tổng quan mô hình
    total_params = model.count_params()
    print(f"Tổng số tham số: {total_params:,}")
    print(f"Số lớp: {len(model.layers)}")
    
    # Lưu thông tin để so sánh
    layer_info = []
    for i, layer in enumerate(model.layers):
        layer_name = layer.name
        layer_type = layer.__class__.__name__
        # Sử dụng layer.output.shape thay vì layer.output_shape
        output_shape = layer.output.shape
        
        layer_details = {
            'Model': model_name,
            'Layer Index': i,
            'Layer Name': layer_name,
            'Layer Type': layer_type,
            'Output Shape': output_shape,
            'Weights': []
        }
        
        # Kiểm tra và lưu thông tin trọng số
        if layer.trainable_weights:
            weights = layer.get_weights()
            layer_details['Num Weights'] = len(weights)
            for j, w in enumerate(weights):
                weight_info = {
                    'Matrix Index': j,
                    'Shape': w.shape,
                    'Sample Values': w.flatten()[:5] if w.size > 0 else "N/A"
                }
                layer_details['Weights'].append(weight_info)
        else:
            layer_details['Num Weights'] = 0
        
        layer_info.append(layer_details)
    
    return layer_info, total_params

# Hàm so sánh các mô hình
def compare_models(layer_infos, total_params_list, models_info):
    print("\n=== So sánh Các Mô Hình ===")
    
    # Kiểm tra độ dài của total_params_list
    if len(total_params_list) != len(models_info):
        print(f"Warning: Số lượng tham số ({len(total_params_list)}) không khớp với số mô hình ({len(models_info)}).")
        return
    
    # So sánh tổng số tham số
    params_df = pd.DataFrame({
        'Model': [info['name'] for info in models_info],
        'Total Parameters': total_params_list
    })
    print("\nSo sánh tổng số tham số:")
    print(params_df.to_string(index=False))
    
    # So sánh kiến trúc
    print("\nSo sánh kiến trúc (các lớp):")
    for i in range(len(layer_infos[0])):  # Duyệt qua từng lớp
        print(f"\nLớp {i}:")
        for j, (model_info, layers) in enumerate(zip(models_info, layer_infos)):
            layer = layers[i]
            print(f"  {model_info['name']}:")
            print(f"    Layer Name: {layer['Layer Name']}, Type: {layer['Layer Type']}")
            print(f"    Output Shape: {layer['Output Shape']}")
            print(f"    Number of Weight Matrices: {layer['Num Weights']}")
            for weight in layer['Weights']:
                print(f"      Matrix {weight['Matrix Index']}: Shape {weight['Shape']}, Sample Values: {weight['Sample Values']}")
    
    # So sánh trọng số (ví dụ: lớp Conv2D đầu tiên)
    print("\nSo sánh trọng số của lớp Conv2D đầu tiên:")
    for j, (model_info, layers) in enumerate(zip(models_info, layer_infos)):
        conv_layer = next(layer for layer in layers if 'conv2d' in layer['Layer Name'].lower())
        print(f"\n{model_info['name']}:")
        for weight in conv_layer['Weights']:
            print(f"  Matrix {weight['Matrix Index']}: Shape {weight['Shape']}, Sample Values: {weight['Sample Values']}")

# Thông tin 3 mô hình
models_info = [
    {
        'name': 'resnet_mel_instrument_classifier_1x1.h5',
        'kernel_size': 1,
        'filters': [16, 32, 64],
        'path': r'D:\Python\AI-Instrument-Classifier\models\resnet_mel_instrument_classifier_1x1.h5'
    },
    {
        'name': 'resnet_mel_instrument_classifier_3x3.h5',
        'kernel_size': 3,
        'filters': [32, 64, 128],
        'path': r'D:\Python\AI-Instrument-Classifier\models\resnet_mel_instrument_classifier_3x3.h5'
    },
    {
        'name': 'resnet_mel_instrument_classifier_5x5.h5',
        'kernel_size': 5,
        'filters': [64, 128, 256],
        'path': r'D:\Python\AI-Instrument-Classifier\models\resnet_mel_instrument_classifier_5x5.h5'
    }
]

# Hàm chính
def main():
    print("TensorFlow Version:", tf.__version__)
    
    layer_infos = []
    total_params_list = []
    
    # Phân tích từng mô hình
    for model_info in models_info:
        try:
            model = load_model_with_fallback(
                model_info['path'],
                model_info['kernel_size'],
                model_info['filters']
            )
            layer_info, total_params = analyze_layers_and_weights(model, model_info['name'])
            layer_infos.append(layer_info)
            total_params_list.append(total_params)
        except Exception as e:
            print(f"Không thể phân tích mô hình {model_info['name']}: {e}")
            continue
    
    # So sánh các mô hình
    compare_models(layer_infos, total_params_list, models_info)

if __name__ == "__main__":
    main()