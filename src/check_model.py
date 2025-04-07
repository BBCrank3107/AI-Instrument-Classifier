import tensorflow as tf
from tensorflow.keras import layers, models

# Định nghĩa lại kiến trúc ResNet
def residual_block(x, filters, kernel_size=3):
    y = layers.Conv2D(filters, kernel_size, padding='same', activation='relu',
                      kernel_regularizer=tf.keras.regularizers.l2(0.002))(x)
    y = layers.Conv2D(filters, kernel_size, padding='same',  # Sửa lỗi 'same Beh' thành 'same'
                      kernel_regularizer=tf.keras.regularizers.l2(0.002))(y)
    shortcut = layers.Conv2D(filters, 1, padding='same')(x) if x.shape[-1] != filters else x
    y = layers.Add()([shortcut, y])
    y = layers.Activation('relu')(y)
    return y

def build_resnet(input_shape=(128, 32, 1), num_classes=11):
    inputs = layers.Input(shape=input_shape)  # Dùng shape thay vì batch_shape
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
        # Thử tải trực tiếp mô hình
        model = tf.keras.models.load_model(model_path, compile=False)
        print("Đã tải mô hình thành công từ file!")
    except Exception as e:
        print(f"Không thể tải trực tiếp mô hình: {e}")
        print("Đang thử định nghĩa lại mô hình và tải trọng số...")
        # Định nghĩa lại mô hình và tải trọng số
        model = build_resnet(input_shape=(128, 32, 1), num_classes=11)
        try:
            model.load_weights(model_path)
            print("Đã tải trọng số thành công!")
        except Exception as weight_error:
            raise ValueError(f"Không thể tải trọng số: {weight_error}")
    return model

# Hàm chính để kiểm tra mô hình
def main():
    print("TensorFlow Version:", tf.__version__)
    
    # Đường dẫn đến file mô hình
    model_path = r'D:\Python\AI-Instrument-Classifier\models\resnet_mel_instrument_classifier_89.h5'
    
    # Tải mô hình và in summary
    try:
        model = load_model_with_fallback(model_path)
        model.summary()  # In cấu trúc mô hình
    except Exception as e:
        print(f"Không thể tải mô hình: {e}")

if __name__ == "__main__":
    main()