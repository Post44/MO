from sklearn.model_selection import train_test_split
import numpy as np
import tarfile
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import LearningRateScheduler

def extract_images_labels(archive_path):
    img_list = []
    lbl_list = []
    categories = []  
    valid_ext = ['.png']  

    with tarfile.open(archive_path, 'r:gz') as archive:
        for member in archive.getmembers():
            if member.isdir():
                continue  

            dir_name = member.name.split('/')[1]  
            if dir_name not in categories:
                categories.append(dir_name)  

            lbl = categories.index(dir_name)  

            if any(member.name.endswith(ext) for ext in valid_ext):
                file_obj = archive.extractfile(member)
                try:
                    img = Image.open(BytesIO(file_obj.read()))
                    img = img.convert('L') 
                    img_array = np.array(img)

                    img_list.append(img_array)
                    lbl_list.append(lbl)
                except Exception as e:
                    print(f"Ошибка при обработке {member.name}: {e}")
            else:
                print(f"Пропущен файл {member.name} с неподдерживаемым расширением")

    return np.array(img_list), np.array(lbl_list), categories

def build_network(input_dim, layer_sizes, activation_funcs, drop_rate, output_dim):
    model = Sequential()
    model.add(Flatten(input_shape=input_dim))
    model.add(Dense(layer_sizes[0], activation=activation_funcs[0]))
    model.add(Dropout(drop_rate))
    
    for size, act in zip(layer_sizes[1:], activation_funcs[1:]):
        model.add(Dense(size, activation=act))
        model.add(Dropout(drop_rate))
    
    model.add(Dense(output_dim, activation='softmax'))
    return model

def train_network(model, X_train, y_train, X_val, y_val, lr_start, num_epochs, batch_sz):
    def adjust_lr(epoch, lr):
        return lr * 0.5 if epoch > 5 else lr
    
    lr_adjustment = LearningRateScheduler(adjust_lr)
    model.compile(optimizer=SGD(learning_rate=lr_start),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    history = model.fit(X_train, y_train,
                        validation_data=(X_val, y_val),
                        epochs=num_epochs,
                        batch_size=batch_sz,
                        callbacks=[lr_adjustment],
                        verbose=1)
    return history

def test_network(model, X_test, y_test):
    _, acc = model.evaluate(X_test, y_test, verbose=0)
    return acc

def check_balance(labels, categories):
    counts = np.zeros(len(categories), dtype=int)
    for lbl in labels:
        counts[lbl] += 1

    for i, category in enumerate(categories):
        print(f"Класс {category}: {counts[i]} изображений")

def show_sample_images(images, labels, categories, num_samples=10):
    indices = np.random.choice(len(images), num_samples, replace=False)
    plt.figure(figsize=(10, 5))
    
    for i, idx in enumerate(indices):
        plt.subplot(2, 5, i + 1)  
        plt.imshow(images[idx], cmap='gray')  
        plt.title(f"Class: {categories[labels[idx]]}") 
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def find_duplicates(train_data, val_data, test_data):
    train_set = {x.tobytes() for x in train_data}  
    val_set = {x.tobytes() for x in val_data}
    test_set = {x.tobytes() for x in test_data}

    train_val_dupes = train_set & val_set
    train_test_dupes = train_set & test_set

    if train_val_dupes:
        print(f"Дубликаты между обучающей и валидационной выборками: {len(train_val_dupes)}")
    if train_test_dupes:
        print(f"Дубликаты между обучающей и тестовой выборками: {len(train_test_dupes)}")
    if not train_val_dupes and not train_test_dupes:
        print("Нет дубликатов между выборками.")

def remove_duplicate_entries(X_train, y_train, X_val, X_test):
    unique_train = []
    unique_labels = []
    seen = set()

    for i, img in enumerate(X_train):
        img_bytes = img.tobytes()
        if img_bytes not in seen and img_bytes not in {x.tobytes() for x in X_val} and img_bytes not in {x.tobytes() for x in X_test}:
            seen.add(img_bytes)
            unique_train.append(img)
            unique_labels.append(y_train[i])
    
    return np.array(unique_train), np.array(unique_labels)

data_path = 'notMNIST_small.tar.gz'

images, labels, categories = extract_images_labels(data_path)

print(f"\nКоличество изображений: {images.shape[0]}")
print(f"Размер изображений: {images.shape[1:]}")
print(f"Количество классов: {len(categories)}")
print(f"Имена классов: {categories}\n")

check_balance(labels, categories)
show_sample_images(images, labels, categories)

X_train, X_temp, y_train, y_temp = train_test_split(images, labels, test_size=0.15, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=2/3, random_state=42)

print(f"\nРазмер обучающей выборки: {X_train.shape[0]}")
print(f"Размер валидационной выборки: {X_val.shape[0]}")
print(f"Размер тестовой выборки: {X_test.shape[0]}\n")

print("Балансировка обучающей выборки:")
check_balance(y_train, categories)
print("\nБалансировка валидационной выборки:")
check_balance(y_val, categories)
print("\nБалансировка тестовой выборки:")
check_balance(y_test, categories)
print()

find_duplicates(X_train, X_val, X_test)
print()

X_train_clean, y_train_clean = remove_duplicate_entries(X_train, y_train, X_val, X_test)
print("Очистка дубликатов завершена\n")

find_duplicates(X_train_clean, X_val, X_test)
print()

layers = [128, 64, 32]
activations = ['relu', 'relu', 'relu']
drop_prob = 0.1
learning_rate = 0.2
epochs_count = 15
batch_size_val = 32

X_train_clean = X_train_clean.astype('float32') / 255.0
X_val = X_val.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

input_dim = (28, 28, 1)
num_classes = len(np.unique(y_train_clean))

neural_net = build_network(input_dim, layers, activations, drop_prob, num_classes)

train_network(neural_net, X_train_clean, y_train_clean, X_val, y_val, learning_rate, epochs_count, batch_size_val)

final_accuracy = test_network(neural_net, X_test, y_test)

print(f"\nТочность модели: {final_accuracy}")