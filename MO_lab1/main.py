import tarfile
import numpy as np
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

archive_path = 'notMNIST_small.tar.gz'

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

def train_logistic_model(X_train, y_train, X_test, y_test, sample_sizes):
    acc_scores = []
    
    for size in sample_sizes:
        X_train_sample = X_train[:size]
        y_train_sample = y_train[:size]
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_sample.reshape(len(X_train_sample), -1))
        X_test_scaled = scaler.transform(X_test.reshape(len(X_test), -1))

        model = LogisticRegression(max_iter=11, class_weight="balanced")
        model.fit(X_train_scaled, y_train_sample)

        y_pred = model.predict(X_test_scaled)
        acc = accuracy_score(y_test, y_pred)
        acc_scores.append(acc)
        
    return acc_scores

images, labels, class_names = extract_images_labels(archive_path)

print(f"Всего изображений: {images.shape[0]}")
print(f"Размер изображений: {images.shape[1:]}")
print(f"Классы: {class_names}")

check_balance(labels, class_names)
show_sample_images(images, labels, class_names)

X_train, X_temp, y_train, y_temp = train_test_split(images, labels, test_size=0.15, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=2/3, random_state=42)

print(f"Размеры выборок: обучающая - {X_train.shape[0]}, валидационная - {X_val.shape[0]}, тестовая - {X_test.shape[0]}")
print(f"Обучающая выборка")
check_balance(y_train, class_names)
print(f"Валидационная выборка")
check_balance(y_val, class_names)
print(f"Тестовая выборка")
check_balance(y_test, class_names)

find_duplicates(X_train, X_val, X_test)

X_train_clean, y_train_clean = remove_duplicate_entries(X_train, y_train, X_val, X_test)
print("Дубликаты удалены.")

find_duplicates(X_train_clean, X_val, X_test)

sample_sizes = [50, 100, 1000, 20000]
accuracies = train_logistic_model(X_train_clean, y_train_clean, X_test, y_test, sample_sizes)

print("Точность классификатора при разных размерах выборки:")
for size, acc in zip(sample_sizes, accuracies):
    print(f"{size}: {acc:.4f}")

plt.plot(sample_sizes, accuracies, marker='o')
plt.xlabel('Размер обучающей выборки')
plt.ylabel('Точность')
plt.title('Точность модели в зависимости от размера выборки')
plt.grid(True)
plt.show()
