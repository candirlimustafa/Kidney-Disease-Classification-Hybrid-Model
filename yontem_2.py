import os
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler # YENİ EKLENDİ
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


DATASET_PATH = r'C:\Users\mstfc\Desktop\bobrek\data\raw'
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

print(f"Veri Yolu: {DATASET_PATH}")

# =============================================================================
# ADIM 1: MODEL YÜKLEME
# =============================================================================
print("\n>>> ADIM 1: EfficientNet-B0 Modeli Yükleniyor...")

feature_extractor = EfficientNetB0(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3),
    pooling='avg'
)
feature_extractor.trainable = False

# =============================================================================
# ADIM 2: ÖZELLİK ÇIKARIMI (DÜZELTİLDİ)
# =============================================================================
print("\n>>> ADIM 2: Görüntülerden Özellik Vektörleri Çıkarılıyor...")


datagen = ImageDataGenerator() 

if os.path.exists(DATASET_PATH):
    generator = datagen.flow_from_directory(
        DATASET_PATH,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False 
    )
else:
    raise ValueError("Hata: Veri seti klasörü bulunamadı!")


features = feature_extractor.predict(generator, verbose=1)
labels = generator.classes

print(f"Özellik Çıkarımı Tamamlandı! Boyut: {features.shape}")


print(">>> Özellikler Standartlaştırılıyor...")
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)


X_train, X_test, y_train, y_test = train_test_split(
    features_scaled, labels, test_size=0.2, random_state=42, stratify=labels
)

# =============================================================================
# ADIM 3: SVM OPTİMİZASYONU (GA)
# =============================================================================
print("\n>>> ADIM 3: SVM Hiperparametre Optimizasyonu (GA) Başlıyor...")

# DÜZELTME 3: 'gamma' için 'scale' ve 'auto' eklendi.
param_space = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.1, 0.01, 0.001],
    'kernel': ['rbf']
}

def create_individual():
    return {
        'C': random.choice(param_space['C']),
        'gamma': random.choice(param_space['gamma']),
        'kernel': 'rbf'
    }

def fitness_function(individual):

    subset_size = int(len(X_train) * 0.1) 
    X_sub = X_train[:subset_size]
    y_sub = y_train[:subset_size]
    
    clf = SVC(C=individual['C'], gamma=individual['gamma'], kernel='rbf')
    clf.fit(X_sub, y_sub)
    
    predictions = clf.predict(X_test[:subset_size])
    return accuracy_score(y_test[:subset_size], predictions)

def crossover(p1, p2):
    child = p1.copy()
    if random.random() < 0.5: child['C'] = p2['C']
    if random.random() < 0.5: child['gamma'] = p2['gamma']
    return child

def mutate(individual):
    if random.random() < 0.2:
        key = random.choice(['C', 'gamma'])
        individual[key] = random.choice(param_space[key])
    return individual

# GA Ayarları
POPULATION_SIZE = 4
GENERATIONS = 2

population = [create_individual() for _ in range(POPULATION_SIZE)]
best_individual = None
best_score = -1

for gen in range(GENERATIONS):
    print(f"\n--- Nesil {gen+1}/{GENERATIONS} ---")
    scores = []
    
    for ind in population:
        try:
            acc = fitness_function(ind)
            scores.append((ind, acc))

            
            if acc > best_score:
                best_score = acc
                best_individual = ind
                print(f"!!! YENİ EN İYİ: {acc:.4f} (Param: {ind}) !!!")
        except Exception as e:
            print(f"Hata: {e}")
            scores.append((ind, 0))

    scores.sort(key=lambda x: x[1], reverse=True)
    top_parents = [x[0] for x in scores[:2]]
    
    new_population = top_parents[:]
    while len(new_population) < POPULATION_SIZE:
        p1 = random.choice(top_parents)
        p2 = random.choice(top_parents)
        child = mutate(crossover(p1, p2))
        new_population.append(child)
    population = new_population

print(f"\n*** OPTİMİZASYON TAMAMLANDI ***")
print(f"En İyi Parametreler: {best_individual}")

# =============================================================================
# ADIM 4: FİNAL EĞİTİM VE SONUÇLAR
# =============================================================================
print("\n>>> ADIM 4: Final Modeli Eğitimi (Tüm Veriyle)...")

final_svm = SVC(
    C=best_individual['C'], 
    gamma=best_individual['gamma'], 
    kernel='rbf'
)

final_svm.fit(X_train, y_train)
y_pred = final_svm.predict(X_test)


class_names = list(generator.class_indices.keys())
print("\n--- Sınıflandırma Raporu ---")
print(classification_report(y_test, y_pred, target_names=class_names))


cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('Karışıklık Matrisi (Hibrit Model)')
plt.ylabel('Gerçek Sınıf')
plt.xlabel('Tahmin Edilen Sınıf')
plt.savefig('hibrit_model_sonuc_duzeltilmis.png')
print("\nGrafik kaydedildi.")
plt.show()