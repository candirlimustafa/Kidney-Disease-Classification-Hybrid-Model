import os
import numpy as np
import random
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping


DATASET_PATH = r'C:\Users\mstfc\Desktop\bobrek\data\raw' 

# Görüntü AyarlarıIMG_SIZE = (224, 224)
NUM_CLASSES = 4 # Cyst, Normal, Stone, Tumor  (normalizasyon işlemi )


EPOCHS_FOR_GA = 1      
POPULATION_SIZE = 3     
GENERATIONS = 2         


FINAL_EPOCHS = 10       


print("GPU Mevcut mu?: ", len(tf.config.list_physical_devices('GPU')) > 0)

# Veri Yükleyici (Data Generator) - Validasyon ayırma (%20)
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2 
)


def build_model(learning_rate, dropout_rate):
    """
    EfficientNetB0 tabanlı model oluşturur.
    """
    base_model = EfficientNetB0(   #BURADA YÖNTEM 1 İN 1. ADIMI GERÇEKLEŞİYOR MODEL YÜKLEME YAPILIYOR
        weights='imagenet',
        include_top=False, 
        input_shape=(224, 224, 3)
    )
    
    base_model.trainable = True 

    x = base_model.output                           #BURADA YÖNTEM 1 İN 2. ADIMI GERÇEKLEŞTİRİYORUZ SON KATMANI DEĞİŞTİRİP 4 NOROBLÜ YENİ ÇIKIŞ KATMANI EKLENDİ
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout_rate)(x)
    predictions = layers.Dense(NUM_CLASSES, activation='softmax')(x)    # 4 Sınıf (Normal, Kist, Taş, Tümör)

    model = models.Model(inputs=base_model.input, outputs=predictions)

    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# --- 3. GENETİK ALGORİTMA (HİPERPARAMETRE OPTİMİZASYONU) --- BURADA YÖNTEM 1 3. ADIM GERÇEKLEŞİYOR GENETİK ALGORİTMA İLE PARAMETRE SEÇİMİ

param_space = {
    'learning_rate': [1e-3, 1e-4],
    'batch_size': [16, 32], 
    'dropout_rate': [0.2, 0.3, 0.4]
}

def create_individual():
    return {k: random.choice(v) for k, v in param_space.items()}

def fitness_function(individual):
    """Bireyin başarısını ölçer (HATA BURADAYDI, DÜZELTİLDİ)."""
    print(f"\n--- Birey Test Ediliyor: {individual} ---")
    
    train_gen = datagen.flow_from_directory(
        DATASET_PATH, target_size=IMG_SIZE, batch_size=individual['batch_size'],
        class_mode='categorical', subset='training'
    )
    val_gen = datagen.flow_from_directory(
        DATASET_PATH, target_size=IMG_SIZE, batch_size=individual['batch_size'],
        class_mode='categorical', subset='validation'
    )

    try:
        model = build_model(individual['learning_rate'], individual['dropout_rate'])
        
        history = model.fit(
            train_gen,
            epochs=EPOCHS_FOR_GA,
            validation_data=val_gen,
            verbose=1
        )
        score = history.history['val_accuracy'][-1]
    except Exception as e:
        print(f"Hata oluştu (Birey atlanıyor): {e}")
        score = 0

    print(f" -> Skor (Val Acc): {score:.4f}")
    tf.keras.backend.clear_session()
    return score

def crossover(parent1, parent2):
    child = {}
    for key in parent1:
        child[key] = random.choice([parent1[key], parent2[key]])
    return child

def mutate(individual):
    if random.random() < 0.2:
        key = random.choice(list(param_space.keys()))
        individual[key] = random.choice(param_space[key])
    return individual

# --- GA DÖNGÜSÜ ---
population = [create_individual() for _ in range(POPULATION_SIZE)]
best_individual = None
best_score = -1

print(f"\n>>> Genetik Algoritma Başlıyor ({GENERATIONS} Nesil)...")

for gen in range(GENERATIONS):
    print(f"\n=== NESİL {gen+1} ===")
    
    scores = []
    for ind in population:
        acc = fitness_function(ind)
        scores.append((ind, acc))
        if acc > best_score:
            best_score = acc
            best_individual = ind
            print(f"!!! YENİ EN İYİ BULUNDU: {best_score:.4f} !!!")
    
    scores.sort(key=lambda x: x[1], reverse=True)
    top_parents = [x[0] for x in scores[:2]] 
    
    new_pop = top_parents[:]
    while len(new_pop) < POPULATION_SIZE:
        p1, p2 = random.choice(top_parents), random.choice(top_parents)
        child = mutate(crossover(p1, p2))
        new_pop.append(child)
    
    population = new_pop

print(f"\n*** OPTİMİZASYON TAMAMLANDI ***")
print(f"En İyi Parametreler: {best_individual}")

# --- 4. İNCE AYAR (FINE-TUNING) ---
print("\n>>> Final Model Eğitimi Başlıyor...")

final_bs = best_individual['batch_size']
final_lr = best_individual['learning_rate']
final_dr = best_individual['dropout_rate']

final_train_gen = datagen.flow_from_directory(
    DATASET_PATH, target_size=IMG_SIZE, batch_size=final_bs,
    class_mode='categorical', subset='training'
)
final_val_gen = datagen.flow_from_directory(
    DATASET_PATH, target_size=IMG_SIZE, batch_size=final_bs,
    class_mode='categorical', subset='validation'
)

model = build_model(final_lr, final_dr)  # BURADA YÖNTEM 1 İN 4. MADDESİ İNCE AYAR KISMI YAPILIYOR GENETİK ALGORİTMADAKİ PARAMETRELER ALINIM FAZLA EPOCH SAYISI İLE EĞİTİLİYOR

callbacks = [
    ModelCheckpoint('best_kidney_model.keras', save_best_only=True, monitor='val_accuracy'),
    EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
]

history = model.fit(
    final_train_gen,
    epochs=FINAL_EPOCHS,
    validation_data=final_val_gen,
    callbacks=callbacks
)


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(len(acc))

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()