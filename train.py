import os
import numpy as np
import librosa
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
import pickle
import random

# ======================
# PATH DATASET
# ======================
data_path = "RAVDESS"

# ======================
# LABEL MAP
# ======================
emotion_map = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fear",
    "07": "disgust",
    "08": "surprise"
}

# ======================
# AUGMENTATION
# ======================
def augment_audio(audio, sr):
    choice = random.choice(["noise", "pitch", "speed", "none"])

    if choice == "noise":
        noise = np.random.randn(len(audio))
        audio = audio + 0.005 * noise

    elif choice == "pitch":
        audio = librosa.effects.pitch_shift(audio, sr=sr, n_steps=2)

    elif choice == "speed":
        audio = librosa.effects.time_stretch(audio, rate=1.2)

    return audio

# ======================
# FEATURE EXTRACTION
# ======================
def extract_mel(file_path, augment=False):
    audio, sr = librosa.load(file_path, sr=22050, duration=3)

    if augment:
        audio = augment_audio(audio, sr)

    mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    # padding
    if mel_db.shape[1] < 128:
        pad = 128 - mel_db.shape[1]
        mel_db = np.pad(mel_db, ((0,0),(0,pad)), mode='constant')

    mel_db = mel_db[:, :128]

    # 🔥 STANDARDIZATION (LEBIH BAGUS)
    mel_db = (mel_db - np.mean(mel_db)) / (np.std(mel_db) + 1e-6)

    return mel_db

# ======================
# LOAD DATA
# ======================
X, y = [], []

all_files = []
for root, _, files in os.walk(data_path):
    for f in files:
        if f.endswith(".wav"):
            all_files.append(os.path.join(root, f))

print("Total files:", len(all_files))

for file_path in tqdm(all_files):
    file_name = os.path.basename(file_path)

    emotion_code = file_name.split("-")[2]
    emotion = emotion_map[emotion_code]

    # original
    mel = extract_mel(file_path, augment=False)
    X.append(mel)
    y.append(emotion)

    # 🔥 AUGMENTED DATA
    mel_aug = extract_mel(file_path, augment=True)
    X.append(mel_aug)
    y.append(emotion)

X = np.array(X)
y = np.array(y)

X = X[..., np.newaxis]

# ======================
# LABEL ENCODING
# ======================
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# ======================
# SPLIT DATA
# ======================
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

# ======================
# MODEL (IMPROVED CNN)
# ======================
model = tf.keras.Sequential([

    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(128,128,1)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(256, (3,3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(len(le.classes_), activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# ======================
# CALLBACKS
# ======================
callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(patience=3)
]

# ======================
# TRAINING
# ======================
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=50,
    batch_size=32,
    callbacks=callbacks
)

# ======================
# EVALUATION
# ======================
loss, acc = model.evaluate(X_test, y_test)
print("Accuracy:", acc)

# ======================
# SAVE
# ======================
model.save("emotion_model.keras")

with open("label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)

print("MODEL IMPROVED & SAVED!")