import gdown
import zipfile
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import librosa

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from scipy.signal import butter, lfilter
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

FILE_ID = "1bceZrbOMPSXTTTMBx8XqDBwsSMussPHj"
zip_path = "CEG3004_Project_Data.zip"

# Download the zip file
gdown.download(f"https://drive.google.com/uc?id={FILE_ID}", zip_path, quiet=False)

# Define your local project root
DATA_ROOT = r"C:\Users\omgwt\PycharmProjects\CEG3004_DSP_Proj"

# Make sure the directory exists
os.makedirs(DATA_ROOT, exist_ok=True)

# Extract the zip into your project root
with zipfile.ZipFile(zip_path, 'r') as z:
    z.extractall(DATA_ROOT)

print("Dataset extracted to:", DATA_ROOT)

# Now set train and submission directories
TRAIN_DIR = os.path.join(DATA_ROOT, 'data', 'train')
SUBMISSION_DIR = os.path.join(DATA_ROOT, 'data', 'submission')

print('TRAIN_DIR:', TRAIN_DIR)
print('SUBMISSION_DIR:', SUBMISSION_DIR)

assert os.path.exists(TRAIN_DIR), 'Train directory not found'
assert os.path.exists(SUBMISSION_DIR), 'Submission directory not found'

labels_csv = os.path.join(TRAIN_DIR, 'labels.csv')
audio_train_dir = os.path.join(TRAIN_DIR, 'audio')

df = pd.read_csv(labels_csv)
df['clip_id'] = df['clip_id'].astype(str)
df['label'] = df['label'].astype(str)
print('Train rows:', len(df))
df.head()


### FUNCTIONS ###

def load_audio(path, sr=16000):
    """Load mono audio, resample to sr."""
    y, sr_out = librosa.load(path, sr=sr, mono=True)
    y = np.nan_to_num(y).astype(np.float32)
    return y, sr_out


def preprocess_audio(y, sr, target_len=5.0):
    """Improved preprocessing."""

    # 1. Remove silence
    y, _ = librosa.effects.trim(y, top_db=20)

    # 2. Normalize (peak normalization)
    if np.max(np.abs(y)) > 0:
        y = y / np.max(np.abs(y))

    # 3. Pre-emphasis filter (boost high freq)
    y = np.append(y[0], y[1:] - 0.97 * y[:-1])

    # 4. Fix length (pad or truncate to 5 seconds)
    target_samples = int(sr * target_len)

    if len(y) < target_samples:
        y = np.pad(y, (0, target_samples - len(y)))
    else:
        y = y[:target_samples]

    return y


def augment_noise(y, noise_factor=None):
    if noise_factor is None:
        noise_factor = np.random.uniform(0.001, 0.02)
    noise = np.random.randn(len(y))
    return y + noise_factor * noise


def augment_gain(y, gain_range=(0.7, 1.3)):
    gain = np.random.uniform(*gain_range)
    return y * gain


def bandpass_filter(y, sr, low=None, high=None):
    if low is None:
        low = np.random.uniform(200, 1000)
    if high is None:
        high = np.random.uniform(3000, 6000)
    if low >= high:
        low, high = 200, 4000
    b, a = butter(4, [low / (sr / 2), high / (sr / 2)], btype='band')
    return lfilter(b, a, y)


def augment_shift(y, max_shift=0.2):
    shift = int(np.random.uniform(-max_shift, max_shift) * len(y))
    return np.roll(y, shift)


# Feature extraction (baseline: MFCC stats)

def features_mfcc_stats(y, sr, n_mfcc=20, n_fft=1024, hop=256):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop)
    d1 = librosa.feature.delta(mfcc)
    d2 = librosa.feature.delta(mfcc, order=2)

    def stats(M):
        return np.concatenate([M.mean(axis=1), M.std(axis=1)], axis=0)

    return np.concatenate([stats(mfcc), stats(d1), stats(d2)], axis=0).astype(np.float32)


def extract_features(path, sr=16000, augment_type=None):
    y, sr = load_audio(path, sr)
    y = preprocess_audio(y, sr)

    if augment_type == "noise":
        y = augment_noise(y)
    elif augment_type == "gain":
        y = augment_gain(y)
    elif augment_type == "bandpass":
        y = bandpass_filter(y, sr)
    elif augment_type == "shift":
        y = augment_shift(y)

    def stats(x):
        # Split into 4 temporal segments and compute stats per segment
        n_frames = x.shape[1]
        seg_size = n_frames // 4
        segment_stats = []
        for i in range(4):
            seg = x[:, i * seg_size:(i + 1) * seg_size]
            if seg.shape[1] > 0:
                segment_stats.extend([seg.mean(axis=1), seg.std(axis=1)])
            else:
                segment_stats.extend([np.zeros(x.shape[0]), np.zeros(x.shape[0])])
        return np.concatenate([
            x.mean(axis=1), x.std(axis=1),  # global stats
            *segment_stats  # per-segment stats
        ])

    # MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)

    # Spectral features
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)

    # Zero crossing rate
    zcr = librosa.feature.zero_crossing_rate(y)

    # Log-mel spectrogram stats
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=40, n_fft=1024, hop_length=256)
    log_mel = librosa.power_to_db(mel)

    # Spectral contrast (captures peaks vs valleys across frequency bands)
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_fft=1024, hop_length=256)

    # RMS energy
    rms = librosa.feature.rms(y=y)

    # Spectral flatness (tonal vs noisy)
    #flatness = librosa.feature.spectral_flatness(y=y)


    features = np.concatenate([
        stats(mfcc),
        stats(mfcc_delta),
        stats(mfcc_delta2),
        stats(centroid),
        stats(bandwidth),
        stats(rolloff),
        stats(zcr),
        stats(log_mel),
        stats(contrast),
        stats(rms)
    ])

    return features.astype(np.float32)


# Build training feature matrix

X, y, paths = [], [], []
classes = sorted(df['label'].unique().tolist())
label_to_idx = {c: i for i, c in enumerate(classes)}
idx_to_label = {i: c for c, i in label_to_idx.items()}

for _, r in tqdm(df.iterrows(), total=len(df)):
    clip_id = r['clip_id']
    wav_path = os.path.join(audio_train_dir, f'{clip_id}.wav')
    X.append(extract_features(wav_path, augment_type=None))
    y.append(label_to_idx[r['label']])
    paths.append(wav_path)  # keep the path for later

X = np.stack(X, axis=0)
y = np.array(y, dtype=np.int64)
print('X shape:', X.shape, 'num_classes:', len(classes))

X_tr, X_va, y_tr, y_va, paths_tr, paths_va = train_test_split(
    X, y, paths, test_size=0.2, random_state=42, stratify=y
)

# TODO: Change model/hyperparameters

# Build augmented dataset: clean + each augmentation
X_tr_augmented, y_tr_augmented = [], []
count = 0

for wav_path, label in zip(paths_tr, y_tr):
    X_tr_augmented.append(extract_features(wav_path, augment_type=None))
    y_tr_augmented.append(label)
    for aug in ["noise", "gain", "bandpass", "shift"]:
        for i in range(2):
            X_tr_augmented.append(extract_features(wav_path, augment_type=aug))
            y_tr_augmented.append(label)

    count += 1
    print(f'Audio file no. {count}')

X_tr_combined = np.stack(X_tr_augmented, axis=0)
y_tr_combined = np.array(y_tr_augmented, dtype=np.int64)

model = Pipeline([
    ('clf', RandomForestClassifier(
        n_estimators=300,
        max_depth=15,
        min_samples_leaf=5,
        max_features='sqrt',
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    ))
])


cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X_tr_combined, y_tr_combined,
                         cv=cv, scoring='f1_macro', n_jobs=-1)
print(f'CV Macro-F1: {scores.mean():.4f} ± {scores.std():.4f}')

# --- Final fit on ALL augmented training data ---
model.fit(X_tr_combined, y_tr_combined)

# --- Evaluate on held-out validation set ---
y_pred = model.predict(X_va)

print(classification_report(y_va, y_pred, target_names=[idx_to_label[i] for i in range(len(classes))]))
print('Macro-F1:', f1_score(y_va, y_pred, average='macro'))

model_filename = 'Pr_7_model' + '.joblib'
joblib.dump(model, model_filename)
print(f'Model saved as {model_filename}. Downloading to your computer...')

cm = confusion_matrix(y_va, y_pred)
fig, ax = plt.subplots(figsize=(12, 10))
ConfusionMatrixDisplay(cm, display_labels=[idx_to_label[i] for i in range(len(classes))]).plot(ax=ax, xticks_rotation=45)
plt.tight_layout()
plt.show()

sub_meta = pd.read_csv(os.path.join(SUBMISSION_DIR, 'metadata.csv'))
sub_meta['clip_id'] = sub_meta['clip_id'].astype(str)
audio_sub_dir = os.path.join(SUBMISSION_DIR, 'audio')
print('Submission rows:', len(sub_meta))
sub_meta.head()

OUT_CSV = f'Pr_7_predictions.csv'

pred_rows = []
for _, r in tqdm(sub_meta.iterrows(), total=len(sub_meta)):
    clip_id = r['clip_id']
    wav_path = os.path.join(audio_sub_dir, f'{clip_id}.wav')
    feat = extract_features(wav_path)
    pred_idx = int(model.predict(feat.reshape(1, -1))[0])
    pred_label = idx_to_label[pred_idx]
    pred_rows.append((clip_id, pred_label))

out = pd.DataFrame(pred_rows, columns=['clip_id', 'predicted_label'])
out.to_csv(OUT_CSV, index=False)

print(f'Predictions saved to {OUT_CSV}. Downloading to your computer...')