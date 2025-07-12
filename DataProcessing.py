# === Script 1: generate_spectrograms.py ===
import os, glob
import numpy as np
import librosa
import matplotlib.pyplot as plt
import librosa.display
from PIL import Image

# Settings
RAW_DATA_DIR = 'D:/UNI/EECE 4 second term/Nueral Networks/Projects Mohsen/assignment3_mohsen/free-spoken-digit-dataset-master/recordings'   # adjust to your path
TRAIN_OUT_DIR = 'Train_spectrograms'
TEST_OUT_DIR  = 'Test_spectrograms'
SAMPLE_RATE   = 8000
FRAME_LENGTH  = int(SAMPLE_RATE * 0.015)  # 15ms
HOP_LENGTH    = FRAME_LENGTH - 20        # 100 samples
N_MELS        = 128


def scale_minmax(X, min_val=0.0, max_val=1.0):
    X_std = (X - X.min()) / (X.max() - X.min())
    return X_std * (max_val - min_val) + min_val


def parse_filename(fp):
    name = os.path.splitext(os.path.basename(fp))[0]
    digit, person, idx = name.split('_')
    return digit, person, int(idx)


def build_file_lists(raw_dir):
    recs = {}
    for fp in glob.glob(os.path.join(raw_dir, '*.wav')):
        d,p,i = parse_filename(fp)
        recs.setdefault(d, {}).setdefault(p, []).append((i, fp))
    train, test = [], []
    for d, persons in recs.items():
        for p, lst in persons.items():
            for idx, fp in sorted(lst):
                if idx <= 4:
                    test.append((fp, d))
                else:
                    train.append((fp, d))
    return train, test


def process_and_save(fp, out_path):
    y, sr = librosa.load(fp, sr=SAMPLE_RATE)
    S = librosa.feature.melspectrogram(
        y=y, sr=sr,
        n_mels=N_MELS,
        n_fft=FRAME_LENGTH,
        hop_length=HOP_LENGTH
    )
    mels = np.log(S + 1e-9)
    img = (scale_minmax(mels, 0, 255)).astype(np.uint8)
    img = np.flip(img, axis=0)
    img = 255 - img

    # Directly save the array as an image to preserve original dimensions
    im = Image.fromarray(img)
    im.save(out_path)

    # plt.figure(figsize=(3,3))
    # librosa.display.specshow(img, sr=sr, hop_length=HOP_LENGTH, x_axis=None, y_axis=None)
    # plt.axis('off')
    # plt.savefig(out_path, bbox_inches='tight', pad_inches=0)
    # plt.close()


if __name__ == '__main__':
    # create output directories
    for base in [TRAIN_OUT_DIR, TEST_OUT_DIR]:
        for d in map(str, range(10)):
            os.makedirs(os.path.join(base, d), exist_ok=True)

    train_list, test_list = build_file_lists(RAW_DATA_DIR)
    print(f"Train files: {len(train_list)}, Test files: {len(test_list)}")

    for fp, d in train_list:
        out = os.path.join(TRAIN_OUT_DIR, d, f"{os.path.basename(fp)[:-4]}.png")
        process_and_save(fp, out)
    for fp, d in test_list:
        out = os.path.join(TEST_OUT_DIR, d, f"{os.path.basename(fp)[:-4]}.png")
        process_and_save(fp, out)
    print("Done generating spectrograms.")