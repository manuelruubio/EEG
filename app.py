from flask import Flask, request, render_template_string
import numpy as np
import scipy.io
import mne
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, DepthwiseConv2D, SeparableConv2D, BatchNormalization, Activation, AveragePooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from werkzeug.utils import secure_filename

# ===========================
# Configuración inicial
# ===========================
app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
tsfreq = 128
window_size = 4 * tsfreq

# ===========================
# Modelo EEGNet
# ===========================
def build_eegnet(input_shape=(19, window_size, 1), num_classes=2):
    model = Sequential([
        Conv2D(32, (1, 64), padding='same', kernel_regularizer=l2(0.0005), input_shape=input_shape),
        BatchNormalization(),
        DepthwiseConv2D((19, 1), use_bias=False, depth_multiplier=2, padding='valid', depthwise_regularizer=l2(0.0005)),
        BatchNormalization(),
        Activation('elu'),
        AveragePooling2D((1, 4)),
        Dropout(0.5),
        SeparableConv2D(32, (1, 16), padding='same'),
        BatchNormalization(),
        Activation('elu'),
        AveragePooling2D((1, 8)),
        Dropout(0.5),
        Flatten(),
        Dense(num_classes, activation='softmax', kernel_regularizer=l2(0.0005))
    ])
    model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
    return model

model = build_eegnet()
model.load_weights("modelo_eegnet.weights.h5")

# ===========================
# Preprocesamiento
# ===========================
def preprocess_file(file_path):
    mat_data = scipy.io.loadmat(file_path)
    key = list(mat_data.keys())[-1]
    data = mat_data[key].T

    raw = mne.io.RawArray(data, mne.create_info(ch_names=[f"Ch{i}" for i in range(data.shape[0])], sfreq=tsfreq, ch_types=["eeg"]*data.shape[0]), verbose=False)
    raw.filter(l_freq=0.5, h_freq=60, fir_design="firwin", verbose=False)

    ica = mne.preprocessing.ICA(n_components=19, random_state=97, max_iter=800)
    ica.fit(raw)
    ica.exclude = [0, 1]
    raw = ica.apply(raw)

    eeg_filtered_data = raw.get_data()
    num_samples = eeg_filtered_data.shape[1]
    num_windows = num_samples // window_size
    eeg_windows = eeg_filtered_data[:, :num_windows * window_size].reshape(data.shape[0], num_windows, window_size)
    eeg_windows = np.transpose(eeg_windows, (1, 0, 2))
    eeg_windows = (eeg_windows - np.mean(eeg_windows)) / np.std(eeg_windows)

    return eeg_windows[..., np.newaxis]

# ===========================
# Rutas
# ===========================
HTML = """
<!DOCTYPE html>
<html>
<head><title>Clasificador EEG</title></head>
<body style="text-align: center; padding-top: 50px;">
    <h1>Clasificador de EEG - TDAH</h1>
    <form method="POST" enctype="multipart/form-data">
        <input type="file" name="file" accept=".mat" required>
        <br><br>
        <button type="submit">Subir y clasificar</button>
    </form>
    {% if pred is not none %}
        <h2>Resultado: {{ pred }}</h2>
    {% endif %}
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def index():
    pred = None
    if request.method == "POST":
        f = request.files["file"]
        filename = secure_filename(f.filename)
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        f.save(file_path)

        x = preprocess_file(file_path)
        y_probs = model.predict(x)
        y_pred = np.argmax(np.mean(y_probs, axis=0))

        pred = "ADHD" if y_pred == 1 else "Control"

    return render_template_string(HTML, pred=pred)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)