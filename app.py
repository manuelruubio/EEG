import streamlit as st
import numpy as np
import scipy.io
import mne
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv2D, DepthwiseConv2D, SeparableConv2D,
                                     BatchNormalization, Activation, AveragePooling2D,
                                     Flatten, Dense, Dropout)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam

# Configuración
tsfreq = 128
window_size = 4 * tsfreq
input_shape = (19, window_size, 1)
num_classes = 2

# ===========================
# Definir arquitectura EEGNet
# ===========================
def build_eegnet():
    model = Sequential([
        Conv2D(32, (1, 64), padding='same', kernel_regularizer=l2(0.0005), input_shape=input_shape),
        BatchNormalization(),
        DepthwiseConv2D((19, 1), use_bias=False, depth_multiplier=2, padding='valid',
                        depthwise_regularizer=l2(0.0005)),
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

# ===========================
# Cargar y preprocesar .mat
# ===========================
def preprocess_file(mat_file):
    mat_data = scipy.io.loadmat(mat_file)
    key = list(mat_data.keys())[-1]
    data = mat_data[key].T

    raw = mne.io.RawArray(
        data,
        mne.create_info(ch_names=[f"Ch{i}" for i in range(data.shape[0])], sfreq=tsfreq, ch_types=["eeg"] * data.shape[0])
    )
    raw.filter(l_freq=0.5, h_freq=60, fir_design="firwin", verbose=False)

    ica = mne.preprocessing.ICA(n_components=19, random_state=97, max_iter=800)
    ica.fit(raw)
    ica.exclude = [0, 1]
    raw = ica.apply(raw)

    eeg_filtered_data = raw.get_data()
    num_samples = eeg_filtered_data.shape[1]
    num_windows = num_samples // window_size

    if num_windows == 0:
        return None  # muy corto

    eeg_windows = eeg_filtered_data[:, :num_windows * window_size].reshape(data.shape[0], num_windows, window_size)
    eeg_windows = np.transpose(eeg_windows, (1, 0, 2))
    eeg_windows = (eeg_windows - np.mean(eeg_windows)) / np.std(eeg_windows)

    return eeg_windows[..., np.newaxis]  # Añadir canal

# ===========================
# Streamlit
# ===========================
st.title("Clasificador EEG TDAH")
st.markdown("Sube un archivo `.mat` con una señal preformateada para obtener una predicción.")

uploaded_file = st.file_uploader("📂 Subir archivo `.mat`", type=["mat"])

if uploaded_file:
    x = preprocess_file(uploaded_file)
    if x is None:
        st.error("⚠️ La señal es demasiado corta para analizar.")
    else:
        model = build_eegnet()
        model.load_weights("modelo_eegnet.weights.h5")
        y_pred = np.argmax(model.predict(x), axis=1)
        clasificacion = int(np.round(np.mean(y_pred)))  # media de predicciones

        if clasificacion == 1:
            st.success("🧠 Clasificación: **TDAH**")
        else:
            st.success("🧠 Clasificación: **Control sano**")