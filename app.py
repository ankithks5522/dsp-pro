
import io
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import librosa
import soundfile as sf

from scipy.signal import butter, sosfiltfilt, sosfreqz

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="DSP Audio Analyzer",
    page_icon="🎵",
    layout="wide"
)

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def make_wav_bytes(audio, sample_rate):
    """Convert floating-point audio into downloadable WAV bytes."""
    buffer = io.BytesIO()
    audio = np.asarray(audio, dtype=np.float32)

    # Keep audio within the valid PCM range
    audio = np.clip(audio, -1.0, 1.0)

    sf.write(
        buffer,
        audio,
        sample_rate,
        format="WAV",
        subtype="PCM_16"
    )
    return buffer.getvalue()


def plot_time_domain(audio, sample_rate, title):
    """Plot the waveform against time."""
    mono = np.mean(audio, axis=1) if audio.ndim == 2 else audio
    duration = len(mono) / sample_rate

    # Limit plotted points for long recordings
    step = max(1, len(mono) // 100000)
    indices = np.arange(0, len(mono), step)
    times = indices / sample_rate

    fig, ax = plt.subplots(figsize=(12, 3.5))
    ax.plot(times, mono[indices], linewidth=0.65)
    ax.set_title(title)
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Amplitude")
    ax.set_xlim(0, max(duration, 0.001))
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_fft(audio, sample_rate, title, max_frequency=None):
    """Plot the single-sided FFT magnitude spectrum."""
    mono = np.mean(audio, axis=1) if audio.ndim == 2 else audio

    n = len(mono)
    if n == 0:
        return None

    # Remove DC offset and apply a Hann window to reduce leakage
    signal = mono - np.mean(mono)
    window = np.hanning(n)

    spectrum = np.fft.rfft(signal * window)
    frequencies = np.fft.rfftfreq(n, d=1 / sample_rate)

    magnitude = np.abs(spectrum) / max(np.sum(window), 1e-12)
    magnitude_db = 20 * np.log10(magnitude + 1e-10)

    fig, ax = plt.subplots(figsize=(12, 3.5))
    ax.plot(frequencies, magnitude_db, linewidth=0.7)
    ax.set_title(title)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude (dB)")
    ax.set_xlim(0, min(max_frequency or sample_rate / 2,
                       sample_rate / 2))
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_fft_comparison(original, combined, sample_rate):
    """Compare original and combined audio FFT spectra."""
    def get_spectrum(audio):
        mono = np.mean(audio, axis=1) if audio.ndim == 2 else audio
        n = len(mono)
        signal = mono - np.mean(mono)
        window = np.hanning(n)

        spectrum = np.fft.rfft(signal * window)
        frequencies = np.fft.rfftfreq(n, d=1 / sample_rate)
        magnitude = np.abs(spectrum) / max(np.sum(window), 1e-12)
        magnitude_db = 20 * np.log10(magnitude + 1e-10)

        return frequencies, magnitude_db

    f_original, db_original = get_spectrum(original)
    f_combined, db_combined = get_spectrum(combined)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(
        f_original, db_original,
        label="Original Audio",
        linewidth=0.8
    )
    ax.plot(
        f_combined, db_combined,
        label="Combined Filtered Audio",
        linewidth=0.8,
        alpha=0.85
    )

    ax.set_title("FFT Comparison: Original vs Combined Audio")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude (dB)")
    ax.set_xlim(0, min(10000, sample_rate / 2))
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_filter_response(filter_sections, sample_rate, title):
    """Display a filter's magnitude frequency response."""
    frequencies, response = sosfreqz(
        filter_sections,
        worN=4096,
        fs=sample_rate
    )

    response_db = 20 * np.log10(np.abs(response) + 1e-10)

    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(frequencies, response_db, linewidth=0.8)
    ax.set_title(title)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Gain (dB)")
    ax.set_xlim(0, min(10000, sample_rate / 2))
    ax.set_ylim(-100, 5)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


# ============================================================
# CENTERED UPLOAD SCREEN
# ============================================================

if "audio_bytes" not in st.session_state:
    st.session_state.audio_bytes = None
    st.session_state.audio_name = None

if st.session_state.audio_bytes is None:

    st.markdown(
        """
        <div style="height: 12vh;"></div>
        <h1 style="text-align:center;">DSP Audio Analyzer</h1>
        <p style="text-align:center; font-size:18px;">
        Upload an audio file to explore its waveform, FFT,
        frequency filtering, and combined output.
        </p>
        """,
        unsafe_allow_html=True
    )

    _, center, _ = st.columns([1, 2, 1])

    with center:
        uploaded_file = st.file_uploader(
            "Choose your audio file",
            type=["mp3", "wav", "ogg", "flac"],
            help="Upload an MP3, WAV, OGG, or FLAC file."
        )

        if uploaded_file is not None:
            st.session_state.audio_bytes = uploaded_file.getvalue()
            st.session_state.audio_name = uploaded_file.name
            st.rerun()

    st.stop()


# ============================================================
# LOAD AUDIO
# ============================================================

st.title("DSP Audio Analyzer")
st.caption(f"File: {st.session_state.audio_name}")

if st.button("Upload another audio file"):
    st.session_state.audio_bytes = None
    st.session_state.audio_name = None
    st.rerun()

try:
    audio_data, sample_rate = librosa.load(
        io.BytesIO(st.session_state.audio_bytes),
        sr=None,
        mono=False
    )

    # librosa returns stereo as (channels, samples)
    if audio_data.ndim == 1:
        original_audio = audio_data[:, None]
    else:
        original_audio = audio_data.T

    original_audio = original_audio.astype(np.float32)

except Exception as error:
    st.error("Unable to decode this audio file.")
    st.exception(error)
    st.stop()

duration = len(original_audio) / sample_rate
nyquist = sample_rate / 2

st.write(
    f"**Sample rate:** {sample_rate} Hz  |  "
    f"**Duration:** {duration:.2f} seconds  |  "
    f"**Channels:** {original_audio.shape[1]}"
)

if sample_rate / 2 <= 4000:
    st.error(
        "The sample rate is too low for the configured four bands. "
        "Please use an audio file sampled above approximately 8.5 kHz."
    )
    st.stop()

# ============================================================
# FILTER ORDER SLIDER
# ============================================================

st.header("Filter Settings")

filter_order = st.slider(
    "Butterworth filter order",
    min_value=1,
    max_value=25,
    value=25,
    step=1,
    help=(
        "Increasing the order makes the transition between "
        "passband and stopband steeper. A band-pass filter built "
        "from an Nth-order prototype has transfer-function order 2N."
    )
)

st.write(f"Selected prototype order: **{filter_order}**")

upper_frequency = min(10000.0, 0.95 * nyquist)

bands = {
    "Low": (20.0, 250.0),
    "Low-Mid": (250.0, 1000.0),
    "High-Mid": (1000.0, 4000.0),
    "High": (4000.0, upper_frequency)
}

# ============================================================
# ORIGINAL AUDIO ANALYSIS
# ============================================================

st.divider()
st.header("1. Original Audio Analysis")

st.subheader("Original Audio Player")
st.audio(st.session_state.audio_bytes)

st.subheader("Original Time-Domain Analysis")

fig = plot_time_domain(
    original_audio,
    sample_rate,
    "Original Audio Waveform"
)
st.pyplot(fig)
plt.close(fig)

st.subheader("Original Frequency-Domain Analysis (FFT)")

fig = plot_fft(
    original_audio,
    sample_rate,
    "Original Audio FFT Magnitude Spectrum",
    max_frequency=min(10000, nyquist)
)
st.pyplot(fig)
plt.close(fig)

# ============================================================
# FILTER DESIGN AND AUDIO SEPARATION
# ============================================================

st.divider()
st.header("2. Frequency-Band Separation")

st.write(
    "Each band is produced using a Butterworth band-pass filter. "
    "The order slider above controls all four filters."
)

filtered_audio = {}
filter_sections = {}

try:
    for band_name, (low, high) in bands.items():

        sos = butter(
            filter_order,
            [low, high],
            btype="bandpass",
            fs=sample_rate,
            output="sos"
        )

        filtered = sosfiltfilt(
            sos,
            original_audio,
            axis=0
        )

        filter_sections[band_name] = sos
        filtered_audio[band_name] = filtered.astype(np.float32)

except Exception as error:
    st.error(
        "Filtering failed. The audio may be too short for the selected "
        "filter order. Try lowering the order."
    )
    st.exception(error)
    st.stop()

# ============================================================
# DISPLAY EACH FILTERED BAND
# ============================================================

for index, (band_name, (low, high)) in enumerate(bands.items(), start=1):

    st.divider()
    st.header(f"{index + 2}. {band_name} Band")

    st.write(f"**Passband:** {low:.0f} Hz – {high:.0f} Hz")

    st.subheader(f"{band_name} Audio Player")

    band_audio = filtered_audio[band_name]
    band_wav = make_wav_bytes(band_audio, sample_rate)

    st.audio(band_wav, format="audio/wav")

    st.download_button(
        label=f"Download {band_name} Band Audio",
        data=band_wav,
        file_name=f"{band_name.replace('-', '_')}_band.wav",
        mime="audio/wav",
        key=f"download_{band_name}"
    )

    st.subheader(f"{band_name} Time-Domain Analysis")

    fig = plot_time_domain(
        band_audio,
        sample_rate,
        f"{band_name} Band Waveform"
    )
    st.pyplot(fig)
    plt.close(fig)

    st.subheader(f"{band_name} Frequency-Domain Analysis (FFT)")

    fig = plot_fft(
        band_audio,
        sample_rate,
        f"{band_name} Band FFT Magnitude Spectrum",
        max_frequency=min(10000, nyquist)
    )
    st.pyplot(fig)
    plt.close(fig)

    st.subheader(f"{band_name} Filter Frequency Response")

    fig = plot_filter_response(
        filter_sections[band_name],
        sample_rate,
        f"{band_name} Butterworth Filter Response"
    )
    st.pyplot(fig)
    plt.close(fig)

# ============================================================
# AUTOMATIC COMBINATION OF ALL FOUR BANDS
# ============================================================

st.divider()
st.header("7. Combined Audio Analysis")

st.write(
    "The four filtered signals are added together at equal gain. "
    "No contribution sliders are used."
)

combined_audio = sum(filtered_audio.values())

# Preserve relative levels unless clipping would occur
combined_peak = np.max(np.abs(combined_audio))

if combined_peak > 1.0:
    combined_audio = combined_audio / combined_peak

combined_audio = combined_audio.astype(np.float32)
combined_wav = make_wav_bytes(combined_audio, sample_rate)

st.subheader("Combined Audio Player")
st.audio(combined_wav, format="audio/wav")

st.download_button(
    label="Download Combined Audio",
    data=combined_wav,
    file_name="Combined_Separated_Audio.wav",
    mime="audio/wav",
    type="primary"
)

st.subheader("Combined Audio Time-Domain Analysis")

fig = plot_time_domain(
    combined_audio,
    sample_rate,
    "Combined Audio Waveform"
)
st.pyplot(fig)
plt.close(fig)

st.subheader("Combined Audio Frequency-Domain Analysis (FFT)")

fig = plot_fft(
    combined_audio,
    sample_rate,
    "Combined Audio FFT Magnitude Spectrum",
    max_frequency=min(10000, nyquist)
)
st.pyplot(fig)
plt.close(fig)

st.subheader("Combined Audio Filter-Bank Frequency Response")

# Plot the responses of all four filters together
fig, ax = plt.subplots(figsize=(12, 4))

for band_name, sos in filter_sections.items():
    frequencies, response = sosfreqz(
        sos,
        worN=4096,
        fs=sample_rate
    )

    response_db = 20 * np.log10(np.abs(response) + 1e-10)
    ax.plot(frequencies, response_db, label=band_name)

ax.set_title("Frequency Responses of All Four Filters")
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("Gain (dB)")
ax.set_xlim(0, min(10000, nyquist))
ax.set_ylim(-100, 5)
ax.grid(True, alpha=0.3)
ax.legend()
fig.tight_layout()

st.pyplot(fig)
plt.close(fig)

st.subheader("FFT Comparison: Original vs Combined Audio")

fig = plot_fft_comparison(
    original_audio,
    combined_audio,
    sample_rate
)
st.pyplot(fig)
plt.close(fig)

# ============================================================
# FINAL NOTES
# ============================================================

st.divider()
st.subheader("About the Results")

st.markdown(
    """
    - **Time domain:** shows how signal amplitude changes over time.
    - **FFT:** shows the distribution of signal magnitude across frequencies.
    - **Filter response:** shows which frequencies each filter passes or attenuates.
    - **Combined audio:** is the sum of the four filtered signals.

    **Note:** Frequency bands do not perfectly isolate individual instruments.
    Several instruments can occupy the same frequency range. Filter transitions
    and the combined filter response can also cause differences from the original.
    """
)
