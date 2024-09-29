# script v4.py
import sys
import os
import pyaudio
import numpy as np
import pygame
import tkinter as tk
from tkinter import ttk
from scipy.signal import butter, lfilter
import math
from screeninfo import get_monitors
import time
import json  # For JSON handling

# Configuration Parameters
SAMPLE_RATE = 36669          # 36,669 Hz
CHUNK = 666                  # Number of samples per frame
FORMAT = pyaudio.paInt16     # 16-bit resolution
CHANNELS = 1                 # Mono audio
FPS = 60                     # Frames per second

# VGain Configuration
VGAIN_DEFAULT = 96.0         # Default virtual gain
VGAIN_MIN = 0.0              # Minimum VGain
VGAIN_MAX = 960.0            # Maximum VGain

# VEQ Configuration
FILTER_TYPE = 'None'         # Options: 'None', 'Lowpass', 'Highpass'
CUTOFF_FREQ = 5000           # Cutoff frequency in Hz
FILTER_ORDER = 5             # Filter order

# Echo Configuration
ECHO_ENABLED = False
ECHO_DELAY_TIME = 1.0        # Delay time in seconds (0.1 to 30)
ECHO_DELAY_DRY_WET = 0.5     # 0.0 = all dry, 1.0 = all wet
ECHO_DELAY_FEEDBACK = 0.5    # Feedback amount (0.0 to 10.0)
ECHO_DELAY_GAIN = 1.0        # Gain for delayed signal (0.0 to 24.0 dB)

ECHO_REVERB_TIME = 2.0       # Reverb time in seconds (0.1 to 30)
ECHO_REVERB_DRY_WET = 0.5    # 0.0 = all dry, 1.0 = all wet
ECHO_REVERB_GAIN = 1.0       # Gain for reverberated signal (0.0 to 24.0 dB)

# VOSC Configuration
VOSC_DEFAULT_FREQ = 60        # Default frequency for VOSC (Hz)
VOSC_DEFAULT_AMP = 40         # Default amplitude for VOSC (silent initially)
VOSC_MAX_FREQ = 40000          # Maximum VOSC frequency
VOSC_MAX_AMP = 96.0           # Maximum VOSC amplitude

# List of filter types
FILTER_TYPES = ['None', 'Lowpass', 'Highpass']
current_filter_index = FILTER_TYPES.index(FILTER_TYPE)

# Preset storage
presets = {}

# JSON file for storing presets
PRESETS_FILE = 'presets.json'

class VOSC:
    def __init__(self, frequency=0.0, amplitude=0.0, sample_rate=SAMPLE_RATE, chunk_size=CHUNK):
        self.frequency = frequency
        self.amplitude = amplitude
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.is_playing = False
        self.phase = 0.0

    def toggle_play(self):
        self.is_playing = not self.is_playing
        state = "Playing" if self.is_playing else "Stopped"
        print(f"VOSC {state}")

    def set_frequency(self, freq):
        self.frequency = freq
        print(f"VOSC Frequency set to: {self.frequency} Hz")

    def set_amplitude(self, amp):
        self.amplitude = amp
        print(f"VOSC Amplitude set to: {self.amplitude}")

    def generate_wave(self):
        if not self.is_playing or self.amplitude == 0.0:
            return np.zeros(self.chunk_size, dtype=np.float32)
        phase_increment = 2 * math.pi * self.frequency / self.sample_rate
        phases = self.phase + phase_increment * np.arange(self.chunk_size)
        wave = self.amplitude * np.sin(phases)
        self.phase = (phases[-1] + phase_increment) % (2 * math.pi)
        return wave

class Echo:
    def __init__(self, delay_time, delay_dry_wet, delay_feedback, delay_gain,
                 reverb_time, reverb_dry_wet, reverb_gain, sample_rate=SAMPLE_RATE, chunk_size=CHUNK):
        # Delay Parameters
        self.delay_time = delay_time
        self.delay_dry_wet = delay_dry_wet
        self.delay_feedback = delay_feedback
        self.delay_gain = delay_gain
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.delay_buffer_size = int(self.delay_time * self.sample_rate)
        self.delay_buffer = np.zeros(self.delay_buffer_size, dtype=np.float32)
        self.delay_index = 0

        # Reverb Parameters
        self.reverb_time = reverb_time
        self.reverb_dry_wet = reverb_dry_wet
        self.reverb_gain = reverb_gain
        self.reverb_buffer_size = int(self.reverb_time * self.sample_rate)
        self.reverb_buffer = np.zeros(self.reverb_buffer_size, dtype=np.float32)
        self.reverb_index = 0

    def update_parameters(self, delay_time=None, delay_dry_wet=None, delay_feedback=None, delay_gain=None,
                         reverb_time=None, reverb_dry_wet=None, reverb_gain=None):
        if delay_time is not None and delay_time != self.delay_time:
            self.delay_time = delay_time
            self.delay_buffer_size = int(self.delay_time * self.sample_rate)
            self.delay_buffer = np.zeros(self.delay_buffer_size, dtype=np.float32)
            self.delay_index = 0
        if delay_dry_wet is not None:
            self.delay_dry_wet = delay_dry_wet
        if delay_feedback is not None:
            self.delay_feedback = delay_feedback
        if delay_gain is not None:
            self.delay_gain = delay_gain

        if reverb_time is not None and reverb_time != self.reverb_time:
            self.reverb_time = reverb_time
            self.reverb_buffer_size = int(self.reverb_time * self.sample_rate)
            self.reverb_buffer = np.zeros(self.reverb_buffer_size, dtype=np.float32)
            self.reverb_index = 0
        if reverb_dry_wet is not None:
            self.reverb_dry_wet = reverb_dry_wet
        if reverb_gain is not None:
            self.reverb_gain = reverb_gain

    def process(self, samples):
        # Process Delay
        delayed_samples = np.zeros_like(samples)
        for i in range(len(samples)):
            delayed_samples[i] = self.delay_buffer[self.delay_index]
            # Update delay buffer with current sample + feedback
            self.delay_buffer[self.delay_index] = samples[i] * self.delay_gain + delayed_samples[i] * self.delay_feedback
            self.delay_index = (self.delay_index + 1) % self.delay_buffer_size

        # Mix dry and wet for delay
        delay_output = (1 - self.delay_dry_wet) * samples + self.delay_dry_wet * delayed_samples

        # Process Reverb
        reverbed_samples = np.zeros_like(samples)
        for i in range(len(samples)):
            reverbed_samples[i] = self.reverb_buffer[self.reverb_index]
            # Update reverb buffer with current delay_output + reverberated signal
            self.reverb_buffer[self.reverb_index] = delay_output[i] * self.reverb_gain + reverbed_samples[i] * self.delay_feedback
            self.reverb_index = (self.reverb_index + 1) % self.reverb_buffer_size

        # Mix dry and wet for reverb
        reverb_output = (1 - self.reverb_dry_wet) * delay_output + self.reverb_dry_wet * reverbed_samples

        return reverb_output

class SimpleSelectionDialog:
    def __init__(self, root):
        self.root = root
        self.root.title("Start Visualizer")
        self.root.geometry("300x150")
        self.root.resizable(False, False)
        self.selected = False
        self.offset_x = 0
        self.offset_y = 0
        self.root.bind('<Button-1>', self.click_window)
        self.root.bind('<B1-Motion>', self.drag_window)
        label = ttk.Label(root, text="Press 'Select' to start", font=("Arial", 12))
        label.pack(pady=20)
        select_button = ttk.Button(root, text="Select", command=self.on_select)
        select_button.pack(pady=10)

    def click_window(self, event):
        self.offset_x = event.x
        self.offset_y = event.y

    def drag_window(self, event):
        x = event.x_root - self.offset_x
        y = event.y_root - self.offset_y
        self.root.geometry(f"+{x}+{y}")

    def on_select(self):
        self.selected = True
        self.root.destroy()

    def is_selected(self):
        return self.selected

def show_simple_selection_dialog():
    root = tk.Tk()
    app = SimpleSelectionDialog(root)
    root.mainloop()
    return app.is_selected()

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    if normal_cutoff >= 1:
        normal_cutoff = 0.99
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    if normal_cutoff <= 0:
        normal_cutoff = 0.01
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def apply_filter(data, b, a):
    return lfilter(b, a, data)

def initialize_pyaudio():
    p = pyaudio.PyAudio()
    try:
        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=SAMPLE_RATE,
                        input=True,
                        frames_per_buffer=CHUNK)
        return p, stream
    except Exception as e:
        print(f"Error initializing PyAudio: {e}")
        sys.exit(-1)

def initialize_pygame():
    pygame.init()
    monitors = get_monitors()
    if len(monitors) > 1:
        target_monitor = monitors[1]
    else:
        target_monitor = monitors[0]
    window_width, window_height = target_monitor.width, target_monitor.height
    window_x, window_y = target_monitor.x, target_monitor.y
    os.environ['SDL_VIDEO_WINDOW_POS'] = f"{window_x},{window_y}"
    flags = pygame.NOFRAME
    screen = pygame.display.set_mode((window_width, window_height), flags)
    pygame.display.set_caption("Real-Time Audio Visualizer")
    clock = pygame.time.Clock()
    pygame.event.set_grab(False)
    pygame.mouse.set_visible(True)
    return screen, clock, window_width, window_height

def sample_to_color(sample, vgain=1.0, max_val=32767):
    adjusted_sample = sample * vgain
    adjusted_sample = max(-max_val, min(adjusted_sample, max_val))
    normalized = (adjusted_sample + max_val) / (2 * max_val)
    red = min(int(normalized * 255), 255)
    blue = min(int((1 - normalized) * 255), 255)
    green = min(int(abs(normalized - 0.5) * 510), 255)
    return (red, green, blue)

# Preset encoding functions
def encode_preset(vgain, cutoff_freq, filter_type, vosc_freq, vosc_amp, vosc_playing,
                 echo_enabled, echo_delay_time, echo_delay_dry_wet, echo_delay_feedback, echo_delay_gain,
                 echo_reverb_time, echo_reverb_dry_wet, echo_reverb_gain):
    return {
        'vgain': vgain,
        'cutoff_freq': cutoff_freq,
        'filter_type': filter_type,
        'vosc_freq': vosc_freq,
        'vosc_amp': vosc_amp,
        'vosc_playing': vosc_playing,
        'echo_enabled': echo_enabled,
        'echo_delay_time': echo_delay_time,
        'echo_delay_dry_wet': echo_delay_dry_wet,
        'echo_delay_feedback': echo_delay_feedback,
        'echo_delay_gain': echo_delay_gain,
        'echo_reverb_time': echo_reverb_time,
        'echo_reverb_dry_wet': echo_reverb_dry_wet,
        'echo_reverb_gain': echo_reverb_gain
    }

def decode_preset(preset_dict):
    return (
        preset_dict['vgain'],
        preset_dict['cutoff_freq'],
        preset_dict['filter_type'],
        preset_dict['vosc_freq'],
        preset_dict['vosc_amp'],
        preset_dict['vosc_playing'],
        preset_dict['echo_enabled'],
        preset_dict['echo_delay_time'],
        preset_dict['echo_delay_dry_wet'],
        preset_dict['echo_delay_feedback'],
        preset_dict['echo_delay_gain'],
        preset_dict['echo_reverb_time'],
        preset_dict['echo_reverb_dry_wet'],
        preset_dict['echo_reverb_gain']
    )

def apply_preset(preset_dict, vosc, echo):
    global vgain, CUTOFF_FREQ, FILTER_TYPE, filter_b, filter_a
    (vgain, CUTOFF_FREQ, FILTER_TYPE, vosc_freq, vosc_amp, vosc_playing,
     echo_enabled, echo_delay_time, echo_delay_dry_wet, echo_delay_feedback, echo_delay_gain,
     echo_reverb_time, echo_reverb_dry_wet, echo_reverb_gain) = decode_preset(preset_dict)
    
    if FILTER_TYPE == 'Lowpass':
        filter_b, filter_a = butter_lowpass(CUTOFF_FREQ, SAMPLE_RATE, FILTER_ORDER)
    elif FILTER_TYPE == 'Highpass':
        filter_b, filter_a = butter_highpass(CUTOFF_FREQ, SAMPLE_RATE, FILTER_ORDER)
    else:
        filter_b, filter_a = None, None

    vosc.set_frequency(vosc_freq)
    vosc.set_amplitude(vosc_amp)
    vosc.is_playing = vosc_playing

    # Apply Echo settings
    global ECHO_ENABLED, ECHO_DELAY_TIME, ECHO_DELAY_DRY_WET, ECHO_DELAY_FEEDBACK, ECHO_DELAY_GAIN
    global ECHO_REVERB_TIME, ECHO_REVERB_DRY_WET, ECHO_REVERB_GAIN

    ECHO_ENABLED = echo_enabled
    ECHO_DELAY_TIME = echo_delay_time
    ECHO_DELAY_DRY_WET = echo_delay_dry_wet
    ECHO_DELAY_FEEDBACK = echo_delay_feedback
    ECHO_DELAY_GAIN = echo_delay_gain
    ECHO_REVERB_TIME = echo_reverb_time
    ECHO_REVERB_DRY_WET = echo_reverb_dry_wet
    ECHO_REVERB_GAIN = echo_reverb_gain

    echo.update_parameters(
        delay_time=ECHO_DELAY_TIME,
        delay_dry_wet=ECHO_DELAY_DRY_WET,
        delay_feedback=ECHO_DELAY_FEEDBACK,
        delay_gain=ECHO_DELAY_GAIN,
        reverb_time=ECHO_REVERB_TIME,
        reverb_dry_wet=ECHO_REVERB_DRY_WET,
        reverb_gain=ECHO_REVERB_GAIN
    )

    print(f"Preset applied: VGain={vgain:.2f}, Cutoff={CUTOFF_FREQ:.0f} Hz, Filter={FILTER_TYPE}, "
          f"VOSC Freq={vosc_freq:.2f} Hz, VOSC Amp={vosc_amp:.2f}, VOSC Playing={vosc_playing}, "
          f"Echo Enabled={ECHO_ENABLED}, Delay Time={ECHO_DELAY_TIME:.2f} sec, "
          f"Delay Dry/Wet={ECHO_DELAY_DRY_WET:.2f}, Delay Feedback={ECHO_DELAY_FEEDBACK:.2f}, "
          f"Delay Gain={ECHO_DELAY_GAIN:.2f} dB, Reverb Time={ECHO_REVERB_TIME:.2f} sec, "
          f"Reverb Dry/Wet={ECHO_REVERB_DRY_WET:.2f}, Reverb Gain={ECHO_REVERB_GAIN:.2f} dB")

def load_presets():
    global presets
    try:
        with open(PRESETS_FILE, 'r') as f:
            presets = json.load(f)
        print(f"Loaded {len(presets)} presets from {PRESETS_FILE}")
    except FileNotFoundError:
        print(f"Presets file {PRESETS_FILE} not found. Starting with empty presets.")
    except json.JSONDecodeError:
        print(f"Error decoding {PRESETS_FILE}. Starting with empty presets.")

def save_presets():
    with open(PRESETS_FILE, 'w') as f:
        json.dump(presets, f, indent=4)
    print(f"Saved {len(presets)} presets to {PRESETS_FILE}")

def handle_text_input(screen, font, prompt, max_length, show_info, render_frame):
    input_text = ""
    input_active = True
    while input_active:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return None
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN:
                    input_active = False
                elif event.key == pygame.K_BACKSPACE:
                    input_text = input_text[:-1]
                elif len(input_text) < max_length:
                    if event.unicode.isalnum():
                        input_text += event.unicode.upper()

        render_frame()

        if show_info:
            prompt_surface = font.render(prompt, True, (255, 255, 255))
            screen.blit(prompt_surface, (10, 10))
            input_surface = font.render(input_text, True, (255, 255, 255))
            screen.blit(input_surface, (10, 50))

        pygame.display.flip()

    return input_text

# Helper function for logarithmic adjustment
def adjust_logarithmic(value, direction, min_value, max_value, factor=1.1):
    """
    Adjusts the value logarithmically based on the direction.

    :param value: Current value of the parameter.
    :param direction: 'up' to increase, 'down' to decrease.
    :param min_value: Minimum allowable value.
    :param max_value: Maximum allowable value.
    :param factor: Multiplicative factor for adjustment.
    :return: Adjusted value.
    """
    if direction == 'up':
        new_value = value * factor
        return min(new_value, max_value)
    elif direction == 'down':
        new_value = value / factor
        return max(new_value, min_value)
    return value

def main_visualizer():
    global FILTER_TYPE, CUTOFF_FREQ, FILTER_TYPES, current_filter_index
    global VGAIN_MIN, VGAIN_MAX, vgain, filter_b, filter_a, presets
    global ECHO_ENABLED, ECHO_DELAY_TIME, ECHO_DELAY_DRY_WET, ECHO_DELAY_FEEDBACK, ECHO_DELAY_GAIN
    global ECHO_REVERB_TIME, ECHO_REVERB_DRY_WET, ECHO_REVERB_GAIN

    p, stream = initialize_pyaudio()
    screen, clock, window_width, window_height = initialize_pygame()
    vgain = VGAIN_DEFAULT
    filter_b, filter_a = None, None
    if FILTER_TYPE == 'Lowpass':
        filter_b, filter_a = butter_lowpass(CUTOFF_FREQ, SAMPLE_RATE, FILTER_ORDER)
    elif FILTER_TYPE == 'Highpass':
        filter_b, filter_a = butter_highpass(CUTOFF_FREQ, SAMPLE_RATE, FILTER_ORDER)

    # Initialize Echo
    echo = Echo(
        delay_time=ECHO_DELAY_TIME,
        delay_dry_wet=ECHO_DELAY_DRY_WET,
        delay_feedback=ECHO_DELAY_FEEDBACK,
        delay_gain=ECHO_DELAY_GAIN,
        reverb_time=ECHO_REVERB_TIME,
        reverb_dry_wet=ECHO_REVERB_DRY_WET,
        reverb_gain=ECHO_REVERB_GAIN,
        sample_rate=SAMPLE_RATE,
        chunk_size=CHUNK
    )

    pygame.font.init()
    font_size = 24
    font = pygame.font.SysFont("Arial", font_size)
    show_info = False
    effects = ['VGain', 'VEQ', 'Echo', 'VOSC']
    selected_effect_index = 0
    selected_effect = effects[selected_effect_index]
    effect_parameters = {
        'VGain': ['VGAIN'],
        'VEQ': ['CUTOFF_FREQ', 'FILTER_TYPE'],
        'Echo': ['ECHO_ENABLED', 'DELAY_TIME', 'DELAY_DRY_WET', 'DELAY_FEEDBACK', 'DELAY_GAIN',
                 'REVERB_TIME', 'REVERB_DRY_WET', 'REVERB_GAIN'],
        'VOSC': ['VOSC_FREQ', 'VOSC_AMP', 'VOSC_PLAY']
    }
    current_param_index = 0
    adjust_delay = 0.1
    last_adjust_time = time.time()
    vosc = VOSC(VOSC_DEFAULT_FREQ, VOSC_DEFAULT_AMP)
    input_mode = None

    # Load presets at startup
    load_presets()

    def render_frame():
        global CHUNK, vgain, FILTER_TYPE, filter_b, filter_a, ECHO_ENABLED
        global CUTOFF_FREQ, ECHO_DELAY_TIME, ECHO_DELAY_DRY_WET, ECHO_DELAY_FEEDBACK, ECHO_DELAY_GAIN
        global ECHO_REVERB_TIME, ECHO_REVERB_DRY_WET, ECHO_REVERB_GAIN

        # Read audio data
        try:
            data = stream.read(CHUNK, exception_on_overflow=False)
        except Exception as e:
            print(f"Error reading audio stream: {e}")
            return
        samples = np.frombuffer(data, dtype=np.int16).astype(np.float32)

        # Generate VOSC wave and mix with samples
        vosc_wave = vosc.generate_wave()
        samples += vosc_wave * 32767  # Assuming max amplitude scaling
        samples *= vgain

        # Apply VEQ filter if enabled
        if FILTER_TYPE != 'None' and filter_b is not None and filter_a is not None:
            samples = apply_filter(samples, filter_b, filter_a)

        # Apply master gain is already done via vgain

        # Process Echo send
        if ECHO_ENABLED:
            vsend = echo.process(samples)
            # Apply Echo Gain (convert dB to linear)
            vsend_gain_linear = 10 ** (ECHO_DELAY_GAIN / 20)
            vsend *= vsend_gain_linear
        else:
            vsend = np.zeros_like(samples)

        # Master source (dry signal)
        vsource = samples

        # Mix dry and Echo send
        master_output = vsource + vsend

        # Clip samples to prevent overflow
        master_output = np.clip(master_output, -32767, 32767)
        master_output = master_output.astype(np.int16)
        if len(master_output) < CHUNK:
            master_output = np.pad(master_output, (0, CHUNK - len(master_output)), 'constant')

        # Convert samples to colors for visualization
        colors = [sample_to_color(sample, vgain) for sample in master_output]

        # Render visualization
        screen.fill((0, 0, 0))
        band_width = window_width / CHUNK
        for i, color in enumerate(colors):
            x = i * band_width
            rect = pygame.Rect(x, 0, band_width, window_height)
            pygame.draw.rect(screen, color, rect)

        # Display information if enabled
        if show_info:
            info_lines = []
            info_lines.append(f"Selected Effect: {selected_effect}")
            params = effect_parameters[selected_effect]
            for param in params:
                if selected_effect == 'VGain' and param == 'VGAIN':
                    info_lines.append(f"VGain: {vgain:.1f}")
                elif selected_effect == 'VEQ' and param == 'CUTOFF_FREQ':
                    info_lines.append(f"CUTOFF_FREQ: {CUTOFF_FREQ:.0f} Hz")
                elif selected_effect == 'VEQ' and param == 'FILTER_TYPE':
                    info_lines.append(f"FILTER_TYPE: {FILTER_TYPE}")
                elif selected_effect == 'Echo':
                    if param == 'ECHO_ENABLED':
                        info_lines.append(f"Echo Enabled: {ECHO_ENABLED}")
                    elif param == 'DELAY_TIME':
                        info_lines.append(f"Delay Time: {ECHO_DELAY_TIME:.2f} sec")
                    elif param == 'DELAY_DRY_WET':
                        info_lines.append(f"Delay Dry/Wet: {ECHO_DELAY_DRY_WET:.2f}")
                    elif param == 'DELAY_FEEDBACK':
                        info_lines.append(f"Delay Feedback: {ECHO_DELAY_FEEDBACK:.2f}")
                    elif param == 'DELAY_GAIN':
                        info_lines.append(f"Delay Gain: {ECHO_DELAY_GAIN:.2f} dB")
                    elif param == 'REVERB_TIME':
                        info_lines.append(f"Reverb Time: {ECHO_REVERB_TIME:.2f} sec")
                    elif param == 'REVERB_DRY_WET':
                        info_lines.append(f"Reverb Dry/Wet: {ECHO_REVERB_DRY_WET:.2f}")
                    elif param == 'REVERB_GAIN':
                        info_lines.append(f"Reverb Gain: {ECHO_REVERB_GAIN:.2f} dB")
                elif selected_effect == 'VOSC' and param == 'VOSC_FREQ':
                    info_lines.append(f"VOSC_FREQ: {vosc.frequency:.2f} Hz")
                elif selected_effect == 'VOSC' and param == 'VOSC_AMP':
                    info_lines.append(f"VOSC_AMP: {vosc.amplitude:.2f}")
                elif selected_effect == 'VOSC' and param == 'VOSC_PLAY':
                    info_lines.append(f"VOSC_PLAY: {'Playing' if vosc.is_playing else 'Stopped'}")
            for idx, line in enumerate(info_lines):
                text_surface = font.render(line, True, (255, 255, 255))
                screen.blit(text_surface, (10, 10 + idx * (font_size + 5)))

    while True:
        current_time = time.time()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                save_presets()  # Save presets before exiting
                raise KeyboardInterrupt
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_F1:
                    show_info = not show_info
                elif event.key == pygame.K_h:
                    input_mode = "load_preset"
                elif event.key == pygame.K_s:
                    input_mode = "save_preset"
                elif event.key == pygame.K_LEFT:
                    selected_effect_index = (selected_effect_index - 1) % len(effects)
                    selected_effect = effects[selected_effect_index]
                    current_param_index = 0
                elif event.key == pygame.K_RIGHT:
                    selected_effect_index = (selected_effect_index + 1) % len(effects)
                    selected_effect = effects[selected_effect_index]
                    current_param_index = 0
                elif event.key == pygame.K_TAB:
                    params = effect_parameters[selected_effect]
                    if params:
                        current_param_index = (current_param_index + 1) % len(params)
                elif event.key == pygame.K_p:
                    vosc.toggle_play()
                elif event.key == pygame.K_f:
                    vosc.set_frequency(vosc.frequency + 100)
                elif event.key == pygame.K_v:
                    vosc.set_frequency(vosc.frequency - 100)
                elif event.key == pygame.K_a:
                    vosc.set_amplitude(min(vosc.amplitude + 0.1, VOSC_MAX_AMP))
                elif event.key == pygame.K_d:
                    vosc.set_amplitude(max(vosc.amplitude - 0.1, 0.0))
                elif event.key == pygame.K_e:
                    ECHO_ENABLED = not ECHO_ENABLED
                    print(f"Echo Enabled: {ECHO_ENABLED}")

        if input_mode == "save_preset":
            preset_name = handle_text_input(screen, font, "Enter a name for the preset:", 20, show_info, render_frame)
            if preset_name:
                preset_dict = encode_preset(
                    vgain, CUTOFF_FREQ, FILTER_TYPE, vosc.frequency, vosc.amplitude, vosc.is_playing,
                    ECHO_ENABLED, ECHO_DELAY_TIME, ECHO_DELAY_DRY_WET, ECHO_DELAY_FEEDBACK, ECHO_DELAY_GAIN,
                    ECHO_REVERB_TIME, ECHO_REVERB_DRY_WET, ECHO_REVERB_GAIN
                )
                presets[preset_name] = preset_dict
                print(f"Preset '{preset_name}' saved")
                save_presets()  # Save presets to file immediately after adding a new one
            input_mode = None
        elif input_mode == "load_preset":
            preset_name = handle_text_input(screen, font, "Enter the name of the preset to load:", 20, show_info, render_frame)
            if preset_name and preset_name in presets:
                apply_preset(presets[preset_name], vosc, echo)
            elif preset_name:
                print(f"Preset '{preset_name}' not found.")
            input_mode = None

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP] or keys[pygame.K_DOWN]:
            if current_time - last_adjust_time >= adjust_delay:
                params = effect_parameters[selected_effect]
                if params:
                    current_param = params[current_param_index]
                    if selected_effect == 'VGain':
                        if current_param == 'VGAIN':
                            if keys[pygame.K_UP]:
                                vgain = adjust_logarithmic(vgain, 'up', VGAIN_MIN, VGAIN_MAX)
                            elif keys[pygame.K_DOWN]:
                                vgain = adjust_logarithmic(vgain, 'down', VGAIN_MIN, VGAIN_MAX)
                    elif selected_effect == 'VEQ':
                        if current_param == 'CUTOFF_FREQ':
                            if keys[pygame.K_UP]:
                                CUTOFF_FREQ = adjust_logarithmic(CUTOFF_FREQ, 'up', 100.0, SAMPLE_RATE / 2 - 100.0)
                                if FILTER_TYPE == 'Lowpass':
                                    filter_b, filter_a = butter_lowpass(CUTOFF_FREQ, SAMPLE_RATE, FILTER_ORDER)
                                elif FILTER_TYPE == 'Highpass':
                                    filter_b, filter_a = butter_highpass(CUTOFF_FREQ, SAMPLE_RATE, FILTER_ORDER)
                            elif keys[pygame.K_DOWN]:
                                CUTOFF_FREQ = adjust_logarithmic(CUTOFF_FREQ, 'down', 100.0, SAMPLE_RATE / 2 - 100.0)
                                if FILTER_TYPE == 'Lowpass':
                                    filter_b, filter_a = butter_lowpass(CUTOFF_FREQ, SAMPLE_RATE, FILTER_ORDER)
                                elif FILTER_TYPE == 'Highpass':
                                    filter_b, filter_a = butter_highpass(CUTOFF_FREQ, SAMPLE_RATE, FILTER_ORDER)
                        elif current_param == 'FILTER_TYPE':
                            if keys[pygame.K_UP]:
                                current_filter_index = (current_filter_index + 1) % len(FILTER_TYPES)
                                FILTER_TYPE = FILTER_TYPES[current_filter_index]
                                if FILTER_TYPE == 'Lowpass':
                                    filter_b, filter_a = butter_lowpass(CUTOFF_FREQ, SAMPLE_RATE, FILTER_ORDER)
                                elif FILTER_TYPE == 'Highpass':
                                    filter_b, filter_a = butter_highpass(CUTOFF_FREQ, SAMPLE_RATE, FILTER_ORDER)
                                else:
                                    filter_b, filter_a = None, None
                            elif keys[pygame.K_DOWN]:
                                current_filter_index = (current_filter_index - 1) % len(FILTER_TYPES)
                                FILTER_TYPE = FILTER_TYPES[current_filter_index]
                                if FILTER_TYPE == 'Lowpass':
                                    filter_b, filter_a = butter_lowpass(CUTOFF_FREQ, SAMPLE_RATE, FILTER_ORDER)
                                elif FILTER_TYPE == 'Highpass':
                                    filter_b, filter_a = butter_highpass(CUTOFF_FREQ, SAMPLE_RATE, FILTER_ORDER)
                                else:
                                    filter_b, filter_a = None, None
                    elif selected_effect == 'Echo':
                        if current_param == 'ECHO_ENABLED':
                            if keys[pygame.K_UP]:
                                ECHO_ENABLED = True
                                print(f"Echo Enabled: {ECHO_ENABLED}")
                            elif keys[pygame.K_DOWN]:
                                ECHO_ENABLED = False
                                print(f"Echo Enabled: {ECHO_ENABLED}")
                        elif current_param == 'DELAY_TIME':
                            if keys[pygame.K_UP]:
                                ECHO_DELAY_TIME = adjust_logarithmic(ECHO_DELAY_TIME, 'up', 0.1, 30.0)
                                echo.update_parameters(delay_time=ECHO_DELAY_TIME)
                            elif keys[pygame.K_DOWN]:
                                ECHO_DELAY_TIME = adjust_logarithmic(ECHO_DELAY_TIME, 'down', 0.0, 30.0)
                                echo.update_parameters(delay_time=ECHO_DELAY_TIME)
                        elif current_param == 'DELAY_DRY_WET':
                            if keys[pygame.K_UP]:
                                ECHO_DELAY_DRY_WET = min(ECHO_DELAY_DRY_WET + 0.05, 1.0)
                                echo.update_parameters(delay_dry_wet=ECHO_DELAY_DRY_WET)
                            elif keys[pygame.K_DOWN]:
                                ECHO_DELAY_DRY_WET = max(ECHO_DELAY_DRY_WET - 0.05, 0.0)
                                echo.update_parameters(delay_dry_wet=ECHO_DELAY_DRY_WET)
                        elif current_param == 'DELAY_FEEDBACK':
                            if keys[pygame.K_UP]:
                                ECHO_DELAY_FEEDBACK = min(ECHO_DELAY_FEEDBACK + 0.1, 10.0)
                                echo.update_parameters(delay_feedback=ECHO_DELAY_FEEDBACK)
                            elif keys[pygame.K_DOWN]:
                                ECHO_DELAY_FEEDBACK = max(ECHO_DELAY_FEEDBACK - 0.1, 0.0)
                                echo.update_parameters(delay_feedback=ECHO_DELAY_FEEDBACK)
                        elif current_param == 'DELAY_GAIN':
                            if keys[pygame.K_UP]:
                                ECHO_DELAY_GAIN = min(ECHO_DELAY_GAIN + 0.5, 24.0)  # Assuming dB scale
                                echo.update_parameters(delay_gain=ECHO_DELAY_GAIN)
                            elif keys[pygame.K_DOWN]:
                                ECHO_DELAY_GAIN = max(ECHO_DELAY_GAIN - 0.5, 0.0)
                                echo.update_parameters(delay_gain=ECHO_DELAY_GAIN)
                        elif current_param == 'REVERB_TIME':
                            if keys[pygame.K_UP]:
                                ECHO_REVERB_TIME = adjust_logarithmic(ECHO_REVERB_TIME, 'up', 0.1, 30.0)
                                echo.update_parameters(reverb_time=ECHO_REVERB_TIME)
                            elif keys[pygame.K_DOWN]:
                                ECHO_REVERB_TIME = adjust_logarithmic(ECHO_REVERB_TIME, 'down', 0.1, 30.0)
                                echo.update_parameters(reverb_time=ECHO_REVERB_TIME)
                        elif current_param == 'REVERB_DRY_WET':
                            if keys[pygame.K_UP]:
                                ECHO_REVERB_DRY_WET = min(ECHO_REVERB_DRY_WET + 0.05, 1.0)
                                echo.update_parameters(reverb_dry_wet=ECHO_REVERB_DRY_WET)
                            elif keys[pygame.K_DOWN]:
                                ECHO_REVERB_DRY_WET = max(ECHO_REVERB_DRY_WET - 0.05, 0.0)
                                echo.update_parameters(reverb_dry_wet=ECHO_REVERB_DRY_WET)
                        elif current_param == 'REVERB_GAIN':
                            if keys[pygame.K_UP]:
                                ECHO_REVERB_GAIN = min(ECHO_REVERB_GAIN + 0.5, 24.0)  # Assuming dB scale
                                echo.update_parameters(reverb_gain=ECHO_REVERB_GAIN)
                            elif keys[pygame.K_DOWN]:
                                ECHO_REVERB_GAIN = max(ECHO_REVERB_GAIN - 0.5, 0.0)
                                echo.update_parameters(reverb_gain=ECHO_REVERB_GAIN)
                    elif selected_effect == 'VOSC':
                        if current_param == 'VOSC_FREQ':
                            if keys[pygame.K_UP]:
                                vosc.set_frequency(adjust_logarithmic(vosc.frequency, 'up', -20000.0, VOSC_MAX_FREQ))
                            elif keys[pygame.K_DOWN]:
                                vosc.set_frequency(adjust_logarithmic(vosc.frequency, 'down', -20000.0, VOSC_MAX_FREQ))
                        elif current_param == 'VOSC_AMP':
                            if keys[pygame.K_UP]:
                                vosc.set_amplitude(adjust_logarithmic(vosc.amplitude, 'up', 0.0, VOSC_MAX_AMP))
                            elif keys[pygame.K_DOWN]:
                                vosc.set_amplitude(adjust_logarithmic(vosc.amplitude, 'down', 0.0, VOSC_MAX_AMP))
                        elif current_param == 'VOSC_PLAY':
                            if keys[pygame.K_UP]:
                                vosc.toggle_play()
                            elif keys[pygame.K_DOWN]:
                                vosc.toggle_play()
                last_adjust_time = current_time

        render_frame()
        pygame.display.flip()
        clock.tick(FPS)

if __name__ == "__main__":
    selected = show_simple_selection_dialog()
    try:
        if selected:
            main_visualizer()
        else:
            sys.exit(0)
    except KeyboardInterrupt:
        pygame.quit()
        sys.exit()
    except Exception as e:
        pygame.quit()
        print(f"An error occurred: {e}")
        sys.exit(1)
