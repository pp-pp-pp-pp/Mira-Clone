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

# Configuration Parameters
SAMPLE_RATE = 39996        # 39,996 Hz
CHUNK = 666                # Number of samples per frame
FORMAT = pyaudio.paInt16   # 16-bit resolution
CHANNELS = 1               # Mono audio
FPS = 60                   # Frames per second

# VGain Configuration
VGAIN_DEFAULT = 1.0        # Default virtual gain
VGAIN_STEP = 0.1            # Increment step for VGain
VGAIN_MIN = 0.1             # Minimum VGain
VGAIN_MAX = 5.0             # Maximum VGain

# VEQ Configuration
FILTER_TYPE = 'None'        # Options: 'None', 'Lowpass', 'Highpass'
CUTOFF_FREQ = 5000          # Cutoff frequency in Hz
FILTER_ORDER = 5            # Filter order

# VDelay Configuration
DELAY_ENABLED = False
DELAY_TIME = 0.2            # Delay time in seconds
DELAY_FEEDBACK = 0.5        # Feedback coefficient

# VReverb Configuration
REVERB_ENABLED = False
REVERB_DECAY = 0.5          # Reverb decay factor

# VOSC Configuration
VOSC_DEFAULT_FREQ = 0.0   # Default frequency for VOSC (Hz)
VOSC_DEFAULT_AMP = 0.0      # Default amplitude for VOSC (silent initially)

class VOSC:
    def __init__(self, frequency=0.0, amplitude=0.0, sample_rate=39996, chunk_size=666):
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

class VDelay:
    def __init__(self, delay_time, feedback, sample_rate, chunk_size):
        self.delay_time = delay_time
        self.feedback = feedback
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.buffer_size = chunk_size * int(math.ceil(delay_time * sample_rate / chunk_size))
        self.buffer = np.zeros(self.buffer_size, dtype=np.float32)
        self.index = 0

    def update_parameters(self, delay_time=None, feedback=None):
        if delay_time is not None and delay_time != self.delay_time:
            self.delay_time = delay_time
            new_buffer_size = self.chunk_size * int(math.ceil(delay_time * self.sample_rate / self.chunk_size))
            if new_buffer_size != self.buffer_size:
                self.buffer_size = new_buffer_size
                self.buffer = np.zeros(self.buffer_size, dtype=np.float32)
                self.index = 0
        if feedback is not None:
            self.feedback = feedback

    def process(self, samples):
        try:
            delayed_samples = self.buffer[self.index:self.index + len(samples)]
            if len(delayed_samples) < len(samples):
                first_part = self.buffer[self.index:]
                second_part = self.buffer[:len(samples) - len(delayed_samples)]
                delayed_samples = np.concatenate((first_part, second_part))
            output = samples + self.feedback * delayed_samples
            self.buffer[self.index:self.index + len(samples)] = samples
            self.index = (self.index + len(samples)) % self.buffer_size
            return output
        except:
            return samples

class VReverb:
    def __init__(self, decay, sample_rate, chunk_size):
        self.decay = decay
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.buffer_size = chunk_size * int(math.ceil(0.1 * sample_rate / chunk_size))
        self.buffer = np.zeros(self.buffer_size, dtype=np.float32)
        self.index = 0

    def update_parameters(self, decay=None):
        if decay is not None:
            self.decay = decay

    def process(self, samples):
        try:
            reverberated = self.buffer[self.index:self.index + len(samples)]
            if len(reverberated) < len(samples):
                first_part = self.buffer[self.index:]
                second_part = self.buffer[:len(samples) - len(reverberated)]
                reverberated = np.concatenate((first_part, second_part))
            output = samples + self.decay * reverberated
            self.buffer[self.index:self.index + len(samples)] = output
            self.index = (self.index + len(samples)) % self.buffer_size
            return output
        except:
            return samples

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
    except:
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

def main_visualizer():
    global FILTER_TYPE, DELAY_ENABLED, REVERB_ENABLED
    global CUTOFF_FREQ, DELAY_TIME, DELAY_FEEDBACK, REVERB_DECAY
    p, stream = initialize_pyaudio()
    screen, clock, window_width, window_height = initialize_pygame()
    vgain = VGAIN_DEFAULT
    filter_b, filter_a = None, None
    if FILTER_TYPE == 'Lowpass':
        filter_b, filter_a = butter_lowpass(CUTOFF_FREQ, SAMPLE_RATE, FILTER_ORDER)
    elif FILTER_TYPE == 'Highpass':
        filter_b, filter_a = butter_highpass(CUTOFF_FREQ, SAMPLE_RATE, FILTER_ORDER)
    delay_effect = VDelay(DELAY_TIME, DELAY_FEEDBACK, SAMPLE_RATE, CHUNK) if DELAY_ENABLED else None
    reverb_effect = VReverb(REVERB_DECAY, SAMPLE_RATE, CHUNK) if REVERB_ENABLED else None
    pygame.font.init()
    font_size = 24
    font = pygame.font.SysFont("Arial", font_size)
    show_info = False
    effects = ['VGain', 'VEQ', 'VDelay', 'VReverb', 'VOSC']
    selected_effect_index = 0
    selected_effect = effects[selected_effect_index]
    effect_parameters = {
        'VGain': ['VGAIN'],
        'VEQ': ['CUTOFF_FREQ', 'FILTER_TYPE'],
        'VDelay': ['DELAY_TIME', 'DELAY_FEEDBACK'],
        'VReverb': ['REVERB_DECAY'],
        'VOSC': ['VOSC_FREQ', 'VOSC_AMP', 'VOSC_PLAY']
    }
    current_param_index = 0
    adjust_delay = 0.1
    last_adjust_time = time.time()
    vosc = VOSC(VOSC_DEFAULT_FREQ, VOSC_DEFAULT_AMP)
    while True:
        current_time = time.time()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                raise KeyboardInterrupt
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_F1:
                    show_info = not show_info
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
                elif event.key == pygame.K_ESCAPE:
                    raise KeyboardInterrupt
                elif event.key == pygame.K_p:
                    vosc.toggle_play()
                elif event.key == pygame.K_f:
                    vosc.set_frequency(vosc.frequency + 100)
                elif event.key == pygame.K_s:
                    vosc.set_frequency(vosc.frequency - 100)
                elif event.key == pygame.K_a:
                    vosc.set_amplitude(min(vosc.amplitude + 0.1, 1.0))
                elif event.key == pygame.K_d:
                    vosc.set_amplitude(max(vosc.amplitude - 0.1, 0.0))
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP] or keys[pygame.K_DOWN]:
            if current_time - last_adjust_time >= adjust_delay:
                params = effect_parameters[selected_effect]
                if params:
                    current_param = params[current_param_index]
                    if keys[pygame.K_UP]:
                        if selected_effect == 'VGain' and current_param == 'VGAIN':
                            vgain = min(vgain + VGAIN_STEP, VGAIN_MAX)
                        elif selected_effect == 'VEQ' and current_param == 'CUTOFF_FREQ':
                            CUTOFF_FREQ = min(CUTOFF_FREQ + 500, SAMPLE_RATE / 2 - 100)
                            if FILTER_TYPE == 'Lowpass':
                                filter_b, filter_a = butter_lowpass(CUTOFF_FREQ, SAMPLE_RATE, FILTER_ORDER)
                            elif FILTER_TYPE == 'Highpass':
                                filter_b, filter_a = butter_highpass(CUTOFF_FREQ, SAMPLE_RATE, FILTER_ORDER)
                        elif selected_effect == 'VDelay' and current_param == 'DELAY_TIME':
                            DELAY_TIME = min(DELAY_TIME + 0.05, 2.0)
                            if delay_effect:
                                delay_effect.update_parameters(delay_time=DELAY_TIME)
                        elif selected_effect == 'VDelay' and current_param == 'DELAY_FEEDBACK':
                            DELAY_FEEDBACK = min(DELAY_FEEDBACK + 0.1, 0.9)
                            if delay_effect:
                                delay_effect.update_parameters(feedback=DELAY_FEEDBACK)
                        elif selected_effect == 'VReverb' and current_param == 'REVERB_DECAY':
                            REVERB_DECAY = min(REVERB_DECAY + 0.1, 0.9)
                            if reverb_effect:
                                reverb_effect.update_parameters(decay=REVERB_DECAY)
                        elif selected_effect == 'VOSC' and current_param == 'VOSC_FREQ':
                            vosc.set_frequency(vosc.frequency + 10)
                        elif selected_effect == 'VOSC' and current_param == 'VOSC_AMP':
                            vosc.set_amplitude(min(vosc.amplitude + .1, 1.0))
                    elif keys[pygame.K_DOWN]:
                        if selected_effect == 'VGain' and current_param == 'VGAIN':
                            vgain = max(vgain - VGAIN_STEP, VGAIN_MIN)
                        elif selected_effect == 'VEQ' and current_param == 'CUTOFF_FREQ':
                            CUTOFF_FREQ = max(CUTOFF_FREQ - 500, 100)
                            if FILTER_TYPE == 'Lowpass':
                                filter_b, filter_a = butter_lowpass(CUTOFF_FREQ, SAMPLE_RATE, FILTER_ORDER)
                            elif FILTER_TYPE == 'Highpass':
                                filter_b, filter_a = butter_highpass(CUTOFF_FREQ, SAMPLE_RATE, FILTER_ORDER)
                        elif selected_effect == 'VDelay' and current_param == 'DELAY_TIME':
                            DELAY_TIME = max(DELAY_TIME - 0.05, 0.05)
                            if delay_effect:
                                delay_effect.update_parameters(delay_time=DELAY_TIME)
                        elif selected_effect == 'VDelay' and current_param == 'DELAY_FEEDBACK':
                            DELAY_FEEDBACK = max(DELAY_FEEDBACK - 10, 0.1)
                            if delay_effect:
                                delay_effect.update_parameters(feedback=DELAY_FEEDBACK)
                        elif selected_effect == 'VReverb' and current_param == 'REVERB_DECAY':
                            REVERB_DECAY = max(REVERB_DECAY - 0.1, 0.1)
                            if reverb_effect:
                                reverb_effect.update_parameters(decay=REVERB_DECAY)
                        elif selected_effect == 'VOSC' and current_param == 'VOSC_FREQ':
                            vosc.set_frequency(vosc.frequency - 10)
                        elif selected_effect == 'VOSC' and current_param == 'VOSC_AMP':
                            vosc.set_amplitude(max(vosc.amplitude - 0.1, 0.0))
                last_adjust_time = current_time
        data = stream.read(CHUNK, exception_on_overflow=False)
        samples = np.frombuffer(data, dtype=np.int16).astype(np.float32)
        vosc_wave = vosc.generate_wave()
        samples += vosc_wave * 32767
        samples *= vgain
        if FILTER_TYPE != 'None' and filter_b is not None and filter_a is not None:
            samples = apply_filter(samples, filter_b, filter_a)
        if DELAY_ENABLED and delay_effect is not None:
            samples = delay_effect.process(samples)
        if REVERB_ENABLED and reverb_effect is not None:
            samples = reverb_effect.process(samples)
        samples = np.clip(samples, -32767, 32767)
        samples = samples.astype(np.int16)
        if len(samples) < CHUNK:
            samples = np.pad(samples, (0, CHUNK - len(samples)), 'constant')
        colors = [sample_to_color(sample, vgain) for sample in samples]
        screen.fill((0, 0, 0))
        band_width = window_width / CHUNK
        for i, color in enumerate(colors):
            x = i * band_width
            rect = pygame.Rect(x, 0, band_width, window_height)
            pygame.draw.rect(screen, color, rect)
        #selected_effect_text = f"#yellow text"
        #text_surface = font.render(selected_effect_text, True, (255, 255, 0))
        #screen.blit(text_surface, (10, window_height - 30))
        if show_info:
            info_lines = []
            info_lines.append(f"VGain: {vgain:.1f}")
            params = effect_parameters[selected_effect]
            for param in params:
                if selected_effect == 'VGain' and param == 'VGAIN':
                    continue
                elif selected_effect == 'VEQ' and param == 'CUTOFF_FREQ':
                    info_lines.append(f"CUTOFF_FREQ: {CUTOFF_FREQ} Hz")
                elif selected_effect == 'VEQ' and param == 'FILTER_TYPE':
                    info_lines.append(f"FILTER_TYPE: {FILTER_TYPE}")
                elif selected_effect == 'VDelay' and param == 'DELAY_TIME':
                    info_lines.append(f"DELAY_TIME: {DELAY_TIME:.2f} sec")
                elif selected_effect == 'VDelay' and param == 'DELAY_FEEDBACK':
                    info_lines.append(f"DELAY_FEEDBACK: {DELAY_FEEDBACK:.1f}")
                elif selected_effect == 'VReverb' and param == 'REVERB_DECAY':
                    info_lines.append(f"REVERB_DECAY: {REVERB_DECAY:.1f}")
                elif selected_effect == 'VOSC' and param == 'VOSC_FREQ':
                    info_lines.append(f"VOSC_FREQ: {vosc.frequency} Hz")
                elif selected_effect == 'VOSC' and param == 'VOSC_AMP':
                    info_lines.append(f"VOSC_AMP: {vosc.amplitude}")
                info_lines.append(f"VOSC_PLAY: {'Playing' if vosc.is_playing else 'Stopped'}")
            for idx, line in enumerate(info_lines):
                text_surface = font.render(line, True, (255, 255, 255))
                screen.blit(text_surface, (10, 10 + idx * (font_size + 5)))
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
    except:
        pygame.quit()
        sys.exit(1)
