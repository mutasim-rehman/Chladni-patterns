import pygame
import librosa
import numpy as np
import matplotlib.cm as cm
import warnings
from collections import deque
import os
import math

warnings.filterwarnings('ignore', category=UserWarning)

# --- Configuration & Constants ---
AUDIO_DIRECTORY = 'pieces'
FRAME_RATE = 44100
N_FFT = 4096
HOP_LENGTH = 1024

# Visualization Settings
PLATE_SIZE_PIX = 900
PANEL_WIDTH = 350
SCREEN_WIDTH = PLATE_SIZE_PIX + PANEL_WIDTH
SCREEN_HEIGHT = PLATE_SIZE_PIX
GRID_SIZE = 400
COLOR_MAP = cm.plasma
FONT_SIZE = 18

# UI Colors
COLORS = {
    'bg_main': (15, 15, 25),
    'bg_panel': (25, 25, 35),
    'accent': (120, 80, 255),
    'accent_light': (160, 120, 255),
    'text_primary': (240, 240, 250),
    'text_secondary': (180, 180, 190),
    'text_dim': (120, 120, 130),
    'particle_active': (255, 200, 120),
    'particle_new': (120, 255, 200),
    'success': (80, 255, 120),
    'warning': (255, 180, 80),
    'node_color': (80, 150, 255),
    'glass_overlay': (40, 40, 60, 120)
}

# --- Physics Configuration ---
INITIAL_SAND_PARTICLES = 15000
SAND_RADIUS = 1
VIBRATION_REPEL = 3.0
NODE_ATTRACTION = 1.5
BROWNIAN = 0.04

# --- Friction Model ---
BASE_FRICTION = 1.05
VELOCITY_DRAG = 0.05
SETTLED_INERTIA = 2.5

# --- Boundary Physics ---
BOUNDARY_MARGIN = 15

# --- RECYCLING SYSTEM Configuration ---
SPAWNING_ENABLED = True
STUCK_BOUNDARY_ZONE = 10
MAX_TOTAL_PARTICLES = 50000

# Plate properties
PLATE_PROPS = {
    'L': 0.4, 'h': 0.002, 'E': 200e9,
    'rho': 7850, 'nu': 0.3, 'shape': 'square'
}
D = (PLATE_PROPS['E'] * PLATE_PROPS['h'] ** 3) / (12 * (1 - PLATE_PROPS['nu'] ** 2))
FREQ_CONSTANT = (np.pi / (2 * PLATE_PROPS['L'] ** 2)) * np.sqrt(D / (PLATE_PROPS['rho'] * PLATE_PROPS['h']))


def get_audio_files():
    """Get list of audio files from the pieces directory."""
    if not os.path.exists(AUDIO_DIRECTORY):
        print(f"Warning: Directory '{AUDIO_DIRECTORY}' not found. Creating it...")
        os.makedirs(AUDIO_DIRECTORY)
        return []

    audio_extensions = ['.mp3', '.wav', '.flac', '.ogg', '.m4a']
    files = []

    for file in os.listdir(AUDIO_DIRECTORY):
        if any(file.lower().endswith(ext) for ext in audio_extensions):
            files.append(file)

    return sorted(files)


def draw_glassmorphic_rect(surface, rect, color, alpha=120, border_radius=15):
    """Draw a glassmorphic rectangle with blur effect."""
    # Create a surface for the glass effect
    glass_surf = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)

    # Fill with semi-transparent color
    pygame.draw.rect(glass_surf, (*color, alpha), glass_surf.get_rect(), border_radius=border_radius)

    # Add a subtle border
    pygame.draw.rect(glass_surf, (*COLORS['accent_light'], 60), glass_surf.get_rect(),
                     width=2, border_radius=border_radius)

    surface.blit(glass_surf, rect.topleft)


def draw_animated_background(surface, time_offset):
    """Draw an animated particle background."""
    width, height = surface.get_size()

    # Create flowing background particles
    for i in range(50):
        x = (i * 137.5 + time_offset * 20) % width
        y = (i * 97.3 + math.sin(time_offset * 0.5 + i) * 50) % height
        alpha = int(30 + 20 * math.sin(time_offset * 0.3 + i))
        size = 2 + int(2 * math.sin(time_offset * 0.4 + i * 0.1))

        color = (*COLORS['accent'], alpha)

        # Create a temporary surface for alpha blending
        particle_surf = pygame.Surface((size * 2, size * 2), pygame.SRCALPHA)
        pygame.draw.circle(particle_surf, color, (size, size), size)
        surface.blit(particle_surf, (int(x - size), int(y - size)))


def show_file_selection_menu(screen, font, title_font):
    """Show an elegant file selection menu."""
    files = get_audio_files()

    if not files:
        # Show no files found message
        screen.fill(COLORS['bg_main'])

        # Animated background
        draw_animated_background(screen, pygame.time.get_ticks() / 1000.0)

        title_text = title_font.render("Chladni Resonance", True, COLORS['text_primary'])
        screen.blit(title_text, (SCREEN_WIDTH // 2 - title_text.get_width() // 2, 200))

        # Glass panel for message
        panel_rect = pygame.Rect(SCREEN_WIDTH // 2 - 300, 350, 600, 200)
        draw_glassmorphic_rect(screen, panel_rect, COLORS['bg_panel'])

        msg_lines = [
            "No audio files found!",
            f"Please add audio files (.mp3, .wav, .flac, .ogg, .m4a)",
            f"to the '{AUDIO_DIRECTORY}' directory",
            "",
            "Press ESC to exit"
        ]

        for i, line in enumerate(msg_lines):
            color = COLORS['warning'] if i == 0 else COLORS['text_secondary']
            text = font.render(line, True, color)
            y = 380 + i * 30
            screen.blit(text, (SCREEN_WIDTH // 2 - text.get_width() // 2, y))

        pygame.display.flip()

        # Wait for escape
        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    return None
        return None

    selected_index = 0
    scroll_offset = 0
    max_visible = 12
    clock = pygame.time.Clock()

    while True:
        time_offset = pygame.time.get_ticks() / 1000.0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return None
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return None
                elif event.key == pygame.K_UP:
                    selected_index = (selected_index - 1) % len(files)
                elif event.key == pygame.K_DOWN:
                    selected_index = (selected_index + 1) % len(files)
                elif event.key == pygame.K_RETURN:
                    return os.path.join(AUDIO_DIRECTORY, files[selected_index])

        # Adjust scroll to keep selected item visible
        if selected_index < scroll_offset:
            scroll_offset = selected_index
        elif selected_index >= scroll_offset + max_visible:
            scroll_offset = selected_index - max_visible + 1

        screen.fill(COLORS['bg_main'])

        # Animated background
        draw_animated_background(screen, time_offset)

        # Title
        title_text = title_font.render("Chladni Resonance", True, COLORS['text_primary'])
        screen.blit(title_text, (SCREEN_WIDTH // 2 - title_text.get_width() // 2, 50))

        subtitle_text = font.render("Select an audio file to visualize", True, COLORS['text_secondary'])
        screen.blit(subtitle_text, (SCREEN_WIDTH // 2 - subtitle_text.get_width() // 2, 100))

        # File list panel
        list_rect = pygame.Rect(150, 180, SCREEN_WIDTH - 300, max_visible * 45 + 40)
        draw_glassmorphic_rect(screen, list_rect, COLORS['bg_panel'], alpha=150)

        # Draw files
        for i in range(max_visible):
            file_index = scroll_offset + i
            if file_index >= len(files):
                break

            filename = files[file_index]
            is_selected = file_index == selected_index

            y = 200 + i * 45

            # Selection highlight
            if is_selected:
                highlight_rect = pygame.Rect(170, y - 5, SCREEN_WIDTH - 340, 35)
                draw_glassmorphic_rect(screen, highlight_rect, COLORS['accent'], alpha=180)

            # File name (remove extension for display)
            display_name = os.path.splitext(filename)[0]
            if len(display_name) > 40:
                display_name = display_name[:37] + "..."

            color = COLORS['text_primary'] if is_selected else COLORS['text_secondary']
            text = font.render(display_name, True, color)
            screen.blit(text, (190, y))

            # File extension
            ext = os.path.splitext(filename)[1].upper()
            ext_text = pygame.font.Font(None, 14).render(ext, True, COLORS['accent_light'])
            screen.blit(ext_text, (SCREEN_WIDTH - 320, y + 5))

        # Scrollbar if needed
        if len(files) > max_visible:
            scrollbar_rect = pygame.Rect(SCREEN_WIDTH - 170, 200, 4, max_visible * 45)
            pygame.draw.rect(screen, COLORS['text_dim'], scrollbar_rect)

            thumb_height = max(20, (max_visible * 45) // len(files) * max_visible)
            thumb_y = 200 + (scroll_offset / len(files)) * (max_visible * 45 - thumb_height)
            thumb_rect = pygame.Rect(SCREEN_WIDTH - 172, thumb_y, 8, thumb_height)
            pygame.draw.rect(screen, COLORS['accent'], thumb_rect, border_radius=4)

        # Instructions
        instructions = [
            "↑↓ Navigate  •  ENTER Select  •  ESC Exit"
        ]

        for i, instruction in enumerate(instructions):
            text = font.render(instruction, True, COLORS['text_dim'])
            screen.blit(text, (SCREEN_WIDTH // 2 - text.get_width() // 2, SCREEN_HEIGHT - 80 + i * 25))

        pygame.display.flip()
        clock.tick(60)


def calculate_resonant_frequency(m, n):
    return FREQ_CONSTANT * (m ** 2 + n ** 2)


def generate_base_pattern(m, n, size, plate_shape='square'):
    x = np.linspace(0, PLATE_PROPS['L'], size)
    y = np.linspace(0, PLATE_PROPS['L'], size)
    xx, yy = np.meshgrid(x, y)
    if plate_shape == 'square':
        pattern = np.sin((m * np.pi * xx) / PLATE_PROPS['L']) * np.sin((n * np.pi * yy) / PLATE_PROPS['L'])
    else:
        cx, cy, r_max = PLATE_PROPS['L'] / 2, PLATE_PROPS['L'] / 2, PLATE_PROPS['L'] / 2
        r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        theta = np.arctan2(yy - cy, xx - cx)
        pattern = np.cos(m * theta) * np.sin(np.pi * n * (r / r_max))
        pattern[r > r_max] = 0
    max_abs = np.max(np.abs(pattern))
    return pattern / (max_abs + 1e-12)


def precompute_plate_modes(max_m=8, max_n=8, plate_shape='square'):
    print(f"Pre-computing modes for a {plate_shape} plate...")
    modes = []
    for m in range(max_m + 1):
        for n in range(max_m + 1):
            if m == 0 and n == 0: continue
            freq = calculate_resonant_frequency(m, n)
            if 20 <= freq <= 20000:
                modes.append({
                    'freq': freq, 'm': m, 'n': n,
                    'pattern': generate_base_pattern(m, n, GRID_SIZE, plate_shape)
                })
    modes.sort(key=lambda x: x['freq'])
    print(f"Computed {len(modes)} modes from {modes[0]['freq']:.2f} to {modes[-1]['freq']:.2f} Hz")
    return modes


class ParticleSystem:
    """Optimized particle system with recycling of boundary particles."""

    def __init__(self, initial_count, grid_size):
        self.grid_size = grid_size
        self.max_particles = MAX_TOTAL_PARTICLES

        # Pre-allocate arrays for maximum capacity
        self.pos = np.zeros((self.max_particles, 2), dtype=np.float32)
        self.vel = np.zeros((self.max_particles, 2), dtype=np.float32)
        self.settled_time = np.zeros(self.max_particles, dtype=np.int32)
        self.spawn_frame = np.zeros(self.max_particles, dtype=np.int32)

        self.active_count = 0
        self.total_spawned = 0
        self.removed_count = 0

        self.reset(initial_count)

    def _populate_particles(self, start_index, count, grid_size, is_initial):
        """Populate a slice of the pre-allocated arrays with new particles."""
        if count <= 0: return

        margin = max(grid_size // 20, 5)

        if is_initial:  # For the very first set of particles, concentrate in the middle
            center_count = int(count * 0.6)
            periphery_count = count - center_count
            center_pos = np.random.normal(loc=grid_size / 2, scale=grid_size / 12, size=(center_count, 2))
            periphery_pos = np.random.uniform(margin, grid_size - margin, (periphery_count, 2))
            pos = np.vstack([center_pos, periphery_pos])
        else:  # For spawned particles, place them randomly everywhere
            pos = np.random.uniform(margin, grid_size - margin, (count, 2))

        pos = np.clip(pos, margin, grid_size - margin)

        self.pos[start_index: start_index + count] = pos
        self.vel[start_index: start_index + count] = np.random.uniform(-0.3, 0.3, (count, 2))
        self.settled_time[start_index: start_index + count] = 0
        self.spawn_frame[start_index: start_index + count] = 0

    def _populate_particles_slots(self, indices, count, grid_size):
        """Populate specific slots with new particles."""
        margin = max(grid_size // 20, 5)
        pos = np.random.uniform(margin, grid_size - margin, (count, 2))
        pos = np.clip(pos, margin, grid_size - margin)

        self.pos[indices] = pos
        self.vel[indices] = np.random.uniform(-0.3, 0.3, (count, 2))
        self.settled_time[indices] = 0
        self.spawn_frame[indices] = 0

    def recycle_boundary_particles(self, current_frame):
        """Remove particles in boundary zone and spawn replacements in their slots."""
        if self.active_count == 0:
            return 0

        active_pos = self.pos[:self.active_count]

        # Check which particles are in the boundary zone
        dist_left = active_pos[:, 0]
        dist_right = self.grid_size - 1 - active_pos[:, 0]
        dist_top = active_pos[:, 1]
        dist_bottom = self.grid_size - 1 - active_pos[:, 1]
        dist_to_edge = np.minimum.reduce([dist_left, dist_right, dist_top, dist_bottom])

        in_boundary_mask = dist_to_edge < STUCK_BOUNDARY_ZONE
        indices_to_recycle = np.where(in_boundary_mask)[0]
        num_to_recycle = len(indices_to_recycle)

        if num_to_recycle > 0:
            # Reuse the slots of recycled particles for new ones
            self._populate_particles_slots(indices_to_recycle, num_to_recycle, self.grid_size)
            self.spawn_frame[indices_to_recycle] = current_frame

            self.total_spawned += num_to_recycle

    def get_stats(self):
        """Return particle statistics."""
        return {
            'total': self.active_count,
            'spawned': self.total_spawned,
            'removed': self.removed_count
        }

    def reset(self, initial_count):
        """Reset particle system to initial state."""
        self.active_count = min(initial_count, self.max_particles)
        # Reset all tracking arrays for the full buffer
        self.settled_time.fill(0)
        self.spawn_frame.fill(0)
        self.total_spawned = 0
        self.removed_count = 0
        self._populate_particles(0, self.active_count, self.grid_size, is_initial=True)


def compute_force_fields(vibration_field, grid_size):
    """Force field computation."""
    from scipy.ndimage import gaussian_filter

    abs_vib = np.abs(vibration_field)
    node_field = 1.0 - (abs_vib / (np.max(abs_vib) + 1e-12))
    node_field = gaussian_filter(node_field, sigma=1.5)

    node_grad_y, node_grad_x = np.gradient(node_field)
    vib_grad_y, vib_grad_x = np.gradient(abs_vib)

    def norm_clip(g):
        gmax = np.max(np.abs(g)) + 1e-12
        return g / gmax

    node_grad_x = norm_clip(node_grad_x)
    node_grad_y = norm_clip(node_grad_y)
    vib_grad_x = norm_clip(vib_grad_x)
    vib_grad_y = norm_clip(vib_grad_y)

    return vib_grad_x, vib_grad_y, node_grad_x, node_grad_y


def update_particle_system(particle_system, current_frame, fields, spawning_enabled):
    """Optimized particle system update using vectorized operations."""
    if particle_system.active_count == 0:
        return

    # Get references to active particles
    pos = particle_system.pos[:particle_system.active_count]
    vel = particle_system.vel[:particle_system.active_count]
    settled_time = particle_system.settled_time[:particle_system.active_count]

    h, w = fields['vibration_field'].shape

    # Get grid indices for all particles
    xi = np.clip(pos[:, 0].astype(int), 0, w - 1)
    yi = np.clip(pos[:, 1].astype(int), 0, h - 1)

    # Vectorized field sampling
    vibration_intensity = np.abs(fields['vibration_field'])[yi, xi]

    # Vectorized physics calculations
    can_settle = vibration_intensity < 0.05
    settled_time[can_settle] += 1
    settled_time[~can_settle] = 0
    is_settled = settled_time > 15

    force_modifier = np.ones(particle_system.active_count, dtype=float)
    force_modifier[is_settled] = 1.0 / SETTLED_INERTIA

    force_x = (-fields['vib_grad_x'][yi, xi] * VIBRATION_REPEL +
               fields['node_grad_x'][yi, xi] * NODE_ATTRACTION)
    force_y = (-fields['vib_grad_y'][yi, xi] * VIBRATION_REPEL +
               fields['node_grad_y'][yi, xi] * NODE_ATTRACTION)

    rnd_x = (np.random.rand(particle_system.active_count) - 0.5) * BROWNIAN
    rnd_y = (np.random.rand(particle_system.active_count) - 0.5) * BROWNIAN

    vel[:, 0] += (force_x * force_modifier) + rnd_x
    vel[:, 1] += (force_y * force_modifier) + rnd_y

    # Vectorized drag calculation
    vel_mag = np.sqrt(vel[:, 0] ** 2 + vel[:, 1] ** 2)
    drag_factor = np.clip(BASE_FRICTION - vel_mag * VELOCITY_DRAG, 0.0, 1.0)
    vel[:, 0] *= drag_factor
    vel[:, 1] *= drag_factor

    # Update positions
    pos[:, 0] += vel[:, 0]
    pos[:, 1] += vel[:, 1]

    # Vectorized boundary handling
    margin_from_edge = 3
    min_bound = margin_from_edge
    max_bound_x = w - margin_from_edge - 1
    max_bound_y = h - margin_from_edge - 1

    # X boundary
    mask_low_x = pos[:, 0] < min_bound
    pos[mask_low_x, 0] = min_bound + (min_bound - pos[mask_low_x, 0]) * 0.2
    vel[mask_low_x, 0] = np.abs(vel[mask_low_x, 0]) * 0.35

    mask_high_x = pos[:, 0] > max_bound_x
    pos[mask_high_x, 0] = max_bound_x - (pos[mask_high_x, 0] - max_bound_x) * 0.2
    vel[mask_high_x, 0] = -np.abs(vel[mask_high_x, 0]) * 0.35

    # Y boundary
    mask_low_y = pos[:, 1] < min_bound
    pos[mask_low_y, 1] = min_bound + (min_bound - pos[mask_low_y, 1]) * 0.2
    vel[mask_low_y, 1] = np.abs(vel[mask_low_y, 1]) * 0.35

    mask_high_y = pos[:, 1] > max_bound_y
    pos[mask_high_y, 1] = max_bound_y - (pos[mask_high_y, 1] - max_bound_y) * 0.2
    vel[mask_high_y, 1] = -np.abs(vel[mask_high_y, 1]) * 0.35

    # Handle recycling
    if spawning_enabled:
        particle_system.recycle_boundary_particles(current_frame)


# --- Audio Analysis ---
def analyze_audio(audio_path):
    print(f"Loading and analyzing audio: {os.path.basename(audio_path)}")
    try:
        y, sr = librosa.load(audio_path, sr=FRAME_RATE, mono=True)
    except Exception as e:
        print(f"FATAL: Could not load audio file '{audio_path}'. Error: {e}")
        return None, None, None
    stft = librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH)
    magnitude = np.abs(stft)
    frame_times = librosa.frames_to_time(np.arange(magnitude.shape[1]), sr=sr, hop_length=HOP_LENGTH)
    print("Audio analysis complete.")
    return magnitude, frame_times, sr


# Create a particle surface for efficient rendering
particle_surface = None


def init_particle_surface():
    """Initialize the particle surface."""
    global particle_surface
    particle_surface = pygame.Surface((PLATE_SIZE_PIX, PLATE_SIZE_PIX), pygame.SRCALPHA)


def draw_particles_optimized(screen, particle_system, settings, frame_counter, time_offset):
    """Optimized particle drawing using surfaces."""
    global particle_surface

    # Clear the particle surface
    particle_surface.fill((0, 0, 0, 0))

    num_to_draw = particle_system.active_count
    if num_to_draw == 0:
        return

    pos_to_draw = particle_system.pos[:num_to_draw]
    spawn_frame_to_draw = particle_system.spawn_frame[:num_to_draw]
    scale = PLATE_SIZE_PIX / GRID_SIZE

    # Convert positions to screen coordinates
    draw_pos = (pos_to_draw * scale).astype(int)

    # Batch draw particles based on age
    new_indices = np.where(frame_counter - spawn_frame_to_draw < 60)[0]
    old_indices = np.where(frame_counter - spawn_frame_to_draw >= 60)[0]

    # Draw older particles
    for i in old_indices:
        pos = tuple(draw_pos[i])
        brightness = 0.8 + 0.2 * math.sin(time_offset * 2 + i * 0.1)
        color = tuple(int(c * brightness) for c in COLORS['particle_active'])
        pygame.draw.circle(particle_surface, color, pos, SAND_RADIUS)

    # Draw newer particles with glow
    for i in new_indices:
        pos = tuple(draw_pos[i])
        age = frame_counter - spawn_frame_to_draw[i]
        glow_intensity = 1 - age / 60
        glow_size = int(3 + 2 * glow_intensity)

        glow_intensity=0
        glow_size=0

        # Outer glow
        if glow_size > 0:
            glow_alpha = int(100 * glow_intensity)
            glow_surf = pygame.Surface((glow_size * 2, glow_size * 2), pygame.SRCALPHA)
            pygame.draw.circle(glow_surf, (*COLORS['particle_new'], glow_alpha),
                               (glow_size, glow_size), glow_size)
            particle_surface.blit(glow_surf, (pos[0] - glow_size, pos[1] - glow_size))

        # Core particle
        pygame.draw.circle(particle_surface, COLORS['particle_new'], pos, SAND_RADIUS)

    # Blit the particle surface to screen
    screen.blit(particle_surface, (0, 0))


# Precompute color maps and other static elements
color_map_cache = {}


def get_cached_color_map(value, cmap_name):
    """Cache color map lookups to avoid repeated calculations."""
    if cmap_name not in color_map_cache:
        color_map_cache[cmap_name] = cm.get_cmap(cmap_name)

    cmap = color_map_cache[cmap_name]
    return (cmap(value)[:3] * 255).astype(np.uint8)


# Precompute frequency bins and mode matching
def precompute_frequency_bins(mode_freqs, freqs_hz):
    """Precompute which frequency bins correspond to which modes."""
    mode_bins = {}
    for i, mode_freq in enumerate(mode_freqs):
        # Find the closest frequency bin
        bin_idx = np.argmin(np.abs(freqs_hz - mode_freq))
        mode_bins[i] = bin_idx
    return mode_bins


def draw_enhanced_ui(screen, font, title_font, current_time_sec, clock, particle_system,
                     active_modes_info, current_spectrum, freqs_hz, settings, spawning_enabled,
                     selected_file, time_offset):
    """Draw the enhanced, beautiful UI."""

    # Right panel background
    panel_rect = pygame.Rect(PLATE_SIZE_PIX, 0, PANEL_WIDTH, SCREEN_HEIGHT)
    draw_glassmorphic_rect(screen, panel_rect, COLORS['bg_panel'], alpha=200)

    right_x0 = PLATE_SIZE_PIX + 20
    right_y0 = 20
    right_width = PANEL_WIDTH - 40

    # Title section
    title_text = title_font.render("Resonance", True, COLORS['text_primary'])
    screen.blit(title_text, (right_x0, right_y0))

    # Currently playing
    current_file = os.path.splitext(os.path.basename(selected_file))[0] if selected_file else "No file"
    if len(current_file) > 20:
        current_file = current_file[:17] + "..."
    file_text = font.render(current_file, True, COLORS['accent_light'])
    screen.blit(file_text, (right_x0, right_y0 + 40))

    # Time and FPS in a glass panel
    stats_rect = pygame.Rect(right_x0, right_y0 + 80, right_width, 60)
    draw_glassmorphic_rect(screen, stats_rect, COLORS['bg_main'], alpha=100)

    time_text = f"⏱ {current_time_sec:.1f}s"
    fps_text = f"⚡ {clock.get_fps():.0f} FPS"

    screen.blit(font.render(time_text, True, COLORS['text_secondary']), (right_x0 + 10, right_y0 + 95))
    screen.blit(font.render(fps_text, True, COLORS['text_secondary']), (right_x0 + 10, right_y0 + 115))

    # Spectrum visualization - More artistic
    spectrum_y = right_y0 + 160
    spectrum_rect = pygame.Rect(right_x0, spectrum_y, right_width, 120)
    draw_glassmorphic_rect(screen, spectrum_rect, COLORS['bg_main'], alpha=120)

    # Draw spectrum as flowing bars
    num_bars = 50
    indices = np.unique(np.geomspace(1, len(freqs_hz) - 1, num=num_bars).astype(int))
    mags = current_spectrum[indices] / (np.max(current_spectrum) + 1e-9)
    bar_w = (right_width - 20) / len(indices)

    for i, mval in enumerate(mags):
        # Animated height
        animated_height = mval * (1 + 0.1 * math.sin(time_offset * 3 + i * 0.2))
        bh = int(animated_height * 100)
        x = right_x0 + 10 + int(i * bar_w)
        y = spectrum_y + 110 - bh

        # Gradient colors based on frequency
        hue = i / len(indices)
        r = int(120 + 135 * hue)
        g = int(80 + 100 * (1 - hue))
        b = int(255 * mval)

        pygame.draw.rect(screen, (r, g, b), (x, y, max(2, int(bar_w * 0.8)), bh))

    # Particle stats in glass panel
    stats = particle_system.get_stats()
    particle_rect = pygame.Rect(right_x0, spectrum_y + 140, right_width, 120)
    draw_glassmorphic_rect(screen, particle_rect, COLORS['bg_main'], alpha=100)

    particle_info = [
        f"◉ Active: {stats['total']:,}",
        f"✨ New: {stats['spawned']:,}",
        f"🔄 Recycling: {'ON' if spawning_enabled else 'OFF'}",
        f"🎵 Modes: {len(active_modes_info)}"
    ]

    for i, info in enumerate(particle_info):
        color = COLORS['success'] if 'ON' in info else COLORS['text_secondary']
        screen.blit(font.render(info, True, color), (right_x0 + 10, spectrum_y + 155 + i * 22))

    # Active modes display
    modes_y = spectrum_y + 280
    if active_modes_info:
        modes_rect = pygame.Rect(right_x0, modes_y, right_width, min(120, len(active_modes_info) * 20 + 20))
        draw_glassmorphic_rect(screen, modes_rect, COLORS['bg_main'], alpha=100)

        modes_title = font.render("Active Resonances", True, COLORS['accent_light'])
        screen.blit(modes_title, (right_x0 + 10, modes_y + 5))

        for i, (m, n, freq) in enumerate(active_modes_info[:5]):
            mode_text = f"({m},{n}) {freq:.0f}Hz"
            color = COLORS['text_secondary']
            screen.blit(font.render(mode_text, True, color), (right_x0 + 15, modes_y + 30 + i * 18))

    # Settings panel
    settings_y = SCREEN_HEIGHT - 200
    settings_rect = pygame.Rect(right_x0, settings_y, right_width, 120)
    draw_glassmorphic_rect(screen, settings_rect, COLORS['bg_main'], alpha=100)

    settings_title = font.render("Settings", True, COLORS['accent_light'])
    screen.blit(settings_title, (right_x0 + 10, settings_y + 5))

    settings_info = [
        f"🎚 Sensitivity: {settings['sensitivity']}%",
        f"👁 Mode: {settings['mode'].title()}",
        f"📐 Shape: {settings['shape'].title()}",
        f"🔗 Nodes: {'Visible' if settings['nodes'] else 'Hidden'}"
    ]

    for i, info in enumerate(settings_info):
        screen.blit(font.render(info, True, COLORS['text_secondary']), (right_x0 + 15, settings_y + 30 + i * 20))

    # Controls at bottom
    controls_y = SCREEN_HEIGHT - 60
    control_lines = [
        "⌨ Controls: ±Sens • C:View • R:Shape • N:Nodes",
        "Space:Reset • S:Spawn • F:File • Esc:Quit"
    ]

    for i, line in enumerate(control_lines):
        text = pygame.font.Font(None, 16).render(line, True, COLORS['text_dim'])
        screen.blit(text, (15, controls_y + i * 18))


def main():
    pygame.init()

    # Enhanced fonts
    font = pygame.font.Font(None, FONT_SIZE)
    title_font = pygame.font.Font(None, 32)

    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption('Chladni Resonance - Interactive Physics Art')

    # Initialize particle surface
    init_particle_surface()

    # Show file selection menu
    selected_file = show_file_selection_menu(screen, font, title_font)
    if not selected_file:
        pygame.quit()
        return

    # Analyze selected audio
    magnitude, frame_times, sample_rate = analyze_audio(selected_file)
    if magnitude is None:
        pygame.quit()
        return

    plate_modes = precompute_plate_modes(plate_shape=PLATE_PROPS['shape'])
    mode_freqs = np.array([m['freq'] for m in plate_modes])

    clock = pygame.time.Clock()
    particle_system = ParticleSystem(INITIAL_SAND_PARTICLES, GRID_SIZE)
    energy_history = deque(maxlen=5)
    settings = {'sensitivity': 90, 'mode': 'sand', 'shape': PLATE_PROPS['shape'], 'nodes': False}
    spawning_enabled = SPAWNING_ENABLED
    frame_counter = 0
    show_file_menu = False

    # Precompute frequency bins
    freqs_hz = librosa.fft_frequencies(sr=sample_rate, n_fft=N_FFT)
    mode_bins = precompute_frequency_bins(mode_freqs, freqs_hz)

    try:
        pygame.mixer.init(frequency=FRAME_RATE)
        pygame.mixer.music.load(selected_file)
        pygame.mixer.music.play()
    except pygame.error as e:
        print(f"Warning: Could not play audio: {e}")

    running = True

    while running:
        frame_counter += 1
        time_offset = pygame.time.get_ticks() / 1000.0

        for event in pygame.event.get():
            if event.type == pygame.QUIT or (
                    event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_PLUS, pygame.K_EQUALS):
                    settings['sensitivity'] = min(99, settings['sensitivity'] + 1)
                elif event.key == pygame.K_MINUS:
                    settings['sensitivity'] = max(80, settings['sensitivity'] - 1)
                elif event.key == pygame.K_c:
                    settings['mode'] = 'heatmap' if settings['mode'] == 'sand' else 'sand'
                elif event.key == pygame.K_n:
                    settings['nodes'] = not settings['nodes']
                elif event.key == pygame.K_SPACE:
                    particle_system.reset(INITIAL_SAND_PARTICLES)
                    frame_counter = 0
                elif event.key == pygame.K_r:
                    settings['shape'] = 'circle' if settings['shape'] == 'square' else 'square'
                    plate_modes = precompute_plate_modes(plate_shape=settings['shape'])
                    mode_freqs = np.array([m['freq'] for m in plate_modes])
                    mode_bins = precompute_frequency_bins(mode_freqs, freqs_hz)
                elif event.key == pygame.K_s:
                    spawning_enabled = not spawning_enabled
                elif event.key == pygame.K_f:
                    # Show file selection menu again
                    pygame.mixer.music.stop()
                    new_file = show_file_selection_menu(screen, font, title_font)
                    if new_file and new_file != selected_file:
                        selected_file = new_file
                        magnitude, frame_times, sample_rate = analyze_audio(selected_file)
                        if magnitude is not None:
                            freqs_hz = librosa.fft_frequencies(sr=sample_rate, n_fft=N_FFT)
                            mode_bins = precompute_frequency_bins(mode_freqs, freqs_hz)
                            particle_system.reset(INITIAL_SAND_PARTICLES)
                            frame_counter = 0
                            try:
                                pygame.mixer.music.load(selected_file)
                                pygame.mixer.music.play()
                            except pygame.error as e:
                                print(f"Warning: Could not play audio: {e}")

        current_time_sec = pygame.mixer.music.get_pos() / 1000.0 if pygame.mixer.music.get_busy() else frame_times[-1]
        frame_index = np.searchsorted(frame_times, current_time_sec)
        frame_index = min(frame_index, magnitude.shape[1] - 1)
        current_spectrum = magnitude[:, frame_index]

        peak_threshold = np.percentile(current_spectrum, settings['sensitivity'])
        total_pattern = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
        active_modes_info = []

        # Use precomputed mode bins for faster processing
        for mode_idx, bin_idx in mode_bins.items():
            if current_spectrum[bin_idx] > peak_threshold:
                best_mode = plate_modes[mode_idx]
                total_pattern += best_mode['pattern'] * current_spectrum[bin_idx]
                active_modes_info.append((best_mode['m'], best_mode['n'], best_mode['freq']))

        energy_history.append(total_pattern)
        avg_pattern = np.mean(energy_history, axis=0) if energy_history else total_pattern
        fields = {'vibration_field': avg_pattern}
        fields['vib_grad_x'], fields['vib_grad_y'], fields['node_grad_x'], fields['node_grad_y'] = compute_force_fields(
            avg_pattern, GRID_SIZE)

        update_particle_system(particle_system, frame_counter, fields, spawning_enabled)

        # --- ENHANCED DRAWING ---
        screen.fill(COLORS['bg_main'])

        # Add subtle animated background to the main area
        main_rect = pygame.Rect(0, 0, PLATE_SIZE_PIX, SCREEN_HEIGHT)
        pygame.draw.rect(screen, COLORS['bg_main'], main_rect)

        # Draw particles with optimized method
        if settings['mode'] == 'heatmap' and np.any(total_pattern):
            # Enhanced heatmap with better colors
            norm = (avg_pattern - avg_pattern.min()) / (np.ptp(avg_pattern) + 1e-9)
            colored = (COLOR_MAP(norm)[:, :, :3] * 255).astype(np.uint8)

            # Add glow effect
            surf = pygame.surfarray.make_surface(colored.swapaxes(0, 1))
            surf = pygame.transform.smoothscale(surf, (PLATE_SIZE_PIX, PLATE_SIZE_PIX))
            screen.blit(surf, (0, 0))

            # Draw particles using optimized method
            draw_particles_optimized(screen, particle_system, settings, frame_counter, time_offset)
        else:
            # Enhanced particle mode
            if settings['nodes'] and np.any(avg_pattern):
                # Beautiful node visualization
                node_viz = np.clip(1 - np.abs(avg_pattern) / (np.max(np.abs(avg_pattern)) + 1e-9), 0, 1)
                colored = (cm.Blues(node_viz)[:, :, :3] * 255).astype(np.uint8)
                surf = pygame.surfarray.make_surface(colored.swapaxes(0, 1))
                surf.set_alpha(100)
                screen.blit(pygame.transform.smoothscale(surf, (PLATE_SIZE_PIX, PLATE_SIZE_PIX)), (0, 0))

            # Draw particles using optimized method
            draw_particles_optimized(screen, particle_system, settings, frame_counter, time_offset)

        # Draw UI every frame to prevent blinking
        draw_enhanced_ui(screen, font, title_font, current_time_sec, clock, particle_system,
                         active_modes_info, current_spectrum, freqs_hz, settings, spawning_enabled,
                         selected_file, time_offset)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


if __name__ == '__main__':
    main()