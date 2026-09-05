# Configuration Parameters

# Simulation Space
WORLD_W = 120.0
WORLD_H = 90.0
DEPTH = 40.0

# Resolution for Image/Video Processing
IMG_W = 800
IMG_H = 600

# Video Synchronization
VIDEO_FPS = 30.0    # Assumed FPS of input video
VIDEO_DT = 1.0 / VIDEO_FPS

# Drone Physics
MASS = 1.0
V_MAX = 50.0        # Increased to match potentially fast video motion
K_P = 3.5           # High P-Gain for tight formation holding
K_V = 8.0           # High Velocity Gain to minimize tracking lag
K_D = 2.0           # High Damping to prevent jitter on fine lines
K_REP = 18.0        # softer repulsion, more stable
R_SAFE = 1.2        # larger safety radius, smoother gradients
DT = 0.02           # Small timestep for stability

# Visualization
VIS_FPS = 30
GLOW_RADIUS = 2     # Small radius for high definition
COLOR_CORE = (255, 255, 255)
COLOR_GLOW = (0, 215, 255)   # Gold/Yellow-ish (BGR)
