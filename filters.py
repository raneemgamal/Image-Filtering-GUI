import numpy as np
import cv2

# --- I/O OPERATIONS ---
def load_image_from_disk(path):
    return cv2.imread(path)

def save_image_to_disk(path, image):
    cv2.imwrite(path, image)

def ensure_grayscale(image):
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image

# --- RGB CHANNELS ---
def get_red_channel(image):
    if len(image.shape) < 3: return image
    res = np.zeros_like(image)
    res[:, :, 2] = image[:, :, 2]
    return res

def get_green_channel(image):
    if len(image.shape) < 3: return image
    res = np.zeros_like(image)
    res[:, :, 1] = image[:, :, 1]
    return res

def get_blue_channel(image):
    if len(image.shape) < 3: return image
    res = np.zeros_like(image)
    res[:, :, 0] = image[:, :, 0]
    return res

# --- ARITHMETIC OPERATIONS ---
def ensure_bgr(image):
    if len(image.shape) == 2:  
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif len(image.shape) == 3 and image.shape[2] == 4:  
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image

def prepare_for_math(img1, img2):
    i1 = ensure_bgr(img1)
    i2 = ensure_bgr(img2)
    
    if i1.dtype != np.uint8:
        i1 = np.clip(i1, 0, 255).astype(np.uint8)
    if i2.dtype != np.uint8:
        i2 = np.clip(i2, 0, 255).astype(np.uint8)
        
    i2_resized = cv2.resize(i2, (i1.shape[1], i1.shape[0]))
    return i1, i2_resized

def blend_images(img1, img2, alpha=0.5):
    i1, i2 = prepare_for_math(img1, img2)
    return cv2.addWeighted(i1, alpha, i2, 1 - alpha, 0)

def add_images(img1, img2):
    i1, i2 = prepare_for_math(img1, img2)
    return cv2.add(i1, i2)

def subtract_images(img1, img2):
    i1, i2 = prepare_for_math(img1, img2)
    return cv2.subtract(i1, i2)

def diff_images(img1, img2):
    i1, i2 = prepare_for_math(img1, img2)
    return cv2.absdiff(i1, i2)

# --- BASIC ENHANCEMENT ---
def apply_weighted_grayscale(image):
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image

def apply_complement(image):
    return 255 - image

def histogram_stretching_color(image):
    if len(image.shape) == 3:
        channels = cv2.split(image)
        stretched = [cv2.normalize(c, None, 0, 255, cv2.NORM_MINMAX) for c in channels]
        return cv2.merge(stretched)
    return cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)

def apply_floyd_steinberg(image):
    gray = ensure_grayscale(image).astype(np.float32)
    h, w = gray.shape
    for y in range(h - 1):
        for x in range(1, w - 1):
            old_px = gray[y, x]
            new_px = 255 if old_px > 128 else 0
            gray[y, x] = new_px
            error = old_px - new_px
            gray[y, x+1] += error * 7/16
            gray[y+1, x-1] += error * 3/16
            gray[y+1, x] += error * 5/16
            gray[y+1, x+1] += error * 1/16
    return np.clip(gray, 0, 255).astype(np.uint8)

# --- SPATIAL FILTERS & NOISE ---
def apply_mean_filter(image, size=3):
    return cv2.blur(image, (size, size))

def restore_image(image):
    return cv2.medianBlur(image, 5)

def apply_min_filter(image, size=3):
    return cv2.erode(image, np.ones((size, size), np.uint8))

def apply_max_filter(image, size=3):
    return cv2.dilate(image, np.ones((size, size), np.uint8))

def add_salt_and_pepper(image, prob=0.02):
    output = image.copy()
    probs = np.random.random(output.shape[:2])
    output[probs < (prob/2)] = 0
    output[probs > 1 - (prob/2)] = 255
    return output

# --- THRESHOLDING ---
def apply_otsu_threshold(image):
    gray = ensure_grayscale(image)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh

def apply_binary_threshold(image, thresh=127):
    gray = ensure_grayscale(image)
    _, binary_img = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
    return binary_img

# --- MORPHOLOGICAL OPERATIONS ---
def apply_erosion(image, size=3):
    return cv2.erode(image, np.ones((size, size), np.uint8))

def apply_dilation(image, size=3):
    return cv2.dilate(image, np.ones((size, size), np.uint8))

def apply_opening(image, size=3):
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, np.ones((size, size), np.uint8))

def apply_closing(image, size=3):
    return cv2.morphologyEx(image, cv2.MORPH_CLOSE, np.ones((size, size), np.uint8))

# --- BRIGHTNESS & CONTRAST ---
def adjust_brightness(image, value):
    return cv2.convertScaleAbs(image, alpha=1, beta=value)

def adjust_contrast(image, factor):
    return cv2.convertScaleAbs(image, alpha=factor, beta=0)

# --- STYLIZATION EFFECTS ---
import numpy as np
import cv2

# ---------- BASIC ----------
def load_image_from_disk(path):
    return cv2.imread(path)

def save_image_to_disk(path, image):
    cv2.imwrite(path, image)

def ensure_bgr(image):
    if len(image.shape) == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image


# ---------- NEW FILTERS ----------

# ð¥ 1. High Pass Sharpen (Ø§Ø­ØªØ±Ø§ÙÙ)
def highpass_sharpen(image):
    image = ensure_bgr(image)
    blur = cv2.GaussianBlur(image, (0,0), 5)
    return cv2.addWeighted(image, 1.8, blur, -0.8, 0)


# ð¬ 2. Cinematic Warm (Ø¨Ø¯ÙÙ Sepia Ø£ÙÙÙ)
def cinematic_warm(image):
    image = ensure_bgr(image).astype(np.float32)

    # increase reds / decrease blues
    image[:,:,2] *= 1.15  # red
    image[:,:,1] *= 1.05  # green
    image[:,:,0] *= 0.9   # blue

    return np.clip(image, 0, 255).astype(np.uint8)


# ð¨ 3. Comic Style (Ø¨Ø¯ÙÙ Cartoon)
def comic_filter(image):
    image = ensure_bgr(image)

    # smooth
    smooth = cv2.bilateralFilter(image, 9, 250, 250)

    # edges
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 150)
    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    # combine
    return cv2.bitwise_and(smooth, 255 - edges)


# ð 4. Dramatic Contrast
def dramatic_filter(image):
    image = ensure_bgr(image)
    return cv2.convertScaleAbs(image, alpha=1.6, beta=-30)


# âï¸ 5. Cool Tone
def cool_filter(image):
    image = ensure_bgr(image).astype(np.float32)

    image[:,:,0] *= 1.2   # blue up
    image[:,:,2] *= 0.9   # red down

    return np.clip(image, 0, 255).astype(np.uint8)


# âï¸ 6. Sketch Effect
def sketch_filter(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    inv = 255 - gray
    blur = cv2.GaussianBlur(inv, (21,21), 0)

    sketch = cv2.divide(gray, 255 - blur, scale=256)
    return sketch