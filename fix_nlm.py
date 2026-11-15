from pathlib import Path

p = Path("benchmark_v28_no_opencv.py")
s = p.read_text()

start = s.index("def nlm_simplified(")
end = s.index("def rgb_to_ycbcr(")

new = '''def nlm_simplified(img, radius=2, search_radius=4, sigma_color=25.0):
    img = img.astype(np.float64)
    h, w = img.shape
    pad = search_radius + radius

    padded = np.pad(img, pad, mode="reflect")
    out = np.zeros_like(img)
    weight_sum = np.zeros_like(img)

    for dy in range(-search_radius, search_radius + 1):
        for dx in range(-search_radius, search_radius + 1):
            if dy == 0 and dx == 0:
                continue

            patch = padded[
                pad + dy - radius:pad + dy + radius + 1,
                pad + dx - radius:pad + dx + radius + 1
            ]

            center_patch = padded[
                pad - radius:pad + radius + 1,
                pad - radius:pad + radius + 1
            ]

            diff = center_patch - patch
            distance = np.mean(diff ** 2)

            weight = np.exp(
                -distance / (2.0 * sigma_color ** 2)
            )

            neighbor = padded[
                pad + dy:pad + dy + h,
                pad + dx:pad + dx + w
            ]

            out += weight * neighbor
            weight_sum += weight

    return out / np.maximum(weight_sum, 1e-12)


'''

p.write_text(s[:start] + new + s[end:])
print("Fonction nlm_simplified remplacee.")
