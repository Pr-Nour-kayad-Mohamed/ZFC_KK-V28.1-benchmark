from pathlib import Path

p = Path("benchmark_v28_no_opencv.py")
s = p.read_text()

start = s.index("def bilateral_approx(")
end = s.index("def nlm_simplified(")

new = '''def bilateral_approx(img, radius=3, sigma_color=25.0, sigma_space=2.0):
    img = img.astype(np.float64)
    h, w = img.shape
    pad = radius

    padded = np.pad(img, pad, mode="reflect")

    yy, xx = np.mgrid[-radius:radius + 1, -radius:radius + 1]
    spatial_weights = np.exp(
        -((xx ** 2 + yy ** 2) / (2.0 * sigma_space ** 2))
    )

    out = np.zeros_like(img)
    weight_sum = np.zeros_like(img)

    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            neighbor = padded[
                pad + dy:pad + dy + h,
                pad + dx:pad + dx + w
            ]

            color_diff = img - neighbor
            color_weights = np.exp(
                -(color_diff ** 2) / (2.0 * sigma_color ** 2)
            )

            weights = spatial_weights[dy + radius, dx + radius] * color_weights

            weight_sum += weights
            out += weights * neighbor

    out = out / np.maximum(weight_sum, 1e-12)

    return out


'''

p.write_text(s[:start] + new + s[end:])
print("Fonction bilateral_approx remplacee.")
