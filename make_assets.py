"""Procedural art for PIZERIA (original illustrations, no third-party assets).
Run: python3 tools/make_assets.py  (needs numpy, Pillow, opencv-python)"""
import numpy as np, math, os, cv2
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, '..', 'assets')
N = 1800; C = N // 2; FINAL = 900
yy, xx = np.mgrid[0:N, 0:N].astype('float32')
DX, DY = xx - C, yy - C
DIST = np.hypot(DX, DY); THETA = np.arctan2(DY, DX)

def blur(a, s): return cv2.GaussianBlur(a, (0, 0), s)
def smooth_noise(size, scale, seed):
    r = np.random.RandomState(seed)
    k = max(2, size // scale + 2)
    small = r.rand(k, k).astype('float32')
    return cv2.resize(small, (size, size), interpolation=cv2.INTER_CUBIC)
def fbm(size, seed, scales=(300, 120, 50, 20), w=(1, .6, .35, .2)):
    t = sum(wi * smooth_noise(size, s, seed + i) for i, (s, wi) in enumerate(zip(scales, w)))
    return ((t - t.min()) / (t.max() - t.min())).astype('float32')
def rad(R, amp, seed):
    r = np.random.RandomState(seed)
    t = sum(r.rand() * np.sin(f * THETA + r.rand() * 6.28) for f in (3, 5, 8, 13)) / 2
    return R + amp * t
def L(a, b, t): return a * (1 - t[..., None]) + b * t[..., None]
def rgba(col, alpha):
    out = np.dstack([np.clip(col, 0, 255), np.clip(alpha, 0, 1) * 255]).astype('uint8')
    return Image.fromarray(out, 'RGBA')
def ss(x): x = np.clip(x, 0, 1); return x * x * (3 - 2 * x)

# ---------- base layers ----------
def make_crust():
    R = rad(790, 14, 1)
    n = fbm(N, 10); n2 = fbm(N, 20, (60, 25, 10), (1, .6, .3)); fine = fbm(N, 33, (6, 3), (1, .6))
    t = np.clip((DIST - 640) / (R - 640), 0, 1)
    bump = np.sin(t * np.pi)
    rim = L(np.array([196, 128, 62.]), np.array([240, 190, 110.]), bump)
    w = ss((DIST - 625) / 30)
    col = L(np.broadcast_to(np.array([232, 200, 148.]), (N, N, 3)).copy(), rim, w)
    col *= (0.88 + 0.24 * n[..., None]) * (0.95 + 0.1 * fine[..., None])
    char = np.clip((n2 - 0.68) / 0.1, 0, 1) * ss((t - 0.05) / 0.2) * ss((0.97 - t) / 0.1)
    col = col * (1 - 0.6 * char[..., None]) + np.array([62, 30, 12.]) * 0.6 * char[..., None]
    col *= (1 - 0.4 * np.clip((t - 0.78) / 0.22, 0, 1) ** 2)[..., None]
    return rgba(col, np.clip((R - DIST) / 2.5, 0, 1))

def make_sauce(kind):
    R = rad(668, 10, 2)
    n = fbm(N, 30); fine = fbm(N, 31, (12, 5), (1, .5))
    c = {'tomato': ((200, 46, 34), (148, 26, 22)),
         'white': ((248, 238, 216), (224, 204, 166)),
         'spicy': ((222, 72, 22), (150, 28, 14))}[kind]
    col = L(np.broadcast_to(np.array(c[0], 'float32'), (N, N, 3)).copy(), np.array(c[1], 'float32'), n)
    gl = np.clip((fine - 0.78) / 0.12, 0, 1)[..., None]
    col += gl * (28 if kind != 'white' else 6)
    if kind == 'white':
        col *= (0.96 + 0.06 * np.sin(DIST / 38 + 4 * n))[..., None]
    if kind == 'spicy':
        r = np.random.RandomState(5); fl = np.zeros((N, N), 'float32')
        for _ in range(900):
            a = r.rand() * 6.28; d = 640 * math.sqrt(r.rand())
            x, y = int(C + d * math.cos(a)), int(C + d * math.sin(a))
            cv2.circle(fl, (x, y), int(3 + r.rand() * 5), 1.0, -1)
        fl = blur(fl, 1.5)[..., None]
        col = col * (1 - 0.7 * fl) + np.array([110, 14, 8.]) * 0.7 * fl
    col *= (1 - 0.25 * ss((DIST - 560) / 110))[..., None]
    return rgba(col, np.clip((R - DIST) / 2.5, 0, 1))

def make_cheese(cov, seed):
    n = fbm(N, seed, (260, 110, 50, 25), (1, .7, .4, .25))
    inside = DIST < 640
    t = np.quantile(n[inside], 1 - cov)
    h = np.clip((n - t) / 0.10, 0, 1) * ss((650 - DIST) / 25)
    hb = blur(h, 5)
    gy, gx = np.gradient(hb)
    shade = (-(gx + gy)) * 520
    g = fbm(N, seed + 5, (90, 40), (1, .6))
    gw = np.clip((g - 0.6) / 0.15, 0, 1) * 0.85
    col = L(np.broadcast_to(np.array([251, 243, 222.]), (N, N, 3)).copy(), np.array([228, 178, 100.]), gw)
    col += shade[..., None] * 1.0
    col *= (1 - 0.22 * (1 - np.clip(hb * 1.3, 0, 1)))[..., None]
    return rgba(col, np.clip(blur(h, 1.5) * 1.4, 0, 1))

# ---------- sprites ----------
def finish(im, shadow=0.42, off=(7, 11), blr=7):
    pad = 40
    W, H = im.size
    big = Image.new('RGBA', (W + 2 * pad, H + 2 * pad), (0, 0, 0, 0))
    big.paste(im, (pad, pad))
    a = np.asarray(big)[..., 3].astype('float32') / 255
    sh = np.roll(np.roll(blur(a, blr), off[1], 0), off[0], 1) * shadow
    s = Image.fromarray(np.dstack([np.zeros((*sh.shape, 3), 'uint8'), (sh * 255).astype('uint8')]), 'RGBA')
    s.alpha_composite(big)
    return s

def sprite_pepperoni(r, seed):
    P = int(2.6 * r); c = P / 2
    y, x = np.mgrid[0:P, 0:P].astype('float32'); dx, dy = x - c, y - c
    d = np.hypot(dx, dy); th = np.arctan2(dy, dx)
    rs = np.random.RandomState(seed)
    R = r * (1 + 0.04 * np.sin(3 * th + rs.rand() * 6) + 0.03 * np.sin(7 * th + rs.rand() * 6))
    t = np.clip(d / R, 0, 1)
    col = L(np.broadcast_to(np.array([206, 58, 38.]), (P, P, 3)).copy(), np.array([166, 34, 26.]), t ** 1.4)
    ring = ss((t - 0.82) / 0.1) * ss((1.0 - t) / 0.05)
    col = col * (1 - 0.35 * ring[..., None])
    n = smooth_noise(P, 12, seed)
    col *= (0.9 + 0.2 * n[..., None])
    fat = np.zeros((P, P), 'float32')
    for _ in range(26):
        a = rs.rand() * 6.28; dd = r * 0.78 * math.sqrt(rs.rand())
        cv2.circle(fat, (int(c + dd * math.cos(a)), int(c + dd * math.sin(a))), int(2 + rs.rand() * 4), 1, -1)
    fat = blur(fat, 1.4)[..., None]
    col = col * (1 - 0.6 * fat) + np.array([240, 150, 120.]) * 0.6 * fat
    hl = np.clip(1 - np.hypot(dx + r * .35, dy + r * .4) / (r * .9), 0, 1) ** 2
    col += hl[..., None] * 34
    return finish(rgba(col, np.clip((R - d) / 1.6, 0, 1)))

def aa_draw(size, fn, scale=4):
    im = Image.new('RGBA', (size[0] * scale, size[1] * scale), (0, 0, 0, 0))
    fn(ImageDraw.Draw(im), scale)
    return im.resize(size, Image.LANCZOS)

def sprite_mushroom(w, seed):
    h = int(w * 0.95)
    def f(d, s):
        right = [(0.5, 0.0), (0.47, 0.06), (0.36, 0.1), (0.22, 0.12), (0.17, 0.2), (0.16, 0.45), (0.14, 0.62)]
        cap = [(0.5 * math.cos(a), -0.5 * math.sin(a) * 0.95) for a in np.linspace(math.pi, 0, 24)]
        pts = cap + right[1:]
        pts = pts + [(-x, y) for x, y in reversed(right[1:])]
        cx, cy = w * s / 2, h * s * 0.55
        poly = [(cx + x * w * s * 0.96, cy + y * h * s * 0.9) for x, y in pts]
        d.polygon(poly, fill=(228, 212, 184, 255))
        inner = [(cx + (x * 0.9) * w * s * 0.96, cy + (y * 0.9 + 0.01) * h * s * 0.9) for x, y in pts]
        d.polygon(inner, fill=(243, 232, 210, 255))
        for a in np.linspace(0.2, 2.9, 7):
            x0 = cx + 0.33 * math.cos(a) * w * s * 0.96; y0 = cy - 0.3 * math.sin(a) * h * s * 0.9
            d.line([(cx + 0.1 * math.cos(a) * w * s, cy + 0.12 * h * s), (x0, y0)], fill=(205, 180, 146, 150), width=int(2 * s))
        d.line([(cx + x * w * s * 0.96, cy + y * h * s * 0.9) for x, y in cap], fill=(150, 112, 80, 255), width=int(4 * s))
    return finish(aa_draw((w, h), f), 0.4, (5, 8), 5)

def sprite_olive(r, seed):
    P = int(2.4 * r)
    def f(d, s):
        c = P * s / 2; R = r * s
        d.ellipse([c - R, c - R * .92, c + R, c + R * .92], fill=(36, 30, 32, 255))
        d.ellipse([c - R * .86, c - R * .8, c + R * .86, c + R * .8], outline=(78, 76, 52, 255), width=int(3 * s))
        d.arc([c - R * .8, c - R * .72, c + R * .8, c + R * .72], 200, 280, fill=(128, 124, 104, 255), width=int(4 * s))
        h = R * .4
        d.ellipse([c - h, c - h * .95, c + h, c + h * .95], fill=(0, 0, 0, 0))
    return finish(aa_draw((P, P), f), 0.4, (5, 7), 4)

def sprite_basil(L_, seed):
    Wd = int(L_ * 0.62); Hh = L_
    def f(d, s):
        cx = Wd * s / 2
        us = np.linspace(0, 1, 40)
        wid = lambda u: math.sin(math.pi * u ** 0.75) * Wd * s * 0.47
        left = [(cx - wid(u), (0.04 + 0.92 * (1 - u)) * Hh * s) for u in us]
        right = [(cx + wid(u), (0.04 + 0.92 * (1 - u)) * Hh * s) for u in reversed(us)]
        d.polygon(left + right, fill=(48, 118, 46, 255))
        inset = [(cx + (x - cx) * 0.8, y) for x, y in left + right]
        d.polygon(inset, fill=(78, 150, 62, 255))
        d.line([(cx, 0.06 * Hh * s), (cx, 0.98 * Hh * s)], fill=(168, 208, 122, 255), width=int(3.5 * s))
        for u in (0.22, 0.4, 0.58, 0.74):
            y = (0.04 + 0.92 * (1 - u)) * Hh * s; ww = wid(u) * .8
            d.line([(cx, y + 0.06 * Hh * s), (cx - ww, y - 0.04 * Hh * s)], fill=(150, 196, 110, 220), width=int(2 * s))
            d.line([(cx, y + 0.06 * Hh * s), (cx + ww, y - 0.04 * Hh * s)], fill=(150, 196, 110, 220), width=int(2 * s))
    return finish(aa_draw((Wd, Hh), f), 0.38, (5, 8), 5)

def sprite_jalapeno(r, seed):
    P = int(2.4 * r); rs = np.random.RandomState(seed)
    def f(d, s):
        c = P * s / 2; R = r * s
        d.ellipse([c - R, c - R, c + R, c + R], fill=(88, 140, 46, 255))
        d.ellipse([c - R * .88, c - R * .88, c + R * .88, c + R * .88], fill=(150, 192, 92, 255))
        d.ellipse([c - R * .62, c - R * .62, c + R * .62, c + R * .62], fill=(214, 226, 164, 255))
        for i in range(9):
            a = i * 0.7 + rs.rand() * .3; dd = R * (0.22 + 0.2 * rs.rand())
            x, y = c + dd * math.cos(a), c + dd * math.sin(a)
            d.ellipse([x - 5 * s, y - 3.5 * s, x + 5 * s, y + 3.5 * s], fill=(244, 238, 196, 255))
    return finish(aa_draw((P, P), f), 0.38, (4, 7), 4)

def sprite_onion(r, seed):
    P = int(2.3 * r); rs = np.random.RandomState(seed)
    def f(d, s):
        c = P * s / 2
        for k, rr in enumerate((0.95, 0.72, 0.5)):
            st = rs.rand() * 360; en = st + 100 + rs.rand() * 120
            R = r * rr * s
            d.arc([c - R, c - R, c + R, c + R], st, en, fill=(196, 150, 196, 255), width=int(11 * s))
            d.arc([c - R, c - R, c + R, c + R], st, en, fill=(240, 224, 240, 255), width=int(6 * s))
    return finish(aa_draw((P, P), f), 0.3, (4, 6), 3)

def sprite_tomato(r):
    P = int(2.4 * r)
    def f(d, s):
        c = P * s / 2; R = r * s
        d.ellipse([c - R, c - R, c + R, c + R], fill=(176, 34, 28, 255))
        d.ellipse([c - R * .92, c - R * .92, c + R * .92, c + R * .92], fill=(222, 70, 54, 255))
        for i in range(5):
            a0 = i * 72 + 8
            d.pieslice([c - R * .78, c - R * .78, c + R * .78, c + R * .78], a0, a0 + 56, fill=(240, 120, 98, 255))
            a = math.radians(a0 + 28)
            d.ellipse([c + R * .45 * math.cos(a) - 6 * s, c + R * .45 * math.sin(a) - 4 * s,
                       c + R * .45 * math.cos(a) + 6 * s, c + R * .45 * math.sin(a) + 4 * s], fill=(248, 214, 140, 255))
        d.ellipse([c - R * .16, c - R * .16, c + R * .16, c + R * .16], fill=(246, 214, 190, 255))
    return finish(aa_draw((P, P), f), 0.4, (6, 9), 5)

def sprite_mozz(r):
    P = int(2.4 * r)
    y, x = np.mgrid[0:P, 0:P].astype('float32'); c = P / 2
    d = np.hypot(x - c, y - c)
    lit = np.clip(1 - np.hypot(x - c + r * .35, y - c + r * .4) / (r * 1.3), 0, 1)
    col = L(np.broadcast_to(np.array([212, 206, 192.]), (P, P, 3)).copy(), np.array([255, 252, 244.]), lit)
    return finish(rgba(col, np.clip((r - d) / 1.5, 0, 1)), 0.4, (6, 10), 6)

# ---------- composition ----------
def scatter(n, Rmax, mind, seed):
    r = np.random.RandomState(seed); pts = []; tries = 0
    while len(pts) < n and tries < 20000:
        tries += 1
        a = r.rand() * 6.283; d = Rmax * math.sqrt(r.rand())
        p = (d * math.cos(a), d * math.sin(a))
        if all(math.hypot(p[0] - q[0], p[1] - q[1]) >= mind for q in pts): pts.append(p)
    return pts

def topping_layer(maker, n, mind, seed, Rmax=590, rot=True, variants=4):
    r = np.random.RandomState(seed)
    sprites = [maker(i) for i in range(variants)]
    canvas = Image.new('RGBA', (N, N), (0, 0, 0, 0))
    for (px, py) in scatter(n, Rmax, mind, seed):
        sp = sprites[r.randint(variants)]
        if rot: sp = sp.rotate(r.rand() * 360, resample=Image.BICUBIC, expand=True)
        canvas.alpha_composite(sp, (int(C + px - sp.width / 2), int(C + py - sp.height / 2)))
    return canvas

TOPPINGS = {
    'pepperoni': lambda: topping_layer(lambda i: sprite_pepperoni(92, 10 + i), 21, 215, 11),
    'mushrooms': lambda: topping_layer(lambda i: sprite_mushroom(150, i), 17, 190, 12),
    'olives': lambda: topping_layer(lambda i: sprite_olive(44, i), 30, 140, 13),
    'basil': lambda: topping_layer(lambda i: sprite_basil(170, i), 11, 260, 14),
    'jalapenos': lambda: topping_layer(lambda i: sprite_jalapeno(50, i), 15, 190, 15),
    'onion': lambda: topping_layer(lambda i: sprite_onion(66, i), 20, 190, 16),
}

def save(im, path, q=86, size=FINAL):
    im = im.resize((size, size), Image.LANCZOS)
    im.save(os.path.join(OUT, path), 'WEBP', quality=q, method=6)

def compose(*layers):
    base = Image.new('RGBA', (N, N), (0, 0, 0, 0))
    for l in layers: base.alpha_composite(l)
    return base

if __name__ == '__main__':
    for d in ('layers', 'pizzas', 'sprites'): os.makedirs(os.path.join(OUT, d), exist_ok=True)
    crust = make_crust()
    sauces = {k: make_sauce(k) for k in ('tomato', 'white', 'spicy')}
    cheese = {'mozzarella': make_cheese(0.62, 40), 'extra': make_cheese(0.88, 41)}
    tops = {k: f() for k, f in TOPPINGS.items()}
    save(crust, 'layers/crust.webp')
    for k, v in sauces.items(): save(v, f'layers/sauce-{k}.webp')
    for k, v in cheese.items(): save(v, f'layers/cheese-{k}.webp')
    for k, v in tops.items(): save(v, f'layers/top-{k}.webp')
    save(compose(crust, sauces['tomato'], cheese['mozzarella'], tops['basil']), 'pizzas/margherita.webp')
    save(compose(crust, sauces['spicy'], cheese['mozzarella'], tops['pepperoni'], tops['jalapenos']), 'pizzas/spicy-pepperoni.webp')
    save(compose(crust, sauces['white'], cheese['mozzarella'], tops['mushrooms'], tops['basil']), 'pizzas/truffle-mushroom.webp')
    save(compose(crust, sauces['tomato'], cheese['mozzarella'], tops['olives'], tops['mushrooms'], tops['onion'], tops['basil']), 'pizzas/garden-veggie.webp')
    save(compose(crust, sauces['tomato'], cheese['extra'], tops['pepperoni'], tops['basil']), 'pizzas/hero.webp', 88, 1200)
    save(compose(crust, sauces['tomato'], cheese['mozzarella'], tops['pepperoni']), 'pizzas/custom-thumb.webp', 86, 240)
    for name, im in {'basil': sprite_basil(300, 3), 'pepperoni': sprite_pepperoni(130, 5), 'mushroom': sprite_mushroom(260, 2),
                     'tomato': sprite_tomato(120), 'mozzarella': sprite_mozz(120), 'olive': sprite_olive(80, 1)}.items():
        im.save(os.path.join(OUT, f'sprites/{name}.webp'), 'WEBP', quality=88, method=6)
    print('ok')
