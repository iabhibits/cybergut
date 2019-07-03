import numpy as np

def random_circle(r0, circles, radi, radius, pad):
    c0_peri_rad = r0+pad+radius
    success = False
    while not success:
        theta = np.random.rand()*360
        x = c0_peri_rad*np.cos(theta)
        y = c0_peri_rad*np.sin(theta)

        success = True
        for (x1, y1), rad in zip(circles, radi):
            dist = (x1-x)**2 + (y1-y)**2
            if dist < (rad+radius+pad)**2:
                success = False
                break
        if success:
            return (x, y)

def get_circles(ps, r0, x0=0, y0=0):
    c0 = (0, 0)
    rs = np.random.randint(r0-20, r0+20, 3)
    circles = []
    radi = []
    final_cs= [(x0, y0)]
    final_rs = [r0]
    for r, p in zip(rs, ps):
        x, y = random_circle(r0, circles, radi, r, p)
        circles.append((x, y))
        radi.append(r)
        x = int(x+.5)+x0
        y = int(y+.5)+y0
        final_cs.append((x, y))
        final_rs.append(r)
    return final_cs, final_rs

def uniform_random_circles(n, min, max, r_min, r_max, pad):
    xs = np.zeros(n, int)
    ys = np.zeros(n, int)
    rs = np.zeros(n, int)
    xs[0] = np.random.randint(min, max, dtype=int)
    ys[0] = np.random.randint(min, max, dtype=int)
    rs[0] = np.random.randint(r_min, r_max)

    for i in range(1, n):
        while True:
            x = np.random.randint(min, max, dtype=int)
            y = np.random.randint(min, max, dtype=int)
            r = np.random.randint(r_min, r_max)
            dst1 = (xs[:i] - x)**2 + (ys[:i] - y)**2
            dst2 = (rs[:i] + r + pad)**2
            if np.all(dst1 > dst2):
                xs[i] = x
                ys[i] = y
                rs[i] = r
                break
    return [((x, y), r) for x, y, r in zip(xs, ys, rs)]