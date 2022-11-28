import io

import cv2
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image as Image
from numpy import random
from scipy.stats import multivariate_normal


def to_rgb(pil_image, bg_color=(255, 255, 255)):
    if pil_image.mode == "L":
        pil_image = pil_image.convert("RGB")
    elif pil_image.mode == "RGBA":
        pil_image.load()  # needed for split()
        background = Image.new('RGB', pil_image.size, bg_color)
        background.paste(
            pil_image,
            mask=pil_image.split()[3]  # 3 is the alpha channel
        )
        pil_image = background

    return pil_image


def render(height, cmap="terrain", shading_intensity=.25,
           light_entrance_angle=30, shading_ksize=9, shading_ellipticity=0.3):
    if height.dtype == np.float32:
        value_range = (0., 1.)
    else:
        value_range = (0, 255)

    img_byte_arr = io.BytesIO()
    plt.imsave(img_byte_arr, height, format="png",
               cmap=cmap, vmin=value_range[0], vmax=value_range[1])
    stream = io.BytesIO(img_byte_arr.getvalue())
    rgb = to_rgb(Image.open(stream))

    if shading_intensity > 0.:
        tick = np.linspace(-1., 1., shading_ksize)
        theta = np.pi * (light_entrance_angle / 180)
        sin = np.sin(theta)
        cos = np.cos(theta)
        kernel = (-sin * tick)[np.newaxis, :] + (cos * tick)[:, np.newaxis]

        eigen = np.array([[1., 0.],
                          [0., shading_ellipticity]], dtype=np.float32)
        eigen *= 0.5
        rotate = np.array([[sin, -cos],
                           [cos, sin]], dtype=np.float32)
        cov = rotate.T @ eigen @ rotate
        distr = multivariate_normal(cov=cov, mean=[0., 0.])
        x, y = np.meshgrid(tick, tick)
        pos = np.dstack((x, y))
        pdf = distr.pdf(pos)
        kernel *= pdf

        grad = cv2.filter2D(src=height, ddepth=-1, kernel=kernel,
                            borderType=cv2.BORDER_REFLECT)
        grad_min = grad.min()
        grad_max = grad.max()
        grad = grad / (grad_max - grad_min)

        hsv = rgb.convert("HSV")
        h, s, v = hsv.split()
        _v = np.asarray(v)
        _v = _v.astype(np.float32) * (1. + shading_intensity * grad)
        _v = _v.clip(value_range[0], value_range[1]).astype(np.uint8)
        _v = Image.fromarray(_v)
        rgb = Image.merge("HSV", (h, s, _v)).convert("RGB")

    return np.asarray(rgb).copy()


class TerrainGenerator:
    def __init__(self):
        self.size = None
        self.generate_size = None
        self.roughness = None
        self.heightMap = None
        self.rng = random.default_rng()

    def generate(self, size=(256, 256), roughness=0., return_uint8=True):
        if hasattr(size, "__getitem__"):
            self.size = size
        else:
            self.size = (size, size)

        self.generate_size =\
            int(np.power(2, np.ceil(np.log2(np.max(self.size))))) + 1
        generate_size = (self.generate_size, self.generate_size)

        self.roughness = np.power(2, roughness)

        self.heightMap = np.empty(generate_size, dtype=np.float32)
        init_height = self.rng.uniform(-1., 1., size=4)
        self.heightMap[0, 0] = init_height[0]
        self.heightMap[-1, 0] = init_height[1]
        self.heightMap[-1, -1] = init_height[2]
        self.heightMap[0, -1] = init_height[3]

        self._divide(self.generate_size - 1, 1.)

        argmin = np.unravel_index(self.heightMap.argmin(axis=None),
                                  self.heightMap.shape)
        argmax = np.unravel_index(self.heightMap.argmax(axis=None),
                                  self.heightMap.shape)
        range = self.heightMap[argmax] - self.heightMap[argmin]
        if range > 1.:
            self.heightMap /= range + 1e-5
        min = self.heightMap[argmin]
        max = self.heightMap[argmax]
        if min < 0.:
            self.heightMap -= min
        elif max > 1.:
            self.heightMap -= (max - 1.)

        crop_x_start =\
            self.rng.integers(0, self.generate_size - self.size[0] + 1)
        crop_y_start =\
            self.rng.integers(0, self.generate_size - self.size[1] + 1)
        result = self.heightMap[crop_x_start:(crop_x_start + self.size[0]),
                                crop_y_start:(crop_y_start + self.size[1])]

        if return_uint8:
            result = (255. * result).astype(np.uint8)
        else:
            result = result.copy()

        return result

    def _divide(self, step, scale):
        if step <= 1:
            return

        half = step // 2
        scale *= 0.5 * self.roughness
        mean = 0.
        std = scale

        square_index = range(half, self.generate_size, step)
        for x in square_index:
            for y in square_index:
                offset = self.rng.normal(mean, std)
                self._square(x, y, half, offset)

        diamond_index_odd = range(half, self.generate_size, step)
        diamond_index_even = range(0, self.generate_size, step)
        for x in diamond_index_odd:
            for y in diamond_index_even:
                offset = self.rng.normal(mean, std)
                self._diamond(x, y, half, offset)
        for x in diamond_index_even:
            for y in diamond_index_odd:
                offset = self.rng.normal(mean, std)
                self._diamond(x, y, half, offset)

        self._divide(half, scale)

    def _square(self, x, y, step, offset):
        xn = x - step
        xp = x + step
        yn = y - step
        yp = y + step

        adjacent = []
        if xn >= 0 and yn >= 0:
            adjacent.append(self.heightMap[xn, yn])
        if xn >= 0 and yp < self.generate_size:
            adjacent.append(self.heightMap[xn, yp])
        if xn < self.generate_size and yn >= 0:
            adjacent.append(self.heightMap[xp, yn])
        if xn < self.generate_size and yp < self.generate_size:
            adjacent.append(self.heightMap[xp, yp])

        self.heightMap[x, y] = np.mean(adjacent) + offset

    def _diamond(self, x, y, step, offset):
        xn = x - step
        xp = x + step
        yn = y - step
        yp = y + step

        adjacent = []
        if xn >= 0:
            adjacent.append(self.heightMap[xn, y])
        if yn >= 0:
            adjacent.append(self.heightMap[x, yn])
        if yp < self.generate_size:
            adjacent.append(self.heightMap[x, yp])
        if xp < self.generate_size:
            adjacent.append(self.heightMap[xp, y])

        self.heightMap[x, y] = np.mean(adjacent) + offset


if __name__ == '__main__':
    gen = TerrainGenerator()
    height = gen.generate((256, 256), 0.2)
    rgb = render(height, shading_ellipticity=.5, light_entrance_angle=30,
                 shading_intensity=.5, shading_ksize=15)
    plt.imsave("test.png", rgb)

    fig = plt.figure(figsize=(14, 10))
    grid = plt.GridSpec(3, 4)
    light_entrance_angles = np.array([[0, 30, 60, 90],
                                      [120, 150, 180, 210],
                                      [240, 270, 300, 330]],
                                     dtype=np.float32)

    for j in range(light_entrance_angles.shape[0]):
        for i in range(light_entrance_angles.shape[1]):
            angle = light_entrance_angles[j, i]
            ax = fig.add_subplot(grid[j, i])
            ax.imshow(render(height, shading_ellipticity=.2,
                             light_entrance_angle=angle,
                             shading_intensity=.3, shading_ksize=31))
            ax.set_axis_off()
            ax.set_title(f'light angle: {light_entrance_angles[j, i]}')

    fig.tight_layout()
    fig.savefig("test.jpg")
