"""
Run this file to visualize how Mandelbrot and Julia sets look like.
Plot the below sets using `Escape time algorithm`

The Mandelbrot set is the set of the complex numbers `c` for which the sequence z_{n+1} = {z_n}^2 + c is bounded (z_0
is set to 0). While, the definition of the Julia sets is the numbers z_0 for which the sequence is bounded, with 
a constant value.
"""

import argparse
from math import log, log2
from PIL import Image, ImageDraw
from collections import defaultdict
from math import floor, ceil

MAX_ITER = 80
# Image size (pixels)
WIDTH = 600
HEIGHT = 400

# Plot window
RE_START = -2
RE_END = 1
IM_START = -1
IM_END = 1

histogram = defaultdict(lambda: 0)


def draw_point(c, z):
    # draws a smoother set
    n = 0
    while abs(z) <= 2 and n < MAX_ITER:
        z = z * z + c
        n += 1
    if n == MAX_ITER:
        return MAX_ITER
    return n + 1 - log(log2(abs(z)))


def mandelbrot(c):
    z = 0
    return draw_point(c, z)


def julia(c, z0):
    z = z0
    return draw_point(c, z)


def linear_interpolation(color1, color2, t):
    # smoothen the colors
    return color1 * (1 - t) + color2 * t


values = {}


# Define a function to parse the command line arguments
def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a Mandelbrot set / Julia set."
    )
    parser.add_argument(
        "type",
        type=str,
        choices=["m", "j"],
        default="m",
        help="Type of set to generate: 'm' for Mandelbrot set, 'j' for Julia set",
    )
    parser.add_argument(
        "--c",
        type=str,
        default="0.285, 0.01",
        help="Complex tuple used to compute the Julia set (ignored for Mandelbrot set)",
    )
    return parser.parse_args()


def main():
    # Parse command line arguments
    args = parse_args()
    if args.c:
        const_c = tuple(map(float, args.c.split(",")))
    else:
        const_c = (0.285, 0.01)
    for x in range(0, WIDTH):
        for y in range(0, HEIGHT):
            # Convert pixel coordinate to complex number
            i = complex(
                RE_START + (x / WIDTH) * (RE_END - RE_START),
                IM_START + (y / HEIGHT) * (IM_END - IM_START),
            )
            if args.type == "m":
                # Compute the number of iterations
                m = mandelbrot(i)
            else:
                c0, c1 = const_c
                c = complex(c0, c1)
                m = julia(c, i)

            values[(x, y)] = m
            if m < MAX_ITER:
                histogram[floor(m)] += 1

    total = sum(histogram.values())
    hues = []
    h = 0
    for i in range(MAX_ITER):
        h += histogram[i] / total
        hues.append(h)
    hues.append(h)

    im = Image.new("HSV", (WIDTH, HEIGHT), (0, 0, 0))
    draw = ImageDraw.Draw(im)

    for x in range(0, WIDTH):
        for y in range(0, HEIGHT):
            m = values[(x, y)]
            # The color depends on the number of iterations
            hue = 255 - int(
                255 * linear_interpolation(hues[floor(m)], hues[ceil(m)], m % 1)
            )
            saturation = 255
            value = 255 if m < MAX_ITER else 0
            # Plot the point
            draw.point([x, y], (hue, saturation, value))

    im.convert("RGB").save("output.png", "PNG")


if __name__ == "__main__":
    main()
    # python draw.py m
    # python draw.py j --c "-0.4, 0.6"
