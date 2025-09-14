#!/usr/bin/env python3
"""
String Art Generator - Python implementation optimized for speed on macOS
Ported from Go version with NumPy and Numba optimizations
"""

import argparse
import time
import math
import numpy as np
from PIL import Image
import aggdraw
from numba import jit
from typing import List, Tuple, Optional


# Constants
MIN_DISTANCE = 30


@jit(nopython=True)
def calculate_pin_coords(pins: int, img_size: int) -> np.ndarray:
    """Calculate pin coordinates around a circle."""
    pin_coords = np.zeros((pins, 2), dtype=np.float64)
    center = img_size / 2
    radius = img_size / 2 - 1

    for i in range(pins):
        angle = 2 * math.pi * i / pins
        pin_coords[i, 0] = math.floor(center + radius * math.cos(angle))  # X
        pin_coords[i, 1] = math.floor(center + radius * math.sin(angle))  # Y

    return pin_coords


@jit(nopython=True)
def bresenham_line(x0: int, y0: int, x1: int, y1: int) -> Tuple[np.ndarray, np.ndarray]:
    """Generate line coordinates using Bresenham's algorithm."""
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    # Estimate maximum number of points
    max_points = dx + dy + 1
    x_coords = np.zeros(max_points, dtype=np.int32)
    y_coords = np.zeros(max_points, dtype=np.int32)

    x, y = x0, y0
    idx = 0

    while True:
        x_coords[idx] = x
        y_coords[idx] = y
        idx += 1

        if x == x1 and y == y1:
            break

        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy

    return x_coords[:idx], y_coords[:idx]


def precalculate_lines(pin_coords: np.ndarray, pins: int):
    """Precalculate all potential lines between pins."""
    # Store line data in a dictionary for efficient lookup
    line_cache = {}

    print("Precalculating potential lines...")
    for i in range(pins):
        if i % 50 == 0:
            print(f"  Processing pin {i}/{pins}")

        for j in range(i + MIN_DISTANCE, pins):
            x0 = int(pin_coords[i, 0])
            y0 = int(pin_coords[i, 1])
            x1 = int(pin_coords[j, 0])
            y1 = int(pin_coords[j, 1])

            xs, ys = bresenham_line(x0, y0, x1, y1)

            # Store both directions
            line_cache[(i, j)] = (xs, ys)
            line_cache[(j, i)] = (xs, ys)

    return line_cache


@jit(nopython=True)
def get_line_error(
    error_img: np.ndarray, x_coords: np.ndarray, y_coords: np.ndarray, img_size: int
) -> float:
    """Calculate the error reduction for a line."""
    total_error = 0.0
    valid_points = 0

    for i in range(len(x_coords)):
        x, y = x_coords[i], y_coords[i]
        if 0 <= x < img_size and 0 <= y < img_size:
            total_error += error_img[y, x]
            valid_points += 1

    return total_error / valid_points if valid_points > 0 else 0.0


@jit(nopython=True)
def apply_line_to_error(
    error_img: np.ndarray,
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    line_weight: int,
    img_size: int,
):
    """Apply a line to the error image."""
    for i in range(len(x_coords)):
        x, y = x_coords[i], y_coords[i]
        if 0 <= x < img_size and 0 <= y < img_size:
            error_img[y, x] -= line_weight


@jit(nopython=True)
def contains_recent_pin(last_pins: np.ndarray, pin: int) -> bool:
    """Check if a pin was used recently."""
    for i in range(len(last_pins)):
        if last_pins[i] == pin:
            return True
    return False


class StringArtGenerator:
    def __init__(
        self,
        pins: int = 300,
        max_lines: int = 4000,
        target_size: int = 500,
        output_size: Optional[int] = None,
        line_weight: int = 8,
        output_weight: Optional[int] = None,
    ):
        self.pins = pins
        self.max_lines = max_lines
        self.target_size = target_size
        self.output_size = output_size or target_size
        self.line_weight = line_weight
        self.output_weight = output_weight or line_weight

        self.pin_coords = None
        self.line_cache = None

    def load_and_process_image(self, filename: str) -> np.ndarray:
        """Load and process input image."""
        print(f"Processing {filename}...")

        # Load image
        img = Image.open(filename).convert("RGB")
        orig_width, orig_height = img.size

        # Crop to square and resize
        size = min(orig_width, orig_height)
        left = (orig_width - size) // 2
        top = (orig_height - size) // 2
        img = img.crop((left, top, left + size, top + size))
        img = img.resize((self.target_size, self.target_size), Image.Resampling.LANCZOS)

        print(
            f"  Resized from {orig_width}x{orig_height} to {self.target_size}x{self.target_size}"
        )

        # Convert to grayscale using luminance formula
        img_array = np.array(img, dtype=np.float64)
        luminance = (
            0.2126 * img_array[:, :, 0]
            + 0.7152 * img_array[:, :, 1]
            + 0.0722 * img_array[:, :, 2]
        )

        return luminance

    def calculate_lines(self, source_image: np.ndarray) -> List[int]:
        """Calculate the optimal sequence of lines."""
        print("Calculating string art lines...")

        # Initialize error image (inverted)
        error = 255.0 - source_image

        # Initialize line sequence
        line_sequence = [0]  # Start from pin 0
        current_pin = 0
        last_pins = np.full(20, -1, dtype=np.int32)  # Track recent pins
        last_pin_idx = 0

        for i in range(self.max_lines):
            if i % 100 == 0:
                print(f"  Processing line {i}/{self.max_lines}")

            best_pin = -1
            max_error = 0.0
            best_x_coords = None
            best_y_coords = None

            # Try all possible pins
            for offset in range(MIN_DISTANCE, self.pins - MIN_DISTANCE):
                test_pin = (current_pin + offset) % self.pins

                # Skip if recently used
                if contains_recent_pin(last_pins, test_pin):
                    continue

                # Get line coordinates
                x_coords, y_coords = self.line_cache[(current_pin, test_pin)]

                # Calculate error reduction
                line_error = get_line_error(error, x_coords, y_coords, self.target_size)

                if line_error > max_error:
                    max_error = line_error
                    best_pin = test_pin
                    best_x_coords = x_coords
                    best_y_coords = y_coords

            if best_pin == -1:
                # No valid pin found
                break

            # Add best pin to sequence
            line_sequence.append(best_pin)

            # Apply line to error image
            apply_line_to_error(
                error, best_x_coords, best_y_coords, self.line_weight, self.target_size
            )

            # Update recent pins tracking
            last_pins[last_pin_idx] = best_pin
            last_pin_idx = (last_pin_idx + 1) % len(last_pins)
            current_pin = best_pin

        print()
        return line_sequence

    def generate_output_image(self, line_sequence: List[int], output_file: str):
        """Generate the final string art image with anti-aliased lines using aggdraw."""
        print("Generating output image with anti-aliased lines...")

        # Create white image
        img = Image.new("RGB", (self.output_size, self.output_size), "white")

        # Create aggdraw canvas for anti-aliased drawing
        canvas = aggdraw.Draw(img)

        # Calculate scale factor
        scale = self.output_size / self.target_size

        # Create pens and brushes for drawing
        circle_pen = aggdraw.Pen("black", 1)
        pin_brush = aggdraw.Brush("black")

        # Calculate line opacity (0-255 range for aggdraw)
        line_opacity = min(255, max(1, self.output_weight))
        line_pen = aggdraw.Pen("rgba(0,0,0)", 1, line_opacity)

        # Draw circle border
        center = self.output_size // 2
        radius = self.output_size // 2 - 1
        canvas.ellipse(
            (center - radius, center - radius, center + radius, center + radius),
            circle_pen,
        )

        # Draw pins
        for coord in self.pin_coords:
            x = int(coord[0] * scale)
            y = int(coord[1] * scale)
            canvas.ellipse((x - 2, y - 2, x + 2, y + 2), circle_pen, pin_brush)

        # Draw anti-aliased lines
        print(f"  Drawing {len(line_sequence) - 1} anti-aliased lines...")
        for i in range(len(line_sequence) - 1):
            if i % 1000 == 0 and i > 0:
                print(f"    Drawing line {i}/{len(line_sequence) - 1}")

            from_pin = line_sequence[i]
            to_pin = line_sequence[i + 1]

            from_coord = self.pin_coords[from_pin]
            to_coord = self.pin_coords[to_pin]

            x0 = int(from_coord[0] * scale)
            y0 = int(from_coord[1] * scale)
            x1 = int(to_coord[0] * scale)
            y1 = int(to_coord[1] * scale)

            canvas.line((x0, y0, x1, y1), line_pen)

        # Flush the canvas to apply all drawing operations
        canvas.flush()

        # Save image
        img.save(output_file, "PNG")
        print(f"Output saved to {output_file}")

    def generate(self, input_file: str, output_file: str = "output.png"):
        """Generate string art from input image."""
        print(
            f"  Pins: {self.pins}, Max lines: {self.max_lines}, Processing size: {self.target_size}"
        )
        print(
            f"  Output size: {self.output_size}, Line weight: {self.line_weight}, Output weight: {self.output_weight}"
        )

        # Load and process image
        source_image = self.load_and_process_image(input_file)

        start_time = time.time()

        # Calculate pin coordinates
        self.pin_coords = calculate_pin_coords(self.pins, self.target_size)

        # Precalculate lines
        self.line_cache = precalculate_lines(self.pin_coords, self.pins)

        # Calculate optimal line sequence
        line_sequence = self.calculate_lines(source_image)

        end_time = time.time()
        print(f"Processing took {end_time - start_time:.2f} seconds")

        # Generate output image
        self.generate_output_image(line_sequence, output_file)


def main():
    parser = argparse.ArgumentParser(description="Generate string art from an image")
    parser.add_argument("--input", required=True, help="Input image file (JPEG or PNG)")
    parser.add_argument("--output", default="output.png", help="Output image file")
    parser.add_argument(
        "--pins", type=int, default=300, help="Number of pins around the circle"
    )
    parser.add_argument(
        "--lines", type=int, default=4000, help="Maximum number of lines to draw"
    )
    parser.add_argument(
        "--size",
        type=int,
        default=500,
        help="Processing size for input image in pixels",
    )
    parser.add_argument(
        "--output-size",
        type=int,
        help="Output image size in pixels (default: same as processing size)",
    )
    parser.add_argument(
        "--weight", type=int, default=8, help="Line weight for darkness calculation"
    )
    parser.add_argument(
        "--output-weight",
        type=int,
        help="Visual line opacity for output image (default: same as weight)",
    )

    args = parser.parse_args()

    generator = StringArtGenerator(
        pins=args.pins,
        max_lines=args.lines,
        target_size=args.size,
        output_size=args.output_size,
        line_weight=args.weight,
        output_weight=args.output_weight,
    )

    generator.generate(args.input, args.output)


if __name__ == "__main__":
    main()
