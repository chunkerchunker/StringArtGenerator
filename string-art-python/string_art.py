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


@jit(nopython=True)
def precalculate_lines_fast(pin_coords: np.ndarray, pins: int, max_line_length: int):
    """JIT-compiled line precalculation for better performance."""
    line_coords_x = np.full((pins, pins, max_line_length), -1, dtype=np.int32)
    line_coords_y = np.full((pins, pins, max_line_length), -1, dtype=np.int32)
    line_lengths = np.zeros((pins, pins), dtype=np.int32)

    for i in range(pins):
        for j in range(i + MIN_DISTANCE, pins):
            x0 = int(pin_coords[i, 0])
            y0 = int(pin_coords[i, 1])
            x1 = int(pin_coords[j, 0])
            y1 = int(pin_coords[j, 1])

            xs, ys = bresenham_line(x0, y0, x1, y1)
            length = len(xs)

            # Store coordinates for both directions
            line_coords_x[i, j, :length] = xs
            line_coords_y[i, j, :length] = ys
            line_coords_x[j, i, :length] = xs
            line_coords_y[j, i, :length] = ys
            line_lengths[i, j] = length
            line_lengths[j, i] = length

    return line_coords_x, line_coords_y, line_lengths


def precalculate_lines(pin_coords: np.ndarray, pins: int):
    """Precalculate all potential lines between pins using optimized JIT function."""
    print("Precalculating potential lines...")

    # Conservative estimate for maximum line length
    max_line_length = int(np.sqrt(2) * 1000) + 10

    # Use JIT-compiled function for the heavy lifting
    line_coords_x, line_coords_y, line_lengths = precalculate_lines_fast(
        pin_coords, pins, max_line_length
    )

    return line_coords_x, line_coords_y, line_lengths


@jit(nopython=True)
def get_line_error_fast(
    error_img: np.ndarray,
    line_coords_x: np.ndarray,
    line_coords_y: np.ndarray,
    from_pin: int,
    to_pin: int,
    line_length: int,
    img_size: int,
) -> float:
    """Calculate the error reduction for a line using precomputed coordinates."""
    total_error = 0.0
    valid_points = 0

    for i in range(line_length):
        x = line_coords_x[from_pin, to_pin, i]
        y = line_coords_y[from_pin, to_pin, i]
        if 0 <= x < img_size and 0 <= y < img_size:
            total_error += error_img[y, x]
            valid_points += 1

    return total_error / valid_points if valid_points > 0 else 0.0  # type: ignore


@jit(nopython=True)
def apply_line_to_error_fast(
    error_img: np.ndarray,
    line_coords_x: np.ndarray,
    line_coords_y: np.ndarray,
    from_pin: int,
    to_pin: int,
    line_length: int,
    line_weight: int,
    img_size: int,
):
    """Apply a line to the error image using precomputed coordinates."""
    for i in range(line_length):
        x = line_coords_x[from_pin, to_pin, i]
        y = line_coords_y[from_pin, to_pin, i]
        if 0 <= x < img_size and 0 <= y < img_size:
            error_img[y, x] -= line_weight


@jit(nopython=True)
def calculate_lines_fast(
    error_img: np.ndarray,
    line_coords_x: np.ndarray,
    line_coords_y: np.ndarray,
    line_lengths: np.ndarray,
    pins: int,
    max_lines: int,
    line_weight: int,
    img_size: int,
) -> np.ndarray:
    """JIT-compiled version of the line calculation algorithm."""
    line_sequence = np.zeros(max_lines + 1, dtype=np.int32)
    line_sequence[0] = 0  # Start from pin 0
    sequence_length = 1

    current_pin = 0
    last_pins = np.full(20, -1, dtype=np.int32)
    last_pin_idx = 0

    for line_idx in range(max_lines):
        best_pin = -1
        max_error = 0.0

        # Try all possible pins
        for offset in range(MIN_DISTANCE, pins - MIN_DISTANCE):
            test_pin = (current_pin + offset) % pins

            # Skip if recently used
            recently_used = False
            for j in range(20):
                if last_pins[j] == test_pin:
                    recently_used = True
                    break
            if recently_used:
                continue

            # Skip if no line exists
            length = line_lengths[current_pin, test_pin]
            if length == 0:
                continue

            # Calculate error reduction
            line_error = get_line_error_fast(
                error_img,
                line_coords_x,
                line_coords_y,
                current_pin,
                test_pin,
                length,
                img_size,
            )

            if line_error > max_error:
                max_error = line_error
                best_pin = test_pin

        if best_pin == -1:
            break

        # Add best pin to sequence
        line_sequence[sequence_length] = best_pin
        sequence_length += 1

        # Apply line to error image
        length = line_lengths[current_pin, best_pin]
        apply_line_to_error_fast(
            error_img,
            line_coords_x,
            line_coords_y,
            current_pin,
            best_pin,
            length,
            line_weight,
            img_size,
        )

        # Update recent pins tracking
        last_pins[last_pin_idx] = best_pin
        last_pin_idx = (last_pin_idx + 1) % 20
        current_pin = best_pin

    return line_sequence[:sequence_length]


def warmup_jit_functions():
    """Warm up JIT-compiled functions to avoid compilation overhead during processing."""
    # Create minimal dummy data to trigger compilation
    dummy_coords = np.array([[5, 5], [10, 10]], dtype=np.float64)
    dummy_error = np.ones((15, 15), dtype=np.float64)
    dummy_line_coords_x = np.zeros((2, 2, 10), dtype=np.int32)
    dummy_line_coords_y = np.zeros((2, 2, 10), dtype=np.int32)
    dummy_lengths = np.array([[0, 5], [5, 0]], dtype=np.int32)

    # Trigger JIT compilation with minimal data
    calculate_pin_coords(2, 15)
    precalculate_lines_fast(dummy_coords, 2, 10)
    calculate_lines_fast(
        dummy_error,
        dummy_line_coords_x,
        dummy_line_coords_y,
        dummy_lengths,
        2,
        2,
        1,
        15,
    )


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
        self.line_coords_x = None
        self.line_coords_y = None
        self.line_lengths = None

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
        """Calculate the optimal sequence of lines using optimized JIT functions."""
        print("Calculating string art lines...")

        # Initialize error image (inverted)
        error = 255.0 - source_image

        # Use the fast JIT-compiled algorithm
        line_sequence_array = calculate_lines_fast(
            error,
            self.line_coords_x,
            self.line_coords_y,
            self.line_lengths,
            self.pins,
            self.max_lines,
            self.line_weight,
            self.target_size,
        )

        print()
        return line_sequence_array.tolist()

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

        # Draw anti-aliased lines in batches for better performance
        print(f"  Drawing {len(line_sequence) - 1} anti-aliased lines...")

        # Pre-compute all line coordinates to reduce per-line overhead
        num_lines = len(line_sequence) - 1
        line_coords = np.zeros((num_lines, 4), dtype=np.int32)

        for i in range(num_lines):
            from_pin = line_sequence[i]
            to_pin = line_sequence[i + 1]

            from_coord = self.pin_coords[from_pin]
            to_coord = self.pin_coords[to_pin]

            line_coords[i, 0] = int(from_coord[0] * scale)  # x0
            line_coords[i, 1] = int(from_coord[1] * scale)  # y0
            line_coords[i, 2] = int(to_coord[0] * scale)  # x1
            line_coords[i, 3] = int(to_coord[1] * scale)  # y1

        # Draw all lines in batches to reduce overhead
        batch_size = 5000
        for batch_start in range(0, num_lines, batch_size):
            batch_end = min(batch_start + batch_size, num_lines)
            if batch_start > 0:
                print(f"    Drawing lines {batch_start}-{batch_end}/{num_lines}")

            for i in range(batch_start, batch_end):
                canvas.line(
                    (
                        line_coords[i, 0],
                        line_coords[i, 1],
                        line_coords[i, 2],
                        line_coords[i, 3],
                    ),
                    line_pen,
                )

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

        # Warm up JIT functions to avoid compilation overhead during timing
        print("Warming up JIT functions...")
        warmup_jit_functions()

        # Load and process image
        source_image = self.load_and_process_image(input_file)

        start_time = time.time()

        # Calculate pin coordinates
        self.pin_coords = calculate_pin_coords(self.pins, self.target_size)

        # Precalculate lines
        self.line_coords_x, self.line_coords_y, self.line_lengths = precalculate_lines(
            self.pin_coords, self.pins
        )

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
