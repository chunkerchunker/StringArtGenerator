# String Art Generator (Python)

A high-performance Python implementation of the string art generator, optimized for speed on macOS using NumPy and Numba.

## Installation

```bash
uv sync
```

## Usage

```bash
# Basic usage
uv run string_art.py --input input.jpg --output output.png

# With custom parameters  
uv run string_art.py --input photo.jpg --output art.png --pins 400 --lines 5000 --size 600 --weight 10
```

## Performance Features

- NumPy vectorized operations
- Numba JIT compilation for critical functions
- Efficient line drawing with Bresenham's algorithm
- Smart caching of line coordinates
- Anti-aliased output lines using aggdraw for smooth, professional results

## Dependencies

- **Python 3.12+**: For best performance
- **NumPy**: Vectorized array operations
- **Pillow**: Image processing
- **Numba**: JIT compilation for speed
- **aggdraw**: Anti-aliased drawing for smooth lines

