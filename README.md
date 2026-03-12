# String Art Generator

This project generates string art images by simulating wrapping thread around pins on a circular frame. Multiple implementations were explored, progressing from a Go reference implementation through C/SIMD optimizations, WASM for browser execution, and finally a pure WebGPU GPU-accelerated version.

Originally based on [this repo](https://github.com/halfmonty/StringArtGenerator), which itself was based on work of reddit user [/u/kmmeerts](https://www.reddit.com/r/DIY/comments/au0ilz/).

See [IMPLEMENTATIONS.md](doc/IMPLEMENTATIONS.md) for details.
