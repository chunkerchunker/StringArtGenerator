// Line cache generation - pre-compute all potential lines between pins
// Matches algorithm from main.go:180-219

export function generateLineCache(imgSize, pins, minDistance) {
    console.log(`Generating line cache: ${imgSize}px, ${pins} pins, min distance ${minDistance}`);

    // 1. Calculate pin coordinates (main.go:180-192)
    const pinCoords = [];
    const center = imgSize / 2;
    const radius = imgSize / 2 - 1;

    for (let i = 0; i < pins; i++) {
        const angle = 2 * Math.PI * i / pins;
        pinCoords.push({
            x: Math.floor(center + radius * Math.cos(angle)),
            y: Math.floor(center + radius * Math.sin(angle))
        });
    }

    // 2. Pre-calculate all potential lines (main.go:194-219)
    const allCoords = [];
    const metadata = new Uint32Array(pins * pins * 2); // [offset, length] pairs
    let currentOffset = 0;

    let linesGenerated = 0;

    for (let i = 0; i < pins; i++) {
        for (let j = i + minDistance; j < pins; j++) {
            const x0 = pinCoords[i].x;
            const y0 = pinCoords[i].y;
            const x1 = pinCoords[j].x;
            const y1 = pinCoords[j].y;

            // Calculate distance
            const dx = x1 - x0;
            const dy = y1 - y0;
            const distance = Math.floor(Math.sqrt(dx * dx + dy * dy));

            // Generate coordinates using linear interpolation (matches num.Linspace)
            // num.Linspace generates d evenly spaced points from start to end (inclusive)
            const coords = [];
            if (distance > 1) {
                for (let k = 0; k < distance; k++) {
                    const t = k / (distance - 1); // Include endpoint
                    const x = Math.floor(x0 + t * dx);
                    const y = Math.floor(y0 + t * dy);
                    coords.push(x, y);
                }
            } else {
                // If distance is 0 or 1, just use start point
                coords.push(Math.floor(x0), Math.floor(y0));
            }

            // Store in flat array
            const length = coords.length / 2;
            const startOffset = currentOffset;  // Offset in terms of coordinate pairs
            allCoords.push(...coords);
            currentOffset += length;  // Increment by number of pairs, not total values

            // Store metadata for both directions (i→j and j→i)
            const idx_ij = i * pins + j;
            const idx_ji = j * pins + i;
            metadata[idx_ij * 2] = startOffset;
            metadata[idx_ij * 2 + 1] = length;
            metadata[idx_ji * 2] = startOffset;
            metadata[idx_ji * 2 + 1] = length;

            linesGenerated++;
        }
    }

    console.log(`Generated ${linesGenerated} unique lines, ${allCoords.length / 2} total coordinates`);
    console.log(`Line cache size: ${(allCoords.length * 4 / 1024 / 1024).toFixed(2)} MB`);

    return {
        pinCoords: pinCoords,
        lineCoordBuffer: new Uint32Array(allCoords),
        lineMetadataBuffer: metadata
    };
}
