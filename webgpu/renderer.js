// Canvas rendering for string art visualization
// Matches rendering from main.go:304-351

export function renderStringArt(canvas, lineSequence, pinCoords, imgSize, outputSize) {
    const ctx = canvas.getContext('2d');

    // Set canvas size
    canvas.width = outputSize;
    canvas.height = outputSize;

    // Clear to white background
    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, outputSize, outputSize);

    // Calculate scale factor
    const scale = outputSize / imgSize;

    // Draw circle border (main.go:316-319)
    const centerOut = outputSize / 2;
    const radiusOut = outputSize / 2 - 1;
    ctx.strokeStyle = 'black';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.arc(centerOut, centerOut, radiusOut, 0, 2 * Math.PI);
    ctx.stroke();

    // Draw pins (main.go:321-326)
    ctx.fillStyle = 'black';
    for (const coord of pinCoords) {
        const scaledX = coord.x * scale;
        const scaledY = coord.y * scale;
        ctx.beginPath();
        ctx.arc(scaledX, scaledY, 2, 0, 2 * Math.PI);
        ctx.fill();
    }

    // Draw lines (main.go:328-339)
    // Use low alpha for cumulative darkening effect
    ctx.strokeStyle = 'rgba(0, 0, 0, 0.03)'; // ~8/255 opacity
    ctx.lineWidth = 1;

    for (let i = 0; i < lineSequence.length - 1; i++) {
        const fromIdx = lineSequence[i];
        const toIdx = lineSequence[i + 1];

        const from = pinCoords[fromIdx];
        const to = pinCoords[toIdx];

        const scaledFromX = from.x * scale;
        const scaledFromY = from.y * scale;
        const scaledToX = to.x * scale;
        const scaledToY = to.y * scale;

        ctx.beginPath();
        ctx.moveTo(scaledFromX, scaledFromY);
        ctx.lineTo(scaledToX, scaledToY);
        ctx.stroke();
    }
}
