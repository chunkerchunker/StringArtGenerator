// Webcam frame preprocessing - capture and convert to luminosity
// Matches algorithm from main.go:107-178

export function captureVideoFrame(videoElement, targetSize) {
    // Create temporary canvas for frame capture
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d', { willReadFrequently: true });

    const videoWidth = videoElement.videoWidth;
    const videoHeight = videoElement.videoHeight;

    // Crop to center square (main.go:135-142)
    const size = Math.min(videoWidth, videoHeight);
    const startX = (videoWidth - size) / 2;
    const startY = (videoHeight - size) / 2;

    // Resize to target size (main.go:144-148)
    canvas.width = targetSize;
    canvas.height = targetSize;
    ctx.drawImage(
        videoElement,
        startX, startY, size, size,  // Source rect (center square)
        0, 0, targetSize, targetSize  // Dest rect (resized)
    );

    return ctx.getImageData(0, 0, targetSize, targetSize);
}

export function convertToLuminosity(imageData) {
    // main.go:167-178 - ITU-R BT.709 luminosity formula
    const pixels = imageData.data;
    const width = imageData.width;
    const height = imageData.height;
    const luminosity = new Float32Array(width * height);

    for (let i = 0; i < luminosity.length; i++) {
        const r = pixels[i * 4];
        const g = pixels[i * 4 + 1];
        const b = pixels[i * 4 + 2];

        // Standard luminosity weights (ITU-R BT.709)
        luminosity[i] = 0.2126 * r + 0.7152 * g + 0.0722 * b;
    }

    return luminosity;
}

export function createErrorBuffer(luminosityArray) {
    // Initialize error buffer: 255.0 - luminosity (main.go:230)
    const errorBuffer = new Float32Array(luminosityArray.length);
    for (let i = 0; i < errorBuffer.length; i++) {
        errorBuffer[i] = 255.0 - luminosityArray[i];
    }
    return errorBuffer;
}
