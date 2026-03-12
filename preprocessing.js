// Webcam frame preprocessing - capture and convert to luminosity
// Matches algorithm from main.go:107-178

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
