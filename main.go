package main

import (
	"flag"
	"fmt"
	"image"
	"image/color"
	"image/jpeg"
	"image/png"
	"io"
	"log"
	"math"

	"os"
	"time"

	ximgdraw "golang.org/x/image/draw"

	"github.com/antha-lang/antha/antha/anthalib/num"
)

// **********************//
//
//	Structs			//
//
// **********************//
type Coord struct {
	X float64
	Y float64
}

const MIN_DISTANCE = 30

// These will be set from command-line flags or defaults
var PINS = 300
var MAX_LINES = 4000
var TARGET_SIZE = 500
var LINE_WEIGHT = 8

// These are calculated based on actual image size
var IMG_SIZE = 500
var IMG_SIZE_FL = float64(500)
var IMG_SIZE_SQ = 250000
var Pin_coords = []Coord{}
var SourceImage = []float64{}
var Line_cache_y = [][]float64{}
var Line_cache_x = [][]float64{}

//**********************//
//		Main			//
//**********************//

func init() {
	image.RegisterFormat("jpeg", "jpeg", jpeg.Decode, jpeg.DecodeConfig)
	image.RegisterFormat("png", "png", png.Decode, png.DecodeConfig)
}

func main() {
	inputFile := flag.String("input", "", "Input image file (JPEG or PNG)")
	outputFile := flag.String("output", "output.png", "Output image file")
	pinsFlag := flag.Int("pins", 300, "Number of pins around the circle (default: 300)")
	maxLinesFlag := flag.Int("lines", 4000, "Maximum number of lines to draw (default: 4000)")
	targetSizeFlag := flag.Int("size", 500, "Target image size in pixels (default: 500)")
	lineWeightFlag := flag.Int("weight", 8, "Line weight for darkness calculation (default: 8, higher = darker)")
	flag.Parse()

	if *inputFile == "" {
		log.Fatal("Please provide an input file using -input flag")
	}

	// Set global variables from flags
	PINS = *pinsFlag
	MAX_LINES = *maxLinesFlag
	TARGET_SIZE = *targetSizeFlag
	LINE_WEIGHT = *lineWeightFlag

	fmt.Printf("Processing %s...\n", *inputFile)
	fmt.Printf("  Pins: %d, Max lines: %d, Target size: %d, Line weight: %d\n", PINS, MAX_LINES, TARGET_SIZE, LINE_WEIGHT)
	SourceImage = importPictureAndGetPixelArray(*inputFile)

	startTime := time.Now()
	calculatePinCoords()
	precalculateAllPotentialLines()
	lineSequence := calculateLines()
	endTime := time.Now()
	diff := endTime.Sub(startTime)
	fmt.Printf("Processing took %.2f seconds\n", diff.Seconds())

	fmt.Printf("Generating output image...\n")
	generateOutputImage(lineSequence, *outputFile)
	fmt.Printf("Output saved to %s\n", *outputFile)
}

func importPictureAndGetPixelArray(filename string) []float64 {
	imgfile, err := os.Open(filename)
	if err != nil {
		log.Fatalf("Error opening image file: %v", err)
	}
	defer imgfile.Close()

	pixels, err := getPixels(imgfile)
	if err != nil {
		log.Fatalf("Error processing image: %v", err)
	}
	return pixels
}

func getPixels(file io.Reader) ([]float64, error) {
	img, _, err := image.Decode(file)
	if err != nil {
		return nil, err
	}

	// Resize image if needed
	bounds := img.Bounds()
	origWidth, origHeight := bounds.Max.X, bounds.Max.Y

	// Scale to fit within TARGET_SIZE while maintaining aspect ratio
	if origWidth != TARGET_SIZE || origHeight != TARGET_SIZE {
		// For string art, we need a square image
		// First, crop to square if needed
		size := min(origHeight, origWidth)

		// Crop to center square
		startX := (origWidth - size) / 2
		startY := (origHeight - size) / 2
		croppedImg := img.(interface {
			SubImage(r image.Rectangle) image.Image
		}).SubImage(image.Rect(startX, startY, startX+size, startY+size))

		// Now resize to target size
		dst := image.NewRGBA(image.Rect(0, 0, TARGET_SIZE, TARGET_SIZE))
		ximgdraw.BiLinear.Scale(dst, dst.Bounds(), croppedImg, croppedImg.Bounds(), ximgdraw.Over, nil)
		img = dst
		fmt.Printf("  Resized from %dx%d to %dx%d\n", origWidth, origHeight, TARGET_SIZE, TARGET_SIZE)
	}

	bounds = img.Bounds()
	width, height := bounds.Max.X, bounds.Max.Y
	IMG_SIZE = width
	IMG_SIZE_FL = float64(IMG_SIZE)
	IMG_SIZE_SQ = IMG_SIZE * IMG_SIZE

	var pixels []float64
	for y := range height {
		for x := range width {
			pixels = append(pixels, rgbaToPixel(img.At(x, y).RGBA()))
		}
	}

	return pixels, nil
}

func rgbaToPixel(r uint32, g uint32, b uint32, _ uint32) float64 {
	// Convert to 8-bit values
	r8 := float64(r / 257)
	g8 := float64(g / 257)
	b8 := float64(b / 257)

	// Calculate luminosity using standard weights (ITU-R BT.709)
	// Human eye is more sensitive to green than red or blue
	luminosity := 0.2126*r8 + 0.7152*g8 + 0.0722*b8

	return luminosity
}

func calculatePinCoords() {
	pin_coords := make([]Coord, PINS)

	center := float64(IMG_SIZE / 2)
	radius := float64(IMG_SIZE/2 - 1)

	for i := range PINS {
		angle := 2 * math.Pi * float64(i) / float64(PINS)
		pin_coords[i] = Coord{X: math.Floor(center + radius*math.Cos(angle)), Y: math.Floor(center + radius*math.Sin(angle))}
	}

	Pin_coords = pin_coords[:]
}

func precalculateAllPotentialLines() {
	cacheSize := PINS * PINS
	line_cache_y := make([][]float64, cacheSize)
	line_cache_x := make([][]float64, cacheSize)

	for i := range PINS {
		for j := i + MIN_DISTANCE; j < PINS; j++ {
			x0 := Pin_coords[i].X
			y0 := Pin_coords[i].Y

			x1 := Pin_coords[j].X
			y1 := Pin_coords[j].Y

			d := math.Floor(math.Sqrt(float64((x1-x0)*(x1-x0) + (y1-y0)*(y1-y0))))
			xs := roundUpFloatArrayToInt(num.Linspace(float64(x0), float64(x1), int(d)))
			ys := roundUpFloatArrayToInt(num.Linspace(float64(y0), float64(y1), int(d)))

			line_cache_y[j*PINS+i] = ys
			line_cache_y[i*PINS+j] = ys
			line_cache_x[j*PINS+i] = xs
			line_cache_x[i*PINS+j] = xs
		}
	}
	Line_cache_y = line_cache_y[:][:]
	Line_cache_x = line_cache_x[:][:]
}

func roundUpFloatArrayToInt(arr []float64) []float64 {
	for i := range arr {
		arr[i] = float64(int(arr[i]))
	}
	return arr
}

func calculateLines() []int {
	fmt.Println("Calculating string art lines...")
	error := num.Sub(num.MulByConst(num.Ones(IMG_SIZE_SQ), float64(255)), SourceImage)

	line_sequence := make([]int, 0, MAX_LINES)
	line_sequence = append(line_sequence, 0) // Start from pin 0
	current_pin := 0
	last_pins := make([]int, 20, 24)
	best_pin := -1
	line_err := float64(0)
	max_err := float64(0)
	index := 0
	inner_index := 0
	for i := 0; i < MAX_LINES; i++ {
		if i%100 == 0 {
			fmt.Printf("  Processing line %d/%d\r", i, MAX_LINES)
		}
		best_pin = -1
		line_err = float64(0)
		max_err = float64(0)

		for offset := MIN_DISTANCE; offset < PINS-MIN_DISTANCE; offset++ {
			test_pin := (current_pin + offset) % PINS
			if contains(last_pins, test_pin) {
				continue
			} else {
				inner_index = test_pin*PINS + current_pin

				line_err = getLineErr(error, Line_cache_y[inner_index], Line_cache_x[inner_index])
				if line_err > max_err {
					max_err = line_err
					best_pin = test_pin
					index = inner_index
				}
			}
		}

		if best_pin == -1 {
			// No valid pin found, stop
			break
		}

		line_sequence = append(line_sequence, best_pin)

		coords1 := Line_cache_y[index]
		coords2 := Line_cache_x[index]
		for i := range coords1 {
			v := int((coords1[i] * IMG_SIZE_FL) + coords2[i])
			error[v] = error[v] - float64(LINE_WEIGHT)
		}

		last_pins = append(last_pins, best_pin)
		last_pins = last_pins[1:]
		current_pin = best_pin
	}
	fmt.Printf("\n")
	return line_sequence
}

func getLineErr(err, coords1, coords2 []float64) float64 {
	sum := float64(0)
	for i := range coords1 {
		sum = sum + err[int((coords1[i]*IMG_SIZE_FL)+coords2[i])]
	}
	return sum / float64(len(coords1))
}

func contains(arr []int, num int) bool {
	for i := range arr {
		if arr[i] == num {
			return true
		}
	}
	return false
}

func generateOutputImage(lineSequence []int, outputFile string) {
	// Create a new white image
	img := image.NewRGBA(image.Rect(0, 0, IMG_SIZE, IMG_SIZE))
	for y := 0; y < IMG_SIZE; y++ {
		for x := 0; x < IMG_SIZE; x++ {
			img.Set(x, y, color.White)
		}
	}

	// Draw the circle border
	drawCircle(img, IMG_SIZE/2, IMG_SIZE/2, IMG_SIZE/2-1, color.Black)

	// Draw the pins
	for _, coord := range Pin_coords {
		drawFilledCircle(img, int(coord.X), int(coord.Y), 2, color.Black)
	}

	// Draw the lines
	for i := 0; i < len(lineSequence)-1; i++ {
		from := Pin_coords[lineSequence[i]]
		to := Pin_coords[lineSequence[i+1]]
		drawLine(img, int(from.X), int(from.Y), int(to.X), int(to.Y), color.RGBA{0, 0, 0, uint8(LINE_WEIGHT)})
	}

	// Save the image
	file, err := os.Create(outputFile)
	if err != nil {
		log.Fatalf("Error creating output file: %v", err)
	}
	defer file.Close()

	if err := png.Encode(file, img); err != nil {
		log.Fatalf("Error encoding image: %v", err)
	}
}

func drawLine(img *image.RGBA, x0, y0, x1, y1 int, c color.Color) {
	dx := abs(x1 - x0)
	dy := abs(y1 - y0)
	sx := 1
	sy := 1
	if x0 > x1 {
		sx = -1
	}
	if y0 > y1 {
		sy = -1
	}
	err := dx - dy

	// Get the line color with alpha
	lineR, lineG, lineB, lineA := c.RGBA()

	for {
		// Blend the line color with the existing pixel
		existingColor := img.RGBAAt(x0, y0)

		// Alpha blending formula: result = src * alpha + dst * (1 - alpha)
		alpha := float64(lineA) / 0xffff
		invAlpha := 1.0 - alpha

		newR := uint8((float64(lineR>>8) * alpha) + (float64(existingColor.R) * invAlpha))
		newG := uint8((float64(lineG>>8) * alpha) + (float64(existingColor.G) * invAlpha))
		newB := uint8((float64(lineB>>8) * alpha) + (float64(existingColor.B) * invAlpha))

		img.SetRGBA(x0, y0, color.RGBA{newR, newG, newB, 255})

		if x0 == x1 && y0 == y1 {
			break
		}
		e2 := 2 * err
		if e2 > -dy {
			err -= dy
			x0 += sx
		}
		if e2 < dx {
			err += dx
			y0 += sy
		}
	}
}

func drawCircle(img *image.RGBA, cx, cy, r int, c color.Color) {
	for angle := 0.0; angle < 2*math.Pi; angle += 0.01 {
		x := cx + int(float64(r)*math.Cos(angle))
		y := cy + int(float64(r)*math.Sin(angle))
		img.Set(x, y, c)
	}
}

func drawFilledCircle(img *image.RGBA, cx, cy, r int, c color.Color) {
	for x := -r; x <= r; x++ {
		for y := -r; y <= r; y++ {
			if x*x+y*y <= r*r {
				img.Set(cx+x, cy+y, c)
			}
		}
	}
}

func abs(x int) int {
	if x < 0 {
		return -x
	}
	return x
}
