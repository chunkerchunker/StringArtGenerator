from manim import *
import numpy as np
from PIL import Image

# ── Shared palette ──────────────────────────────────────────────
BG        = "#0f0f1a"
ACCENT    = "#ff6b35"
ACCENT2   = "#4ecdc4"
ACCENT3   = "#ffe66d"
SUBTLE    = "#555577"
PIN_COLOR = "#ff6b35"
LINE_CLR  = "#cccccc"
DARK_LINE = "#888888"

config.background_color = BG
config.pixel_width = 1920
config.pixel_height = 1080

class StringArtAlgorithm(Scene):
    def construct(self):
        # ---------------------------------------------------------
        # Configuration
        # ---------------------------------------------------------
        image_path = "mona_lisa.png"
        num_pins = 300         # Reduced for demo clarity (Go uses 300)
        max_lines = 4000        # Lines to draw in total
        min_distance = 10      # Min pin distance
        line_weight = 20       # How much to subtract per line
        
        # ---------------------------------------------------------
        # Part 1: Setup & Concept
        # ---------------------------------------------------------
        
        # Title
        title = Text("String Art Algorithm", font_size=48).to_edge(UP)
        self.play(Write(title))
        
        # 1. Show Original Image
        # Load and resize image for processing
        original_pil = Image.open(image_path).convert('L') # Grayscale
        original_pil = original_pil.resize((400, 400))     # Resize for calculation
        img_array = np.array(original_pil)
        
        # Display Image Mobject
        display_img = ImageMobject(image_path).set_height(5)
        self.play(FadeIn(display_img))
        self.wait(1)

        # 2. Explain "Error" Array (Inverted Image)
        # In the Go code: error = 255 - pixel value
        # Darker pixels = Higher numbers = Higher priority
        explanation = Text("Goal: Connect pins through darkest paths", font_size=24, color=YELLOW)
        explanation.next_to(display_img, DOWN)
        
        self.play(Write(explanation))
        self.wait(1)
        
        # Fade image to background opacity to prepare for drawing
        self.play(display_img.animate.set_opacity(0.3), FadeOut(explanation))

        # 3. Create Pins
        # Go code: Uses a circle of PINS
        circle = Circle(radius=2.5, color=WHITE, stroke_width=0.5, stroke_opacity=0.5)
        pins = VGroup()
        pin_coords = [] # Store (x,y) for calculation
        
        # Calculate pin locations
        center = np.array([0,0,0])
        radius = 2.5
        for i in range(num_pins):
            angle = 2 * np.pi * i / num_pins
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            p = Dot(point=np.array([x, y, 0]), radius=0.01, color=BLUE)
            pins.add(p)
            # Store pixel coordinates for the algorithm (mapping -2.5,2.5 to 0,400)
            img_x = int((x + radius) / (2 * radius) * 400)
            img_y = int((-y + radius) / (2 * radius) * 400) # Flip Y for image coords
            pin_coords.append((img_x, img_y))

        self.play(Create(circle), FadeIn(pins))
        self.wait(0.5)

        # ---------------------------------------------------------
        # Part 2: Algorithm Visualization
        # ---------------------------------------------------------
        
        # Run the actual Greedy Algorithm (Python port of your Go code)
        # We simulate the image error array
        error_array = 255 - img_array # Invert: Dark = High Value
        error_array = error_array.astype(float) # Float for subtraction
        
        current_pin = 0
        lines_drawn = VGroup()
        
        # Highlight start pin
        start_dot = Dot(pins[current_pin].get_center(), color=YELLOW, radius=0.04)
        self.play(Create(start_dot))
        
        info_text = Text("Greedy Search", font_size=30, color=YELLOW).to_corner(UL)
        score_text = Text("Score: ???", font_size=24).next_to(info_text, DOWN)
        self.play(FadeIn(info_text), FadeIn(score_text))

        # --- Visualize the logic for the first line ---
        best_pin = -1
        max_score = -1
        
        # Show scanning "rays"
        scan_lines = VGroup()
        
        # We will scan a few specific pins to demonstrate comparison
        demo_targets = [current_pin + 20, current_pin + 50, current_pin + 80]
        
        for t_pin in demo_targets:
            t_pin = t_pin % num_pins
            # Visual line
            temp_line = Line(pins[current_pin].get_center(), pins[t_pin].get_center(), color=RED, stroke_opacity=0.5)
            self.play(Create(temp_line), run_time=0.3)
            
            # Calculate mock score (Sum of pixels on line)
            # Bresenham-ish line sampling
            p0 = pin_coords[current_pin]
            p1 = pin_coords[t_pin]
            num_points = int(np.hypot(p1[0]-p0[0], p1[1]-p0[1]))
            xs = np.linspace(p0[0], p1[0], num_points).astype(int)
            ys = np.linspace(p0[1], p1[1], num_points).astype(int)
            
            # Clip coords
            xs = np.clip(xs, 0, 399)
            ys = np.clip(ys, 0, 399)
            
            score = np.sum(error_array[ys, xs])
            
            # Update score text
            new_score_text = Text(f"Score: {int(score)}", font_size=24).next_to(info_text, DOWN)
            self.play(Transform(score_text, new_score_text), run_time=0.2)
            self.play(FadeOut(temp_line), run_time=0.2)

        self.play(FadeOut(score_text))

        # ---------------------------------------------------------
        # Part 3: Fast Forward Calculation
        # ---------------------------------------------------------
        
        # Text update
        status = Text("Calculating Lines...", font_size=24).to_corner(UL)
        self.play(ReplacementTransform(info_text, status))

        # Run the full loop to generate line data
        line_sequence = []
        
        # Helper to get line coordinates
        def get_line_pixels(p0, p1):
            dist = int(np.hypot(p1[0]-p0[0], p1[1]-p0[1]))
            if dist == 0: return [], []
            xs = np.linspace(p0[0], p1[0], dist).astype(int)
            ys = np.linspace(p0[1], p1[1], dist).astype(int)
            return np.clip(xs, 0, 399), np.clip(ys, 0, 399)

        # Pre-calculate sequence for animation
        # (This mimics the Go 'calculateLines' function)
        prev_pins = [] # simple history to avoid immediate backtracking
        
        for _ in range(max_lines):
            best_score = -1
            best_candidate = -1
            best_path_xs, best_path_ys = None, None
            
            # Check all valid pins
            for offset in range(min_distance, num_pins - min_distance):
                candidate = (current_pin + offset) % num_pins
                
                # Simple loop prevention (Go uses 'contains' on last_pins)
                if len(prev_pins) > 0 and candidate == prev_pins[-1]:
                    continue
                
                xs, ys = get_line_pixels(pin_coords[current_pin], pin_coords[candidate])
                score = np.sum(error_array[ys, xs])
                
                if score > best_score:
                    best_score = score
                    best_candidate = candidate
                    best_path_xs, best_path_ys = xs, ys
            
            if best_candidate != -1:
                # Subtract line weight (Go: error[v] = error[v] - LINE_WEIGHT)
                error_array[best_path_ys, best_path_xs] -= line_weight
                error_array = np.clip(error_array, 0, 255) # Prevent negative
                
                line_sequence.append((current_pin, best_candidate))
                prev_pins.append(current_pin)
                current_pin = best_candidate
            else:
                break

        # ---------------------------------------------------------
        # Part 4: Animate the Result
        # ---------------------------------------------------------
        
        self.play(FadeOut(status))
        
        # Slow draw first 20 lines
        for i in range(min(20, len(line_sequence))):
            start, end = line_sequence[i]
            l = Line(pins[start].get_center(), pins[end].get_center(), color=WHITE, stroke_width=0.5, stroke_opacity=1)
            self.play(Create(l), run_time=0.2)
            lines_drawn.add(l)

        # Remove the slow-drawn lines before fast drawing
        self.play(FadeOut(lines_drawn), run_time=0.5)
        lines_drawn = VGroup()  # Reset

        # Fast draw all lines
        fast_group = VGroup()
        for i in range(len(line_sequence)):
            start, end = line_sequence[i]
            l = Line(pins[start].get_center(), pins[end].get_center(), color=WHITE, stroke_width=0.2, stroke_opacity=0.2)
            fast_group.add(l)
            
        self.play(Create(fast_group), run_time=4, rate_func=linear)
        
        # Remove original image to show string art clearly
        self.play(FadeOut(display_img), FadeOut(circle), FadeOut(pins), FadeOut(start_dot))
        
        final_text = Text("Final String Art Approximation", font_size=32).to_edge(DOWN)
        self.play(Write(final_text))
        
        self.wait(2)