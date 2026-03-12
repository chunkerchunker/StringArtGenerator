from manim import *
import numpy as np
from PIL import Image as PILImage
import math

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

# ── Helpers ─────────────────────────────────────────────────────
# Maximum safe width for content (frame is ~14.2 wide)
MAX_WIDTH = 13.0

def clamp_width(mob, max_width=MAX_WIDTH):
    """Scale mobject down only if it exceeds max_width (preserves kerning)."""
    if mob.width > max_width:
        mob.scale(max_width / mob.width)
    return mob

def scaled_text(text, font_size, color=WHITE, **kwargs):
    """Create text at 2x size and scale down for better kerning."""
    return Text(text, font_size=font_size * 2, color=color, **kwargs).scale(0.5)

def section_title(text, subtitle=None):
    """Create a styled section title group."""
    title = scaled_text(text, 48, color=WHITE, weight=BOLD)
    clamp_width(title)
    if subtitle:
        sub = scaled_text(subtitle, 24, color=SUBTLE)
        clamp_width(sub)
        grp = VGroup(title, sub).arrange(DOWN, buff=0.3)
        return grp
    return title

def pin_on_circle(center, radius, index, total_pins):
    angle = 2 * PI * index / total_pins
    return center + radius * np.array([np.cos(angle), np.sin(angle), 0])


# ═══════════════════════════════════════════════════════════════
#  SCENE 1 — Title & Overview
# ═══════════════════════════════════════════════════════════════
class Scene01_Title(Scene):
    def construct(self):
        title = scaled_text("String Art Algorithm", 64, color=WHITE, weight=BOLD)
        subtitle = scaled_text("How a computer turns a photo into thread art", 28, color=ACCENT2)
        VGroup(title, subtitle).arrange(DOWN, buff=0.4).move_to(ORIGIN)

        # decorative pins & thread
        circle = Circle(radius=2.8, color=SUBTLE, stroke_width=1).shift(RIGHT * 0)
        pins_vg = VGroup()
        n_pins = 60
        for i in range(n_pins):
            p = pin_on_circle(ORIGIN, 2.8, i, n_pins)
            dot = Dot(p, radius=0.03, color=PIN_COLOR)
            pins_vg.add(dot)

        # A few decorative lines
        thread_lines = VGroup()
        np.random.seed(42)
        for _ in range(80):
            a, b = np.random.choice(n_pins, 2, replace=False)
            pa = pin_on_circle(ORIGIN, 2.8, a, n_pins)
            pb = pin_on_circle(ORIGIN, 2.8, b, n_pins)
            thread_lines.add(Line(pa, pb, stroke_width=0.4, stroke_opacity=0.15, color=WHITE))

        bg_art = VGroup(circle, pins_vg, thread_lines).set_opacity(0.3)

        self.play(FadeIn(bg_art, run_time=1))
        self.play(Write(title), run_time=1.2)
        self.play(FadeIn(subtitle, shift=UP * 0.2), run_time=0.8)
        self.wait(1.5)
        self.play(FadeOut(Group(*self.mobjects)), run_time=0.8)


# ═══════════════════════════════════════════════════════════════
#  SCENE 2 — The Concept: Physical string art
# ═══════════════════════════════════════════════════════════════
class Scene02_Concept(Scene):
    def construct(self):
        header = section_title("The Concept")
        header.to_edge(UP, buff=0.5)
        self.play(Write(header), run_time=0.8)

        # Left: description (keep left of center to avoid circle)
        steps = VGroup(
            scaled_text("1. Place pins around a circular frame", 22),
            scaled_text("2. Stretch dark thread between pins", 22),
            scaled_text("3. Thread overlaps → darker regions", 22),
            scaled_text("4. Algorithm chooses which pins to", 22),
            scaled_text("   connect to reproduce a photo", 22),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.25).shift(LEFT * 3.5 + DOWN * 0.3)

        # Right: build up a small string art demo
        center = RIGHT * 2.5 + DOWN * 0.3
        radius = 2.2
        n = 36
        circle = Circle(radius=radius, color=SUBTLE, stroke_width=1.5).move_to(center)
        pins = VGroup()
        for i in range(n):
            p = pin_on_circle(center, radius, i, n)
            pins.add(Dot(p, radius=0.05, color=PIN_COLOR))

        self.play(Create(circle), run_time=0.6)
        self.play(FadeIn(pins), run_time=0.5)
        self.play(FadeIn(steps[0], shift=RIGHT * 0.2), run_time=0.5)
        self.wait(0.3)

        # Animate some thread lines
        np.random.seed(7)
        sequence = [0]
        cur = 0
        for _ in range(50):
            nxt = (cur + np.random.randint(8, n - 8)) % n
            sequence.append(nxt)
            cur = nxt

        thread_lines = VGroup()
        for k in range(len(sequence) - 1):
            pa = pin_on_circle(center, radius, sequence[k], n)
            pb = pin_on_circle(center, radius, sequence[k + 1], n)
            thread_lines.add(Line(pa, pb, stroke_width=0.8, stroke_opacity=0.35, color=WHITE))

        self.play(FadeIn(steps[1], shift=RIGHT * 0.2), run_time=0.4)
        self.play(LaggedStart(*[Create(l) for l in thread_lines[:15]], lag_ratio=0.08), run_time=1.5)
        self.play(FadeIn(steps[2], shift=RIGHT * 0.2), run_time=0.4)
        self.play(LaggedStart(*[Create(l) for l in thread_lines[15:]], lag_ratio=0.04), run_time=1.5)
        self.play(FadeIn(steps[3], shift=RIGHT * 0.2), FadeIn(steps[4], shift=RIGHT * 0.2), run_time=0.5)
        self.wait(1.5)
        self.play(FadeOut(Group(*self.mobjects)), run_time=0.8)


# ═══════════════════════════════════════════════════════════════
#  SCENE 3 — Step 1: Load & prepare the image
# ═══════════════════════════════════════════════════════════════
class Scene03_ImagePrep(Scene):
    def construct(self):
        header = section_title("Step 1: Prepare the Image")
        header.to_edge(UP, buff=0.5)
        self.play(Write(header), run_time=0.8)

        # Load real image (compact layout to fit screen)
        img_mob = ImageMobject("mona_lisa.png").scale_to_fit_height(2.8)
        img_mob.shift(LEFT * 5.2 + DOWN * 0.4)
        label1 = scaled_text("Original", 20, color=SUBTLE).next_to(img_mob, DOWN, buff=0.2)
        self.play(FadeIn(img_mob), FadeIn(label1), run_time=0.8)
        self.wait(0.5)

        # Arrow
        arrow1 = Arrow(LEFT * 3.5 + DOWN * 0.4, LEFT * 2.3 + DOWN * 0.4, color=ACCENT, stroke_width=3)
        step1_text = scaled_text("Crop & resize", 18, color=ACCENT2).next_to(arrow1, UP, buff=0.15)
        self.play(GrowArrow(arrow1), FadeIn(step1_text), run_time=0.6)

        # Squared image with frame
        img_sq = ImageMobject("mona_lisa.png").scale_to_fit_height(2.6)
        img_sq.scale_to_fit_width(2.6)
        img_sq.move_to(LEFT * 0.8 + DOWN * 0.4)
        sq_frame = Square(side_length=2.6, color=ACCENT, stroke_width=2).move_to(img_sq)
        label_sq = scaled_text("500 × 500", 20, color=SUBTLE).next_to(img_sq, DOWN, buff=0.2)
        self.play(FadeIn(img_sq), Create(sq_frame), FadeIn(label_sq), run_time=0.8)
        self.wait(0.5)

        # Arrow to grayscale
        arrow2 = Arrow(RIGHT * 0.8 + DOWN * 0.4, RIGHT * 2.0 + DOWN * 0.4, color=ACCENT, stroke_width=3)
        step2_text = scaled_text("Grayscale", 18, color=ACCENT2).next_to(arrow2, UP, buff=0.15)
        self.play(GrowArrow(arrow2), FadeIn(step2_text), run_time=0.6)

        # Grayscale representation (use a gradient square)
        gray_sq = Square(side_length=2.6, fill_opacity=1, fill_color=GREY_D, stroke_color=ACCENT, stroke_width=2)
        gray_sq.move_to(RIGHT * 4.0 + DOWN * 0.4)
        gray_label = scaled_text("Luminosity", 18, color=SUBTLE).next_to(gray_sq, DOWN, buff=0.2)

        # Create pixel grid overlay
        grid = VGroup()
        gs = 2.6
        n = 8
        cs = gs / n
        for i in range(n):
            for j in range(n):
                val = 0.3 + 0.5 * np.sin(i / n * PI) * np.sin(j / n * PI)
                c = interpolate_color(BLACK, WHITE, val)
                sq = Square(side_length=cs, fill_opacity=1, fill_color=c, stroke_width=0.5, stroke_color=SUBTLE)
                sq.move_to(gray_sq.get_corner(UL) + RIGHT * (j + 0.5) * cs + DOWN * (i + 0.5) * cs)
                grid.add(sq)

        self.play(FadeIn(grid), FadeIn(gray_label), run_time=0.8)

        # Formula
        formula = MathTex(
            r"\text{lum} = 0.2126 \cdot R + 0.7152 \cdot G + 0.0722 \cdot B",
            font_size=28, color=ACCENT3
        ).to_edge(DOWN, buff=0.5)
        self.play(Write(formula), run_time=0.8)
        self.wait(1.5)
        self.play(FadeOut(Group(*self.mobjects)), run_time=0.8)


# ═══════════════════════════════════════════════════════════════
#  SCENE 4 — Step 2: Compute the error image
# ═══════════════════════════════════════════════════════════════
class Scene04_ErrorImage(Scene):
    def construct(self):
        header = section_title("Step 2: Compute the Error Image")
        header.to_edge(UP, buff=0.5)
        self.play(Write(header), run_time=0.8)

        # Show luminosity grid
        n = 10
        gs = 3.0
        cs = gs / n

        np.random.seed(99)
        # Simulate a face-like luminosity pattern
        lum_vals = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                cx, cy = n / 2, n / 2
                d = np.sqrt((i - cy) ** 2 + (j - cx) ** 2) / (n / 2)
                lum_vals[i][j] = max(0, min(1, 0.7 - 0.4 * d + 0.15 * np.sin(i) * np.cos(j)))

        def make_grid(vals, pos, clr_func=lambda v: interpolate_color(BLACK, WHITE, v)):
            grp = VGroup()
            for i in range(n):
                for j in range(n):
                    c = clr_func(vals[i][j])
                    sq = Square(side_length=cs, fill_opacity=1, fill_color=c, stroke_width=0.3, stroke_color=SUBTLE)
                    sq.move_to(pos + RIGHT * (j - n / 2 + 0.5) * cs + DOWN * (i - n / 2 + 0.5) * cs)
                    grp.add(sq)
            return grp

        lum_grid = make_grid(lum_vals, LEFT * 3.5 + DOWN * 0.3)
        lum_label = scaled_text("Luminosity", 20, color=SUBTLE).next_to(lum_grid, DOWN, buff=0.2)
        self.play(FadeIn(lum_grid), FadeIn(lum_label), run_time=0.6)

        # Formula and arrow in center
        formula = MathTex(r"\text{error}[i] = 255 - \text{luminosity}[i]", font_size=30, color=ACCENT3)
        formula.move_to(DOWN * 0.1)

        # Single arrow in the middle
        arrow = Arrow(LEFT * 0.6 + DOWN * 0.55, RIGHT * 0.6 + DOWN * 0.55, color=ACCENT, stroke_width=5)
        self.play(GrowArrow(arrow), Write(formula), run_time=0.8)

        # Error grid (inverted)
        error_vals = 1.0 - lum_vals
        error_grid = make_grid(error_vals, RIGHT * 3.5 + DOWN * 0.3)
        error_label = scaled_text("Error (inverted)", 20, color=SUBTLE).next_to(error_grid, DOWN, buff=0.2)
        self.play(FadeIn(error_grid), FadeIn(error_label), run_time=0.8)

        # Explanation
        note = scaled_text("High error = dark in original = needs thread coverage", 22, color=ACCENT2)
        clamp_width(note)
        note.to_edge(DOWN, buff=0.5)
        self.play(FadeIn(note, shift=UP * 0.2), run_time=0.6)
        self.wait(1.8)
        self.play(FadeOut(Group(*self.mobjects)), run_time=0.8)


# ═══════════════════════════════════════════════════════════════
#  SCENE 5 — Step 3: Place pins
# ═══════════════════════════════════════════════════════════════
class Scene05_Pins(Scene):
    def construct(self):
        header = section_title("Step 3: Place Pins Around a Circle")
        header.to_edge(UP, buff=0.5)
        self.play(Write(header), run_time=0.8)

        center = DOWN * 0.3
        radius = 2.5
        circle = Circle(radius=radius, color=SUBTLE, stroke_width=1.5).move_to(center)
        self.play(Create(circle), run_time=0.6)

        # Formula
        formula = VGroup(
            MathTex(r"\theta_i = \frac{2\pi \cdot i}{N}", font_size=28, color=ACCENT3),
            MathTex(r"x_i = \text{center} + r \cos\theta_i", font_size=24, color=WHITE),
            MathTex(r"y_i = \text{center} + r \sin\theta_i", font_size=24, color=WHITE),
        ).arrange(DOWN, buff=0.15).to_edge(RIGHT, buff=0.8).shift(DOWN * 0.3)

        total_pins = 48  # show fewer for clarity
        pins = VGroup()
        labels = VGroup()
        for i in range(total_pins):
            p = pin_on_circle(center, radius, i, total_pins)
            dot = Dot(p, radius=0.06, color=PIN_COLOR)
            pins.add(dot)
            # Label a few pins
            if i % 12 == 0:
                lbl = scaled_text(str(i), 14, color=ACCENT).move_to(
                    pin_on_circle(center, radius + 0.35, i, total_pins)
                )
                labels.add(lbl)

        # Animate pins appearing one by one (fast)
        self.play(
            LaggedStart(*[FadeIn(p, scale=2) for p in pins], lag_ratio=0.03),
            run_time=2.0
        )
        self.play(FadeIn(labels), run_time=0.4)
        self.play(Write(formula), run_time=1.0)

        # Label
        n_label = scaled_text(f"N = {total_pins} pins equally spaced", 22, color=ACCENT2)
        n_label.to_edge(DOWN, buff=0.5)
        self.play(FadeIn(n_label), run_time=0.5)
        self.wait(1.5)
        self.play(FadeOut(Group(*self.mobjects)), run_time=0.8)


# ═══════════════════════════════════════════════════════════════
#  SCENE 6 — Step 4: Precalculate lines
# ═══════════════════════════════════════════════════════════════
class Scene06_Precalculate(Scene):
    def construct(self):
        header = section_title("Step 4: Precalculate Line Pixel Coordinates")
        header.to_edge(UP, buff=0.5)
        self.play(Write(header), run_time=0.8)

        center = LEFT * 2.5 + DOWN * 0.3
        radius = 2.2
        n = 24
        circle = Circle(radius=radius, color=SUBTLE, stroke_width=1.5).move_to(center)
        pins = VGroup()
        for i in range(n):
            p = pin_on_circle(center, radius, i, n)
            pins.add(Dot(p, radius=0.05, color=PIN_COLOR))

        self.play(Create(circle), FadeIn(pins), run_time=0.5)

        # Show a line between two pins with sampled pixels
        pin_a, pin_b = 2, 14
        pa = pin_on_circle(center, radius, pin_a, n)
        pb = pin_on_circle(center, radius, pin_b, n)

        line = Line(pa, pb, color=ACCENT2, stroke_width=2)
        self.play(Create(line), run_time=0.6)

        # Show pixel samples along the line
        num_samples = 20
        sample_dots = VGroup()
        for k in range(num_samples):
            t = k / (num_samples - 1)
            pos = pa + t * (pb - pa)
            d = Dot(pos, radius=0.04, color=ACCENT3)
            sample_dots.add(d)

        self.play(LaggedStart(*[FadeIn(d, scale=2) for d in sample_dots], lag_ratio=0.05), run_time=1.0)

        # Labels (render larger, scale down to preserve kerning)
        lbl_a = Text(f"Pin {pin_a}", font_size=32, color=ACCENT).scale(0.5).next_to(Dot(pa), LEFT, buff=0.15)
        lbl_b = Text(f"Pin {pin_b}", font_size=32, color=ACCENT).scale(0.5).next_to(Dot(pb), RIGHT, buff=0.15)
        self.play(FadeIn(lbl_a), FadeIn(lbl_b), run_time=0.3)

        # Right side: explanation with indentation
        exp_lines = [
            (0, "For every valid pin pair:", 44, WHITE),
            (1, "1. Compute distance d", 38, ACCENT2),
            (1, "2. Sample d pixels along line", 38, ACCENT2),
            (1, "3. Store (x,y) coords in cache", 38, ACCENT2),
            (0, "Skip pairs closer than MIN_DIST", 38, ACCENT),
            (1, "(short arcs don't help)", 34, SUBTLE),
        ]
        INDENT_WIDTH = 0.25
        explanation = VGroup()
        for indent, txt, size, clr in exp_lines:
            t = Text(txt, font_size=size, color=clr).scale(0.5)
            explanation.add(t)
        explanation.arrange(DOWN, aligned_edge=LEFT, buff=0.22)
        for i, (indent, _, _, _) in enumerate(exp_lines):
            explanation[i].shift(RIGHT * indent * INDENT_WIDTH)
        explanation.shift(RIGHT * 3.0 + DOWN * 0.2)

        self.play(FadeIn(explanation), run_time=0.8)

        # Show cache notation
        cache_text = MathTex(
            r"\text{cache}[i \cdot N + j] = \{(x_k, y_k)\}_{k=1}^{d}",
            font_size=26, color=ACCENT3
        ).to_edge(DOWN, buff=0.5)
        self.play(Write(cache_text), run_time=0.8)
        self.wait(1.5)
        self.play(FadeOut(Group(*self.mobjects)), run_time=0.8)


# ═══════════════════════════════════════════════════════════════
#  SCENE 7 — Step 5: The Greedy Algorithm (core)
# ═══════════════════════════════════════════════════════════════
class Scene07_GreedyAlgo(Scene):
    def construct(self):
        header = section_title("Step 5: Greedy Line Selection", "The core of the algorithm")
        header.to_edge(UP, buff=0.4)
        self.play(Write(header), run_time=0.8)

        # Pseudocode with indentation (indent level, text, color)
        code_lines = [
            (0, "current_pin = 0", WHITE),
            (0, "recent_pins = []", SUBTLE),
            (0, "for iteration in range(MAX_LINES):", WHITE),
            (1, "best_pin, best_score = None, 0", WHITE),
            (1, "for candidate in all_pins:", ACCENT2),
            (2, "if too_close(candidate): continue", SUBTLE),
            (2, "if candidate in recent: continue", SUBTLE),
            (2, "pixels = cache[current, candidate]", ACCENT2),
            (2, "score = mean(error[p] for p in pixels)", ACCENT3),  # line 8
            (2, "if score > best_score:", ACCENT2),
            (3, "best_score = score", ACCENT2),
            (3, "best_pin = candidate", ACCENT2),
            (1, "# Commit the best line", ACCENT),
            (1, "for px in cache[current, best_pin]:", ACCENT),
            (2, "error[px] -= LINE_WEIGHT", ACCENT),  # line 14
            (1, "current_pin = best_pin", WHITE),
        ]

        INDENT_WIDTH = 0.3  # spacing per indent level
        code_vg = VGroup()
        for indent, txt, clr in code_lines:
            t = Text(txt, font_size=32, color=clr, font="Menlo").scale(0.5)
            t.shift(RIGHT * indent * INDENT_WIDTH)
            code_vg.add(t)
        code_vg.arrange(DOWN, aligned_edge=LEFT, buff=0.14)
        # Re-apply indentation after arrange (which resets positions)
        for i, (indent, _, _) in enumerate(code_lines):
            code_vg[i].shift(RIGHT * indent * INDENT_WIDTH)
        code_vg.move_to(ORIGIN + DOWN * 0.3).to_edge(LEFT, buff=0.5)

        # Highlight boxes for key parts
        self.play(FadeIn(code_vg), run_time=1.0)
        self.wait(0.5)

        # Highlight scoring line (index 8)
        highlight = SurroundingRectangle(code_vg[8], color=ACCENT3, buff=0.05, stroke_width=2)
        note1 = Text("← Best = most darkness", font_size=32, color=ACCENT3).scale(0.5)
        note1.move_to(RIGHT * 4.5 + code_vg[8].get_center()[1] * UP)
        self.play(Create(highlight), FadeIn(note1), run_time=0.6)
        self.wait(1.0)

        # Highlight subtraction (index 14)
        highlight2 = SurroundingRectangle(code_vg[14], color=ACCENT, buff=0.05, stroke_width=2)
        note2 = Text("← Prevent redundant coverage", font_size=32, color=ACCENT).scale(0.5)
        note2.move_to(RIGHT * 4.5 + code_vg[14].get_center()[1] * UP)
        self.play(ReplacementTransform(highlight, highlight2), ReplacementTransform(note1, note2), run_time=0.6)
        self.wait(1.8)
        self.play(FadeOut(Group(*self.mobjects)), run_time=0.8)


# ═══════════════════════════════════════════════════════════════
#  SCENE 8 — Animated demo of greedy selection
# ═══════════════════════════════════════════════════════════════
class Scene08_GreedyDemo(Scene):
    def construct(self):
        header = section_title("Greedy Selection in Action")
        header.to_edge(UP, buff=0.5)
        self.play(Write(header), run_time=0.6)

        center = LEFT * 1.5 + DOWN * 0.3
        radius = 2.5
        n = 36
        min_dist = 5

        circle = Circle(radius=radius, color=SUBTLE, stroke_width=1).move_to(center)
        pins = VGroup()
        for i in range(n):
            p = pin_on_circle(center, radius, i, n)
            pins.add(Dot(p, radius=0.05, color=PIN_COLOR))

        self.play(Create(circle), FadeIn(pins), run_time=0.5)

        # Create a "fake" error image (bright center = dark original = high error)
        # We'll use this to guide our greedy choices visually
        # For demo, we pick a reasonable sequence
        np.random.seed(42)

        # Pre-compute a plausible sequence
        error_map = np.random.rand(n) * 0.5 + 0.5  # per-pin "error" for demo
        sequence = [0]
        current = 0
        recent = []

        for _ in range(25):
            best = -1
            best_score = -1
            for off in range(min_dist, n - min_dist):
                cand = (current + off) % n
                if cand in recent:
                    continue
                score = error_map[cand] + 0.3 * abs(cand - n // 2) / n
                if score > best_score:
                    best_score = score
                    best = cand
            if best == -1:
                break
            sequence.append(best)
            error_map[best] *= 0.6
            recent.append(best)
            if len(recent) > 8:
                recent.pop(0)
            current = best

        # Info panel
        info_panel = VGroup(
            scaled_text("Current pin: 0", 20),
            scaled_text("Lines drawn: 0", 20),
            scaled_text("Best candidate: —", 20, color=ACCENT2),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.2).to_edge(RIGHT, buff=0.8).shift(DOWN * 0.3)
        self.play(FadeIn(info_panel), run_time=0.3)

        # Animate the first several line selections
        drawn_lines = VGroup()
        current_dot = Dot(pin_on_circle(center, radius, 0, n), radius=0.1, color=ACCENT3)
        self.play(FadeIn(current_dot), run_time=0.3)

        for step in range(min(20, len(sequence) - 1)):
            from_pin = sequence[step]
            to_pin = sequence[step + 1]
            pa = pin_on_circle(center, radius, from_pin, n)
            pb = pin_on_circle(center, radius, to_pin, n)

            # Show scanning candidates briefly for first few steps
            if step < 3:
                scan_lines = VGroup()
                for off in range(min_dist, min(n - min_dist, min_dist + 8)):
                    cand = (from_pin + off) % n
                    pc = pin_on_circle(center, radius, cand, n)
                    sl = Line(pa, pc, stroke_width=1, stroke_opacity=0.3, color=ACCENT2)
                    scan_lines.add(sl)
                self.play(FadeIn(scan_lines), run_time=0.3)
                self.play(FadeOut(scan_lines), run_time=0.2)

            # Draw the chosen line
            new_line = Line(pa, pb, stroke_width=1.0, stroke_opacity=0.5, color=WHITE)
            drawn_lines.add(new_line)

            # Update info
            new_info = VGroup(
                scaled_text(f"Current pin: {to_pin}", 20),
                scaled_text(f"Lines drawn: {step + 1}", 20),
                scaled_text(f"Best candidate: pin {to_pin}", 20, color=ACCENT2),
            ).arrange(DOWN, aligned_edge=LEFT, buff=0.2).to_edge(RIGHT, buff=0.8).shift(DOWN * 0.3)

            self.play(
                Create(new_line),
                current_dot.animate.move_to(pb),
                FadeOut(info_panel),
                FadeIn(new_info),
                run_time=0.25 if step > 2 else 0.5,
            )
            info_panel = new_info

        self.wait(1.0)
        self.play(FadeOut(Group(*self.mobjects)), run_time=0.8)


# ═══════════════════════════════════════════════════════════════
#  SCENE 9 — Error subtraction visualized
# ═══════════════════════════════════════════════════════════════
class Scene09_ErrorSubtraction(Scene):
    def construct(self):
        header = section_title("Why Subtract Error?")
        header.to_edge(UP, buff=0.5)
        self.play(Write(header), run_time=0.8)

        # Create two grids side by side
        n = 8
        gs = 2.8
        cs = gs / n

        # Initial error values (bright = high error)
        np.random.seed(55)
        error_vals = np.clip(np.random.rand(n, n) * 0.6 + 0.4, 0, 1)
        # Make a diagonal stripe of high error
        for i in range(n):
            j = i
            if 0 <= j < n:
                error_vals[i][j] = 0.95

        def make_grid(vals, pos):
            grp = VGroup()
            for i in range(n):
                for j in range(n):
                    c = interpolate_color(ManimColor("#0f0f2a"), ManimColor("#ff6b35"), vals[i][j])
                    sq = Square(side_length=cs, fill_opacity=1, fill_color=c, stroke_width=0.3, stroke_color=SUBTLE)
                    sq.move_to(pos + RIGHT * (j - n / 2 + 0.5) * cs + DOWN * (i - n / 2 + 0.5) * cs)
                    grp.add(sq)
            return grp

        pos_before = LEFT * 3.5 + DOWN * 0.5
        pos_after = RIGHT * 3.5 + DOWN * 0.5

        grid_before = make_grid(error_vals, pos_before)
        lbl_before = scaled_text("Error before line", 18, color=SUBTLE).next_to(grid_before, DOWN, buff=0.2)
        self.play(FadeIn(grid_before), FadeIn(lbl_before), run_time=0.6)

        # Show the line path on the grid
        line_path = Line(
            pos_before + LEFT * gs / 2 + UP * gs / 2,
            pos_before + RIGHT * gs / 2 + DOWN * gs / 2,
            color=ACCENT3, stroke_width=3
        )
        lbl_line = scaled_text("Thread path", 16, color=ACCENT3).next_to(line_path, UP, buff=0.15)
        self.play(Create(line_path), FadeIn(lbl_line), run_time=0.6)

        # Arrow
        arrow = Arrow(LEFT * 1.2 + DOWN * 0.5, RIGHT * 1.2 + DOWN * 0.5, color=ACCENT)
        sub_text = MathTex(r"\text{error}[px] \mathrel{-}= w", font_size=28, color=ACCENT3).next_to(arrow, UP, buff=0.15)
        self.play(GrowArrow(arrow), Write(sub_text), run_time=0.6)

        # After: reduced error along diagonal
        error_after = error_vals.copy()
        for i in range(n):
            j = i
            if 0 <= j < n:
                error_after[i][j] = max(0, error_after[i][j] - 0.5)

        grid_after = make_grid(error_after, pos_after)
        lbl_after = scaled_text("Error after line", 18, color=SUBTLE).next_to(grid_after, DOWN, buff=0.2)
        self.play(FadeIn(grid_after), FadeIn(lbl_after), run_time=0.6)

        note = scaled_text(
            "Darkness along thread is \"used up\" — future lines target remaining areas",
            20, color=ACCENT2
        )
        clamp_width(note)
        note.to_edge(DOWN, buff=0.4)
        self.play(FadeIn(note, shift=UP * 0.2), run_time=0.6)
        self.wait(1.8)
        self.play(FadeOut(Group(*self.mobjects)), run_time=0.8)


# ═══════════════════════════════════════════════════════════════
#  SCENE 10 — Step 6: Render the output
# ═══════════════════════════════════════════════════════════════
class Scene10_Output(Scene):
    def construct(self):
        header = section_title("Step 6: Render the Output")
        header.to_edge(UP, buff=0.5)
        self.play(Write(header), run_time=0.8)

        # Build up a string art progressively
        center = ORIGIN + DOWN * 0.3
        radius = 2.8
        n = 60

        circle = Circle(radius=radius, color=SUBTLE, stroke_width=1).move_to(center)
        pins = VGroup()
        for i in range(n):
            p = pin_on_circle(center, radius, i, n)
            pins.add(Dot(p, radius=0.03, color=PIN_COLOR))

        self.play(Create(circle), FadeIn(pins), run_time=0.5)

        # Generate a reasonably dense sequence
        np.random.seed(123)
        sequence = [0]
        cur = 0
        recent = []
        min_d = 8
        for _ in range(300):
            best = -1
            best_s = -1
            for off in range(min_d, n - min_d):
                cand = (cur + off) % n
                if cand in recent:
                    continue
                s = np.random.rand() + 0.1 * abs(cand - 30) / 30
                if s > best_s:
                    best_s = s
                    best = cand
            if best == -1:
                break
            sequence.append(best)
            recent.append(best)
            if len(recent) > 15:
                recent.pop(0)
            cur = best

        # Draw in batches with increasing speed
        all_lines = VGroup()
        for k in range(len(sequence) - 1):
            pa = pin_on_circle(center, radius, sequence[k], n)
            pb = pin_on_circle(center, radius, sequence[k + 1], n)
            line = Line(pa, pb, stroke_width=0.5, stroke_opacity=0.15, color=WHITE)
            all_lines.add(line)

        # Phase 1: slow (first 20)
        counter = scaled_text("Lines: 0", 22, color=ACCENT2).to_edge(DOWN, buff=0.5)
        self.play(FadeIn(counter), run_time=0.2)

        batch1 = all_lines[:20]
        self.play(
            LaggedStart(*[Create(l) for l in batch1], lag_ratio=0.06),
            run_time=2.0
        )
        c1 = scaled_text("Lines: 20", 22, color=ACCENT2).to_edge(DOWN, buff=0.5)
        self.play(ReplacementTransform(counter, c1), run_time=0.2)

        # Phase 2: medium (next 80)
        batch2 = all_lines[20:100]
        self.play(
            LaggedStart(*[Create(l) for l in batch2], lag_ratio=0.01),
            run_time=2.0
        )
        c2 = scaled_text("Lines: 100", 22, color=ACCENT2).to_edge(DOWN, buff=0.5)
        self.play(ReplacementTransform(c1, c2), run_time=0.2)

        # Phase 3: fast (rest)
        batch3 = all_lines[100:]
        self.play(
            LaggedStart(*[Create(l) for l in batch3], lag_ratio=0.002),
            run_time=2.0
        )
        c3 = scaled_text(f"Lines: {len(sequence) - 1}", 22, color=ACCENT2).to_edge(DOWN, buff=0.5)
        self.play(ReplacementTransform(c2, c3), run_time=0.2)

        note = scaled_text("Overlapping semi-transparent lines → darker regions", 20, color=ACCENT)
        clamp_width(note)
        note.next_to(c3, UP, buff=0.3)
        self.play(FadeIn(note), run_time=0.5)
        self.wait(1.5)
        self.play(FadeOut(Group(*self.mobjects)), run_time=0.8)


# ═══════════════════════════════════════════════════════════════
#  SCENE 11 — Recap / summary
# ═══════════════════════════════════════════════════════════════
class Scene11_Summary(Scene):
    def construct(self):
        header = section_title("Algorithm Summary")
        header.to_edge(UP, buff=0.5)
        self.play(Write(header), run_time=0.8)

        steps = VGroup(
            VGroup(
                scaled_text("①", 28, color=ACCENT),
                scaled_text("Load image → grayscale → error map (invert)", 22)
            ).arrange(RIGHT, buff=0.3),
            VGroup(
                scaled_text("②", 28, color=ACCENT),
                scaled_text("Place N pins equally around a circle", 22)
            ).arrange(RIGHT, buff=0.3),
            VGroup(
                scaled_text("③", 28, color=ACCENT),
                scaled_text("Cache pixel coords for every valid pin pair", 22)
            ).arrange(RIGHT, buff=0.3),
            VGroup(
                scaled_text("④", 28, color=ACCENT),
                scaled_text("Greedily select line with highest average error", 22)
            ).arrange(RIGHT, buff=0.3),
            VGroup(
                scaled_text("⑤", 28, color=ACCENT),
                scaled_text("Subtract weight along that line's pixels", 22)
            ).arrange(RIGHT, buff=0.3),
            VGroup(
                scaled_text("⑥", 28, color=ACCENT),
                scaled_text("Repeat ④–⑤ for thousands of lines", 22)
            ).arrange(RIGHT, buff=0.3),
            VGroup(
                scaled_text("⑦", 28, color=ACCENT),
                scaled_text("Render: draw all lines with alpha blending", 22)
            ).arrange(RIGHT, buff=0.3),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.3).move_to(ORIGIN + DOWN * 0.2)
        clamp_width(steps, max_width=12.5)

        for step in steps:
            self.play(FadeIn(step, shift=RIGHT * 0.3), run_time=0.4)
            self.wait(0.3)

        # Key insight box
        box_text = Text(
            "Key insight: each line covers the most remaining darkness,\n"
            "subtracting error prevents redundant coverage.",
            font_size=40, color=ACCENT3, line_spacing=1.3
        ).scale(0.5)
        clamp_width(box_text, max_width=12.0)
        box = SurroundingRectangle(box_text, color=ACCENT3, buff=0.2, stroke_width=1.5, corner_radius=0.1)
        insight = VGroup(box, box_text).to_edge(DOWN, buff=0.4)
        self.play(FadeIn(insight), run_time=0.8)
        self.wait(2.0)
        self.play(FadeOut(Group(*self.mobjects)), run_time=0.8)

        # Final title
        final = scaled_text("String Art Algorithm", 48, color=WHITE, weight=BOLD)
        final2 = scaled_text("A single thread, thousands of lines, one image.", 24, color=ACCENT2)
        VGroup(final, final2).arrange(DOWN, buff=0.3)
        self.play(Write(final), run_time=0.8)
        self.play(FadeIn(final2, shift=UP * 0.2), run_time=0.6)
        self.wait(1.5)
        self.play(FadeOut(Group(*self.mobjects)), run_time=1.0)
