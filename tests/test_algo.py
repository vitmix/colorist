"""Check compact histograms against an independent four-connected flood-fill oracle."""

from collections import Counter, deque
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest

try:
    from PIL import Image
except ImportError:
    Image = None

ROOT = Path(__file__).resolve().parents[1]


def expected_pixels(mask, colors, width, height, foreground_threshold=127):
    result = [(0, 0, 0, 255)] * (width * height)
    seen = set()
    for seed in range(width * height):
        if mask[seed] <= foreground_threshold or seed in seen:
            continue
        pending = deque([seed])
        seen.add(seed)
        component = []
        while pending:
            index = pending.popleft()
            component.append(index)
            x, y = index % width, index // width
            for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
                if 0 <= nx < width and 0 <= ny < height:
                    neighbor = ny * width + nx
                    if neighbor not in seen and mask[neighbor] > foreground_threshold:
                        seen.add(neighbor)
                        pending.append(neighbor)
        histogram = Counter(colors[index] for index in component)

        def winner_key(color):
            r, g, b, a = color
            packed = (a << 24) | (r << 16) | (g << 8) | b
            return -histogram[color], packed

        winner = min(histogram, key=winner_key)
        for index in component:
            result[index] = winner
    return result


@unittest.skipIf(Image is None, "Pillow is needed for image fixtures")
class CompactHistogramTests(unittest.TestCase):
    def setUp(self):
        self.exe = ROOT / "build" / ("colorist.exe" if sys.platform == "win32" else "colorist")
        if not self.exe.is_file():
            self.skipTest("Build colorist first")
        build = (ROOT / "build").resolve()
        self.assertEqual(build.parent, ROOT.resolve())
        self.temporary = tempfile.TemporaryDirectory(prefix="palette test ", dir=build)
        self.directory = Path(self.temporary.name)
        self.assertEqual(self.directory.resolve().parent, build)
        self.addCleanup(self.temporary.cleanup)

    def check_case(self, width, height, mask, colors, algorithm="base_spaghetti", parallel=False, threads=None):
        mask_image = Image.new("L", (width, height))
        mask_image.putdata(mask)
        color_image = Image.new("RGBA", (width, height))
        color_image.putdata(colors)
        mask_image.save(self.directory / "mask.png")
        color_image.save(self.directory / "colors.png")
        command = [str(self.exe), "--mask", str(self.directory / "mask.png"),
                   "--color_mask", str(self.directory / "colors.png"),
                   "--output", str(self.directory / "result.png"), "--algo-type", algorithm]
        if parallel or threads is not None:
            command.append("--parallel")
        if threads is not None:
            command.extend(["--threads", str(threads)])
        completed = subprocess.run(command, capture_output=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        with Image.open(self.directory / "result.png") as image:
            if algorithm == "rle":
                # RLE reads the color mask as BGR: alpha is not part of a color. Output is BGR.
                self.assertEqual(image.mode, "RGB")
                opaque = [color[:3] + (255,) for color in colors]
                expected = expected_pixels(mask, opaque, width, height)
            else:
                expected = expected_pixels(mask, colors, width, height)
            self.assertEqual(list(image.convert("RGBA").getdata()),
                             expected)

    def test_huge_packed_black_value(self):
        self.check_case(1, 1, [255], [(0, 0, 0, 255)])

    def test_midrow_completion_waits_for_all_branches_of_a_component(self):
        width, height = 65, 9
        # A full first row connects two arms in the next row. The left arm ends
        # before the midpoint, but the right arm continues and changes the winner.
        mask = [255 if y == 0 or (y == 1 and x < 5) or x >= 40 else 0
                for y in range(height) for x in range(width)]
        colors = [(255, 0, 0, 255) if y < 2 else (0, 255, 0, 255)
                  for y in range(height) for x in range(width)]
        for threads in (1, 2, 4):
            with self.subTest(threads=threads):
                self.check_case(width, height, mask, colors, algorithm="rle", threads=threads)

    def test_rle_odd_widths_long_runs_and_empty_rows(self):
        for width in (1, 3, 17, 33, 65):
            mask = [255 if y in (0, 2, 3) and (y == 3 or x % 3 != 1) else 0
                    for y in range(5) for x in range(width)]
            colors = [(7, 8, 9, 0) if y < 3 else (17, 23, 41, 255)
                      for y in range(5) for x in range(width)]
            for threads in (1, 3):
                with self.subTest(width=width, threads=threads):
                    self.check_case(width, 5, mask, colors, algorithm="rle", threads=threads)

    def test_rle_row_color_changes_and_mixed_runs(self):
        red, green = (255, 0, 0, 255), (0, 255, 0, 255)
        for width in (1, 2, 15, 16, 17, 31, 32, 33, 97):
            colors = [red] * width + [red if x % 2 == 0 else green for x in range(width)] + [green] * (2 * width)
            for threads in (1, 2, 4):
                with self.subTest(width=width, threads=threads):
                    self.check_case(width, 4, [255] * (4 * width), colors,
                                    algorithm="rle", threads=threads)

    def test_rle_ignores_alpha_and_counts_nonadjacent_colors(self):
        for colors in ([(7, 8, 9, 0), (7, 8, 9, 255)] * 32,
                       [(0, 0, 0, 0)] * 31 + [(1, 2, 3, 255)] + [(0, 0, 0, 0)] * 32):
            self.check_case(len(colors), 1, [255] * len(colors), colors, algorithm="rle")

    def test_completion_recycles_histograms_and_defers_stripe_boundaries(self):
        # Each short polygon ends before the next one appears. Colors alternate,
        # exposing stale histogram counts or run links when slots are recycled.
        width, height = 65, 12
        mask = [255 if y % 3 != 2 and x % 5 != 4 else 0
                for y in range(height) for x in range(width)]
        colors = [(255, 0, 0, 0) if (y // 3) % 2 == 0 else (0, 255, 0, 255)
                  for y in range(height) for x in range(width)]
        # A component spans multiple stripe boundaries and changes winner below.
        for y in range(height):
            mask[y * width] = 255
            colors[y * width] = (255, 0, 0, 255) if y < 3 else (0, 0, 255, 255)
        for threads in (1, 2, 3, 4, 6, 12):
            with self.subTest(threads=threads):
                self.check_case(width, height, mask, colors, algorithm="rle", threads=threads)

    def test_rle_keeps_colors_when_histograms_are_reused(self):
        width, height = 65, 13
        # A persistent column keeps rows from being painted, while other components
        # finish and release histogram slots long before their rows are painted.
        mask = [255 if x == 0 or (y % 3 == 0 and x >= 2 and x % 5 != 1) else 0
                for y in range(height) for x in range(width)]
        palette = [(0, 0, 0, 0), (255, 0, 0, 255), (0, 255, 0, 255), (0, 0, 255, 0)]
        colors = [palette[(y // 3 + x // 5) % len(palette)]
                  for y in range(height) for x in range(width)]
        for threads in (1, 2, 3, 4, 13):
            with self.subTest(threads=threads):
                self.check_case(width, height, mask, colors, algorithm="rle", threads=threads)

    def test_completion_merged_roots_stale_row_ids_and_zero_winner(self):
        masks = ([255, 0, 255, 0, 255, 0, 255,
                  255, 255, 255, 255, 255, 255, 255,
                  0, 0, 0, 0, 0, 0, 0,
                  255, 255, 255, 255, 255, 255, 255],
                 [255] * 7 + [0] * 21)
        for mask in masks:
            for threads in (1, 2, 4):
                with self.subTest(mask=mask, threads=threads):
                    self.check_case(7, 4, mask, [(0, 0, 0, 0)] * 28,
                                    algorithm="rle", threads=threads)

    def test_rle_short_spans_background_and_alpha(self):
        for width in (1, 2, 15, 16, 17, 31, 32, 33, 65, 129):
            height = 5
            mask = [255 if y != 2 and x != width - 1 else 0
                    for y in range(height) for x in range(width)]
            colors = [(17, 23, 41, 0)] * (width * height)
            colors[width + width // 2] = (7, 8, 9, 255)
            for threads in (1, 2, 4):
                with self.subTest(width=width, threads=threads):
                    self.check_case(width, height, mask, colors, algorithm="rle", threads=threads)

    def test_rle_merged_histograms_change_winner(self):
        red, green, blue = (255, 0, 0, 255), (0, 255, 0, 255), (0, 0, 255, 255)
        # Two separate roots initially choose red and blue. Connecting them makes
        # green win: four greens plus three connecting pixels beat four reds/blues.
        mask = [255, 255, 0, 255, 255,
                255, 255, 0, 255, 255,
                255, 255, 0, 255, 255,
                0, 255, 255, 255, 0]
        colors = [red, red, blue, blue, blue,
                  red, green, blue, blue, green,
                  green, red, blue, green, blue,
                  blue, green, green, green, blue]
        for threads in (1, 2, 4):
            with self.subTest(threads=threads):
                self.check_case(5, 4, mask, colors, algorithm="rle", threads=threads)
                self.check_case(33, 4, [255] * 132, [red] * 33 + [green] * 99,
                                algorithm="rle", threads=threads)

    def test_first_color_is_packed_zero(self):
        self.check_case(2, 1, [255, 255], [(0, 0, 0, 0)] * 2)

    def test_cached_color_across_polygons_and_rows(self):
        self.check_case(3, 3, [255, 0, 255] * 3, [(17, 23, 41, 255)] * 9)

    def test_tie_uses_color_value_not_discovery_order(self):
        self.check_case(4, 1, [255] * 4, [(255, 0, 0, 255)] * 2 + [(0, 0, 255, 255)] * 2)

    def test_alpha_is_part_of_color(self):
        self.check_case(4, 1, [255] * 4, [(7, 8, 9, 255), (7, 8, 9, 0),
                                             (7, 8, 9, 255), (7, 8, 9, 0)])

    def test_empty_foreground(self):
        self.check_case(3, 1, [0, 1, 127], [(255, 255, 255, 255)] * 3)

    def test_large_palette_and_holes_in_polygon_histograms(self):
        width, height = 31, 31
        mask = [255 if (x + y) % 2 == 0 else 0 for y in range(height) for x in range(width)]
        colors = [(i % 256, i // 256, 123, 255) for i in range(width * height)]
        self.check_case(width, height, mask, colors)

    def test_random_masks_and_colors(self):
        generator = random.Random(42)
        width, height = 13, 9
        for case in range(8):
            with self.subTest(case=case):
                mask = [generator.choice((0, 1, 127, 128, 255)) for _ in range(width * height)]
                palette = [(0, 0, 0, 255), (0, 0, 0, 0)] + [
                    tuple(generator.randrange(256) for _ in range(4)) for _ in range(12)]
                colors = [generator.choice(palette) for _ in mask]
                self.check_case(width, height, mask, colors)

    def test_all_labeling_algorithms_and_thread_modes(self):
        # Tall enough to exercise parallel labeling on ordinary desktop/RPi thread counts.
        width, height = 7, 256
        mask = [255 if x in (0, 1, 3, 5, 6) else 0 for y in range(height) for x in range(width)]
        colors = [((255, 0, 0, 255) if y % 2 == 0 else (0, 0, 255, 255))
                  for y in range(height) for x in range(width)]
        for algorithm in ("base_spaghetti", "base_sauf", "base_bbdt", "rle"):
            for parallel in (False, True):
                with self.subTest(algorithm=algorithm, parallel=parallel):
                    self.check_case(width, height, mask, colors, algorithm, parallel)

    def test_rle_coloring_boundaries_alpha_and_random_components(self):
        palette = [(0, 0, 0, 0), (0, 0, 0, 255), (255, 0, 0, 255), (0, 0, 255, 255)]
        fixtures = [
            (1, 1, [0], [palette[2]]),
            (1, 1, [255], [palette[0]]),
            (4, 1, [255] * 4, [palette[2]] * 2 + [palette[3]] * 2),
            (3, 5, [255, 0, 255, 0, 0, 0, 255, 255, 0, 0, 0, 0, 0, 0, 0], [palette[1]] * 15),
            (5, 1, [0, 1, 127, 128, 255], [palette[2]] * 5),
        ]
        generator = random.Random(123)
        for _ in range(20):
            width, height = generator.randrange(1, 20), generator.randrange(1, 20)
            fixtures.append((width, height,
                             [generator.choice((0, 1, 127, 128, 255)) for _ in range(width * height)],
                             [generator.choice(palette) for _ in range(width * height)]))
        for case, fixture in enumerate(fixtures):
            with self.subTest(case=case):
                self.check_case(*fixture, algorithm="rle")

    def test_rle_color_stretches_preserve_pixel_frequencies(self):
        red, blue, green = (255, 0, 0, 255), (0, 0, 255, 255), (0, 255, 0, 255)
        fixtures = [
            # One long red stretch must beat two shorter blue stretches.
            (15, 1, [255] * 15, [red] * 9 + [blue] * 2 + [green] * 2 + [blue] * 2),
            # Equal RGB with different alpha is one color for RLE (alpha is not read).
            (12, 1, [255] * 12, [(7, 8, 9, 0)] * 5 + [(7, 8, 9, 255)] * 7),
            # A stretch must stop at the mask run's end, even across equal colors.
            (13, 1, [255] * 4 + [0] + [255] * 8, [red] * 5 + [blue] * 8),
            (513, 1, [255] * 513, [(0, 0, 0, 0)] * 513),
        ]
        for case, fixture in enumerate(fixtures):
            with self.subTest(case=case):
                self.check_case(*fixture, algorithm="rle")

    def test_rle_background_skip_lanes_and_row_tails(self):
        palette = [(0, 0, 0, 0), (0, 0, 0, 255), (17, 23, 41, 89)]
        for width in (7, 8, 9, 31, 32, 33, 137):
            with self.subTest(width=width):
                mask, colors = [], []
                for gap in range(width):
                    length = min(width - gap, 1 + gap % 33)
                    mask.extend([0] * gap + [255] * length + [0] * (width - gap - length))
                    colors.extend([palette[gap % len(palette)]] * width)
                    # Separate components and exercise wholly black rows.
                    mask.extend([0] * width)
                    colors.extend([palette[0]] * width)
                self.check_case(width, 2 * width, mask, colors, algorithm="rle")

    def test_rle_long_run_lengths(self):
        color = (17, 23, 41, 255)
        # Large packed lengths can have alpha 255 without being background.
        for width in (4079, 4080, 4081, 4095, 4096, 8192):
            with self.subTest(width=width):
                self.check_case(width, 1, [255] * width, [color] * width, algorithm="rle")

        # A 4080-pixel run of component 0 equals the reserved black marker.
        # It must switch representations after already packed rows and runs.
        width = 4095
        mask = [255 if x == 10 else 0 for x in range(width)]
        mask += [255 if x == 0 or 10 <= x < 4090 else 0 for x in range(width)]
        self.check_case(width, 2, mask, [color] * (2 * width), algorithm="rle")

    def test_rle_recurring_colors_in_large_palette(self):
        # Recurring keys, alpha and packed zero must resolve consistently.
        palette = [(0, 0, 0, 0), (255, 255, 255, 255)] + [
            (i, (i * 73) % 256, (i * 41) % 256, (0, 127, 255)[i % 3])
            for i in range(160)]
        for reverse in (False, True):
            with self.subTest(reverse=reverse):
                order = list(reversed(palette)) if reverse else palette
                colors = order * 3 + [palette[-1]] * 200
                self.check_case(len(colors), 1, [255] * len(colors), colors, algorithm="rle")

    def test_rle_new_color_insertions_keep_existing_histogram_ids(self):
        palette = [(i % 256, i // 256, 43, 255) for i in range(512)]
        palette.sort(key=lambda color: (color[0] << 16) | (color[1] << 8) | color[2], reverse=True)
        # Discover lower-valued colors after accumulating the first color's counts.
        colors = [palette[0]] * 17 + palette[1:] + palette
        self.check_case(len(colors), 1, [255] * len(colors), colors, algorithm="rle")

    def test_rle_large_component_color_counts(self):
        # Frequencies beyond 16 bits must keep their ordering and full counts.
        width = 131071
        colors = [(0, 0, 255, 255)] * 65536 + [(255, 0, 0, 255)] * 65535
        self.check_case(width, 1, [255] * width, colors, algorithm="rle")

    def test_rle_component_ids_above_one_byte(self):
        # 613 isolated components exercise marker bytes beyond the low byte,
        # including IDs whose low byte is 255. Final alpha can be 0 or 255.
        width, height = 35, 35
        mask = [255 if (x + y) % 2 == 0 else 0
                for y in range(height) for x in range(width)]
        colors = [(17, 23, 41, 255 if i % 3 else 0) for i in range(width * height)]
        self.check_case(width, height, mask, colors, algorithm="rle")

    def test_rle_recycled_histogram_has_no_counts_from_losing_component(self):
        # First run of row 1 joins two components; the next run reuses the losing slot.
        mask = [255, 0, 255, 0, 0, 0, 255,
                255, 255, 255, 0, 255, 0, 0]
        colors = [(255, 0, 0, 255)] * 14
        colors[2] = (0, 0, 255, 255)
        colors[6] = (255, 255, 0, 255)
        colors[11] = (0, 255, 0, 255)
        self.check_case(7, 2, mask, colors, algorithm="rle")

    def test_rle_boundary_palettes_have_different_color_ids(self):
        red, blue, green = (255, 0, 0, 255), (0, 0, 255, 255), (0, 255, 0, 255)
        # Three stripes discover R/B, G/B and R/G/B respectively; blue wins globally.
        colors = [red] * 7 + [blue] * 5 + [green] * 7 + [blue] * 5
        colors += [red] * 3 + [green] * 3 + [blue] * 6
        for threads in (2, 3, 6, 12):
            with self.subTest(threads=threads):
                self.check_case(6, 6, [255] * 36, colors, algorithm="rle", threads=threads)

    def test_rle_repeated_boundary_matches_and_same_stripe_roots(self):
        red, blue, green = (255, 0, 0, 255), (0, 0, 255, 255), (0, 255, 0, 255)
        patterns = (
            ["11111", "10101", "10101", "11111", "10101", "11111"],
            # Lower bridge joins two roots in the upper stripe; the upper root wins.
            ["10001", "10001", "10001", "11111", "00000", "00000"],
        )
        for case, rows in enumerate(patterns):
            mask = [255 if value == "1" else 0 for row in rows for value in row]
            colors = [red if x == 0 else blue if x == 4 else green
                      for y in range(6) for x in range(5)]
            for threads in (2, 3, 6):
                with self.subTest(case=case, threads=threads):
                    self.check_case(5, 6, mask, colors, algorithm="rle", threads=threads)

    def test_rle_cross_stripe_ties_and_alpha(self):
        palette = [(0, 0, 0, 0), (0, 0, 0, 255), (255, 255, 255, 255)]
        for order in (palette, list(reversed(palette))):
            for threads in (2, 3, 6, 12):
                with self.subTest(order=order, threads=threads):
                    self.check_case(7, 6, [255] * 42,
                                    [color for color in order for _ in range(14)],
                                    algorithm="rle", threads=threads)

    def test_rle_stripes_preserve_independent_components(self):
        color_a, color_b = (17, 23, 41, 0), (41, 23, 17, 255)
        for width in (4080, 4096):
            mask = [255 if x == 10 else 0 for x in range(width)] + [255] * (3 * width)
            colors = [color_a] * (2 * width) + [color_b] * (2 * width)
            for threads in (2, 3):
                with self.subTest(width=width, threads=threads):
                    self.check_case(width, 4, mask, colors, algorithm="rle", threads=threads)


if __name__ == "__main__":
    unittest.main()
