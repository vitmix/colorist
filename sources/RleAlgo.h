#pragma once

#include "IAlgo.h"

#include <opencv2/core/simd_intrinsics.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <limits>
#include <numeric>
#include <vector>
#include <utility>

#if defined(__aarch64__) && defined(__ARM_NEON)
#include <arm_neon.h>
#endif

#ifdef COLORIST_TIMING
#include <chrono>
#endif

// Run-length connected components with per-component color histograms, in one pass.
//
// - The image is split into horizontal stripes, one per thread. Each stripe scans its rows:
//   foreground runs are found per row, joined to overlapping runs of the previous row with
//   union-find (4-connectivity), and each run's color stretches are added to the
//   component's histogram.
// - A component is finished once a row no longer continues it; its color is selected
//   then and its histogram slot reused.
// - Rows are painted as soon as every component touching them is finished: each row is
//   zeroed (black background) and its colored runs filled, sequentially, shortly behind
//   the scan. Components touching an internal stripe edge are merged across stripes
//   afterwards and painted last.
class RleAlgo : public IAlgo {
public:
    explicit RleAlgo(int num_threads = 1) { SetNumThreads(num_threads); }

    int GetOutputType() const override { return CV_8UC3; }
    // Alpha never decides a color here, so the color mask is read as BGR: 25% fewer input bytes.
    int GetColorMaskType() const override { return CV_8UC3; }
    bool SupportsParallelLabeling() const override { return true; }
    bool SupportsParallelColoring() const override { return true; }

    void SetNumThreads(int num_threads) override {
        CV_Assert(num_threads > 0);
        m_num_threads = num_threads;
    }

#ifdef COLORIST_TIMING
    const std::array<double, 5>& GetTimings() const override { return m_timings_ms; }

    const std::array<const char*, 5>& GetTimingKeys() const override {
        static constexpr std::array<const char*, 5> keys = {
            "setup_ms", "scan_ccl_histogram_ms", "root_resolution_ms",
            "color_selection_ms", "final_coloring_ms"
        };
        return keys;
    }

    const std::array<const char*, 5>& GetStageNames() const override {
        static constexpr std::array<const char*, 5> names = {
            "Setup", "Scan/CCL/paint", "Reconciliation", "Color selection", "Final coloring"
        };
        return names;
    }

    // Last Compute: each stripe's scan start and end, in ms after the scan section began.
    const std::vector<std::pair<double, double>>& GetStripeSpans() const { return m_stripe_spans_ms; }
#endif

    void Compute(const cv::Mat &mask, const cv::Mat &color_mask, cv::Mat& output) override {
        CV_Assert(mask.type() == CV_8UC1);
        CV_Assert(color_mask.type() == GetColorMaskType());
        CV_Assert(mask.size() == color_mask.size());
        CV_Assert(!mask.empty());
        CV_Assert(output.type() == GetOutputType());
        CV_Assert(output.size() == mask.size());
        if (static_cast<std::uint64_t>(mask.rows) * static_cast<std::uint64_t>(mask.cols) >
            std::numeric_limits<std::uint32_t>::max()) {
            CV_Error(cv::Error::StsOutOfRange, "Image exceeds the 32-bit histogram counter limit");
        }

#ifdef COLORIST_TIMING
        using Clock = std::chrono::steady_clock;
        auto stage_start = Clock::now();
        const auto record_time = [this, &stage_start](std::size_t stage) {
            const auto now = Clock::now();
            m_timings_ms[stage] = std::chrono::duration<double, std::milli>(now - stage_start).count();
            stage_start = now;
        };
#endif

        const int height = mask.rows;
        const int width = mask.cols;

        const int stripe_count = std::min(height, m_num_threads);
        m_stripes.resize(stripe_count);

        ResetWorkingBuffers(height);

#ifdef COLORIST_TIMING
        record_time(0);
        m_stripe_spans_ms.assign(stripe_count, {0.0, 0.0});
        const auto scan_start = Clock::now();
        const auto since_scan = [&] {
            return std::chrono::duration<double, std::milli>(Clock::now() - scan_start).count();
        };
#endif

        cv::parallel_for_(cv::Range(0, stripe_count),
            [&](const cv::Range& range) {
                const cv::v_uint8 cmp_block = cv::vx_setall_u8(127);

                for (int i = range.start; i < range.end; ++i) {
#ifdef COLORIST_TIMING
                    m_stripe_spans_ms[i].first = since_scan();
#endif
                    const int stripe_begin = static_cast<int>(std::int64_t(height) * i / stripe_count);
                    const int stripe_end = static_cast<int>(std::int64_t(height) * (i + 1) / stripe_count);
                    const int stripe_rows = stripe_end - stripe_begin;

                    auto& stripe = m_stripes[i];
                    stripe.begin = stripe_begin;
                    stripe.row_start.assign(stripe_rows + 1, 0);
                    stripe.active_by_row.assign(stripe_rows, 0);

                    auto& uni = stripe.uni;
                    auto& hists = stripe.hists;

                    std::uint32_t prev_color = 0;
                    std::size_t prev_color_id = 0;
                    bool cache_valid = false;
                    int painted = stripe_begin;

                    for (int y = stripe_begin; y < stripe_end; ++y) {
                        const auto* mask_ptr = mask.ptr<uchar>(y);
                        const auto* cmask_ptr = color_mask.ptr<uchar>(y); // BGR, 3 bytes per pixel
                        const bool edge_row = (y == stripe_begin && stripe_begin > 0) ||
                                              (y == stripe_end - 1 && stripe_end < height);
                        const int row = y - stripe_begin;
                        stripe.row_start[row] = static_cast<std::uint32_t>(stripe.runs.size());

                        // Previous row's runs; empty on the stripe's first row.
                        const auto [prev_begin, prev_end] = row > 0 ? stripe.RowRange(row - 1)
                                                                    : std::pair<int, int>{0, 0};
                        int cursor = prev_begin;

                        // Pass 1: all foreground runs of the row. Pass 2: label and count each.
                        AppendRowRuns(mask_ptr, width, cmp_block, stripe.runs);
                        const int row_end = static_cast<int>(stripe.runs.size());

                        for (int k = stripe.row_start[row]; k < row_end; ++k) {
                            const int run_start = stripe.runs[k].x0;
                            const int c = stripe.runs[k].x1 + 1;

                            int component_root_id = -1;

                            ForEachOverlap(stripe.runs[k], stripe.runs, cursor, prev_end,
                                [&](const PaintRun& prev_run) {
                                    const auto prev_root = uni.Root(prev_run.id);
                                    if (component_root_id == -1) {
                                        component_root_id = prev_root;
                                        uni.nodes[component_root_id].size++;
                                    } else {
                                        const auto [winner, loser] = uni.Unite(component_root_id, prev_root);
                                        if (loser != -1) {
                                            hists.Merge(uni.nodes[winner].hist_id, uni.nodes[loser].hist_id);
                                            uni.nodes[loser].hist_id = -1;
                                            MergeCompletion(stripe, winner, loser);
                                        }
                                        component_root_id = winner;
                                    }
                                });

                            // New component encountered, create it
                            if (component_root_id == -1) {
                                component_root_id = static_cast<int>(uni.nodes.size());
                                uni.nodes.push_back({static_cast<std::uint32_t>(component_root_id), 1,
                                                     hists.AcquireSlot(), 0, y, y, false});
                                ++stripe.active_by_row[y - stripe_begin];
                            }

                            stripe.runs[k].id = component_root_id;

                            // The slot object stays put during the run; a new color may only
                            // grow its buffer, so index through the reference, not a raw pointer.
                            auto& counts = hists.Get(uni.nodes[component_root_id].hist_id);
                            int x = run_start;
                            while (x < c) {
                                const auto color = LoadBgr(cmask_ptr + 3 * x);
                                const int stretch_start = x;
                                x = StretchEnd(cmask_ptr, x + 1, c, width, color);

                                if (!cache_valid || color != prev_color) {
                                    prev_color = color;
                                    prev_color_id = stripe.GetColorId(color);
                                    cache_valid = true;
                                }
                                counts[prev_color_id] += static_cast<std::uint32_t>(x - stretch_start);
                            }

                            auto& state = stripe.uni.nodes[component_root_id];
                            state.last_seen = y;
                            // Components touching internal stripe edges wait for reconciliation.
                            if (edge_row)
                                MarkBoundary(stripe, component_root_id);
                        }

                        stripe.row_start[row + 1] = static_cast<std::uint32_t>(row_end);
                        CloseRow(stripe, prev_begin, prev_end, y, painted, width, output);
                    }

                    // No row follows the last one: its components end here, except those
                    // touching an internal stripe edge, which wait for reconciliation.
                    const auto [last_begin, last_end] = stripe.RowRange(stripe_rows - 1);
                    CloseRow(stripe, last_begin, last_end, stripe_end, painted, width, output);
#ifdef COLORIST_TIMING
                    m_stripe_spans_ms[i].second = since_scan();
#endif
                }
            }, m_num_threads
        );

#ifdef COLORIST_TIMING
        record_time(1);
#endif

        // Stripes used local node ids. Number them consecutively (global = base + local)
        // and copy their links into one flat union-find for the cross-stripe merge.
        m_base.assign(stripe_count + 1, 0);
        for (int i = 0; i < stripe_count; ++i)
            m_base[i + 1] = m_base[i] + static_cast<int>(m_stripes[i].uni.nodes.size());
        auto& links = m_links.nodes;
        links.resize(m_base[stripe_count]);
        for (int i = 0; i < stripe_count; ++i) {
            const auto& nodes = m_stripes[i].uni.nodes;
            for (std::size_t k = 0; k < nodes.size(); ++k)
                links[m_base[i] + k] = {static_cast<std::uint32_t>(m_base[i]) + nodes[k].parent, nodes[k].size};
        }

        // Join components across each internal edge: the upper stripe's last row
        // against the lower stripe's first row.
        for (int i = 0; i + 1 < stripe_count; ++i) {
            const auto& upper = m_stripes[i];
            const auto& lower = m_stripes[i + 1];
            auto [cursor, upper_end] = upper.RowRange(static_cast<int>(upper.row_start.size()) - 2);
            const auto [lower_begin, lower_end] = lower.RowRange(0);

            for (int k = lower_begin; k < lower_end; ++k) {
                const auto& current = lower.runs[k];
                ForEachOverlap(current, upper.runs, cursor, upper_end,
                    [&](const PaintRun& previous) {
                        const auto [winner, loser] = m_links.Unite(m_base[i] + previous.id,
                                                                   m_base[i + 1] + current.id);
                        if (loser != -1)
                            MergeBoundaryHistograms(winner, loser);
                    });
            }
        }
        m_links.ResolveAll();

#ifdef COLORIST_TIMING
        record_time(2);
#endif

        // One color per global root; components finished during the scan already have one.
        for (int g = 0; g < m_base[stripe_count]; ++g) {
            if (static_cast<int>(links[g].parent) != g)
                continue;
            const int i = StripeOf(g);
            auto& node = m_stripes[i].uni.nodes[g - m_base[i]];
            if (node.hist_id >= 0)
                node.final_color = SelectColor(m_stripes[i], node.hist_id);
        }

        // Cache each root's color on every node for the deferred pass.
        for (int i = 0; i < stripe_count; ++i) {
            auto& nodes = m_stripes[i].uni.nodes;
            for (std::size_t k = 0; k < nodes.size(); ++k)
                nodes[k].final_color = GlobalNode(static_cast<int>(links[m_base[i] + k].parent)).final_color;
        }
#ifdef COLORIST_TIMING
        record_time(3);
#endif

        // Only runs of stripe-edge components remain unpainted.
        cv::parallel_for_(cv::Range(0, stripe_count),
            [&](const cv::Range& range) {
                for (int i = range.start; i < range.end; ++i) {
                    auto& stripe = m_stripes[i];
                    for (const auto& deferred : stripe.deferred) {
                        const auto& run = stripe.runs[deferred.run];
                        FillBgrRun(output.ptr<cv::Vec3b>(deferred.y) + run.x0, run.x1 - run.x0 + 1,
                                   stripe.uni.nodes[run.id].final_color);
                    }
                }
            }, m_num_threads
        );

#ifdef COLORIST_TIMING
        record_time(4);
#endif
    }

private:
    // Byte comparison result (0x00/0xFF lanes) as a scalar bit mask: one bit per byte,
    // LaneBitStride bits apart, lowest byte lowest.
#if defined(__aarch64__) && defined(__ARM_NEON) && CV_SIMD_WIDTH == 16
    static constexpr int LaneBitStride = 4;
    static std::uint64_t LaneBits(const cv::v_uint8& comparison) {
        // Narrow each byte to a nibble; cheaper than v_signmask's shift/table/add-across.
        const std::uint64_t nibbles = vget_lane_u64(vreinterpret_u64_u8(
            vshrn_n_u16(vreinterpretq_u16_u8(comparison.val), 4)), 0);
        return nibbles & 0x1111111111111111ull;
    }
#else
    static constexpr int LaneBitStride = 1;
    static std::uint64_t LaneBits(const cv::v_uint8& comparison) {
        return static_cast<std::uint32_t>(cv::v_signmask(comparison));
    }
#endif

    // First pixel in [x, end) of a BGR row whose color differs from `color`, or end.
    // Stretches are short (median ~12 pixels), so plain 64-bit compares beat SIMD blocks:
    // one 8-byte load covers two pixels (6 bytes; the 2 spare bytes are masked off), and
    // each step checks two pairs. Loads never pass the row's last byte.
    static int StretchEnd(const uchar* bgr, int x, int end, int width, std::uint32_t color) {
        constexpr std::uint64_t kTwoPixels = 0xFFFFFFFFFFFFull;
        const std::uint64_t pair = color | (std::uint64_t{color} << 24);
        const auto differs = [&](int p) { return (Load64(bgr + 3 * p) ^ pair) & kTwoPixels; };
        const auto first_of = [](int p, std::uint64_t diff) { return (diff & 0xFFFFFFu) ? p : p + 1; };

        // An 8-byte load at pixel p reads bytes 3p..3p+7: needs p + 3 <= width.
        for (; x + 4 <= end && x + 5 <= width; x += 4) {
            if (const auto diff = differs(x))
                return first_of(x, diff);
            if (const auto diff = differs(x + 2))
                return first_of(x + 2, diff);
        }
        for (; x + 2 <= end && x + 3 <= width; x += 2)
            if (const auto diff = differs(x))
                return first_of(x, diff);
        for (; x < end; ++x)
            if (LoadBgr(bgr + 3 * x) != color)
                return x;
        return end;
    }

    static void FillBgrRun(cv::Vec3b* destination, int length, std::uint32_t packed_color) {
        const int lanes = cv::VTraits<cv::v_uint8>::vlanes();
        const cv::Vec3b color = UnpackColor(packed_color);
        if (length < lanes) {
            std::fill(destination, destination + length, color);
            return;
        }

#if CV_SIMD_WIDTH == 16
        // 48 bytes of repeated B,G,R built in registers: a stack round trip (interleaving
        // store, then reloads) stalls on store-to-load forwarding. As 64-bit words the
        // pattern has three phases, repeating every 24 bytes: w0 w1 w2 w0 w1 w2.
        const std::uint64_t c = packed_color & 0xFFFFFFu;
        const std::uint64_t w0 = c | c << 24 | c << 48;              // B G R B G R B G
        const std::uint64_t w1 = c >> 16 | c << 8 | c << 32 | c << 56; // R B G R B G R B
        const std::uint64_t w2 = c >> 8 | c << 16 | c << 40;         // G R B G R B G R
        const auto first = cv::v_reinterpret_as_u8(cv::v_uint64x2(w0, w1));
        const auto second = cv::v_reinterpret_as_u8(cv::v_uint64x2(w2, w0));
        const auto third = cv::v_reinterpret_as_u8(cv::v_uint64x2(w1, w2));
#else
        std::array<uchar, 3 * CV_SIMD_WIDTH> pattern;
        cv::v_store_interleave(pattern.data(), cv::vx_setall_u8(color[0]),
                               cv::vx_setall_u8(color[1]), cv::vx_setall_u8(color[2]));
        const auto first = cv::vx_load(pattern.data());
        const auto second = cv::vx_load(pattern.data() + lanes);
        const auto third = cv::vx_load(pattern.data() + 2 * lanes);
#endif
        auto* cursor = reinterpret_cast<uchar*>(destination);
        auto* last = reinterpret_cast<uchar*>(destination + length - lanes);
        const auto store = [&](uchar* target) {
            cv::vx_store(target, first);
            cv::vx_store(target + lanes, second);
            cv::vx_store(target + 2 * lanes, third);
        };
        for (; cursor < last; cursor += 3 * lanes)
            store(cursor);
        store(last);
    }

#ifdef COLORIST_TIMING
    std::array<double, 5> m_timings_ms{};
    std::vector<std::pair<double, double>> m_stripe_spans_ms; // Per stripe: scan start, end.
#endif

    void ResetWorkingBuffers(int height) {
        const auto expected_runs = static_cast<std::size_t>(height) / m_stripes.size() * 10;

        for (auto& stripe : m_stripes) {
            stripe.runs.clear();
            stripe.deferred.clear();
            stripe.uni.nodes.clear();
            stripe.uni.nodes.reserve(expected_runs);
            stripe.hists.Reset();
        }
    }

    // Packs a BGR pixel as B | G << 8 | R << 16, the layout UnpackColor expects.
    // A 2-byte load (B, G on little-endian CPUs) plus the R byte, in registers. (A 3-byte
    // memcpy into a word made GCC assemble the value on the stack: a store-forwarding stall.)
    static inline std::uint32_t LoadBgr(const uchar* pixel) {
        std::uint16_t blue_green;
        std::memcpy(&blue_green, pixel, sizeof(blue_green));
        return blue_green | (std::uint32_t(pixel[2]) << 16);
    }

    // Unaligned 8-byte load; on little-endian CPUs (x86, AArch64) byte 0 is the lowest byte,
    // matching LoadBgr's packing for the two pixels it covers.
    static inline std::uint64_t Load64(const uchar* bytes) {
        std::uint64_t value;
        std::memcpy(&value, bytes, sizeof(value));
        return value;
    }

    static cv::Vec3b UnpackColor(std::uint32_t color) {
        return cv::Vec3b(static_cast<uchar>(color), static_cast<uchar>(color >> 8),
                         static_cast<uchar>(color >> 16));
    }

    int m_num_threads = 1;

    // Direct-mapped color -> palette id cache in front of the sorted palette; misses are
    // only slower, never wrong. 256 entries (3 KB) leave few collisions for palettes of
    // up to a few hundred colors and still fit in L1.
    static constexpr std::size_t ColorCacheSize = 256;
    static_assert(ColorCacheSize != 0 && (ColorCacheSize & (ColorCacheSize - 1)) == 0,
        "Color cache size must be a power of two");

    // Row-ordered runs; the row is implied by Stripe::row_start.
    struct PaintRun {
        int x0;
        int x1; // Inclusive.
        int id; // Node at the time of the scan; resolve with Root().
    };

    using PaintRuns = std::vector<PaintRun>;

    struct DeferredRun {
        std::uint32_t run;
        int y;
    };

    // One union-find node per started component; the fields after size are meaningful on roots.
    struct UnionNode {
        std::uint32_t parent;
        std::uint32_t size;
        int hist_id;               // Histogram slot; -1 once finished or merged away.
        std::uint32_t final_color; // Set when the component is finished.
        int first_row;             // Topmost row; keys this component in active_by_row while counted.
        int last_seen;
        bool boundary;             // Touches an internal stripe edge: not counted, painted after reconciliation.
    };

    // Cross-stripe link: global ids, no per-component state.
    struct LinkNode {
        std::uint32_t parent;
        std::uint32_t size;
    };

    // Union-find with path halving and union by size. Node needs `parent` and `size`.
    // Used per stripe during the scan (UnionNode) and once across stripes (LinkNode).
    template<typename Node>
    struct Unions {
        int Root(int p) {
            while (p != static_cast<int>(nodes[p].parent)) {
                nodes[p].parent = nodes[nodes[p].parent].parent;
                p = static_cast<int>(nodes[p].parent);
            }
            return p;
        }

        // Returns {winner, loser}; loser is -1 when p and q were already joined.
        std::pair<int, int> Unite(int p, int q) {
            const int root_p = Root(p);
            const int root_q = Root(q);
            if (root_p == root_q)
                return {root_p, -1};
            const int winner = nodes[root_p].size >= nodes[root_q].size ? root_p : root_q;
            const int loser = winner == root_p ? root_q : root_p;
            nodes[winner].size += nodes[loser].size;
            nodes[loser].parent = static_cast<std::uint32_t>(winner);
            return {winner, loser};
        }

        // Points every node directly at its root.
        void ResolveAll() {
            for (int p = 0; p < static_cast<int>(nodes.size()); ++p)
                nodes[p].parent = static_cast<std::uint32_t>(Root(p));
        }

        std::vector<Node> nodes;
    };

    struct ColorIdEntry {
        std::uint32_t color;
        std::uint32_t id;
    };

    struct ColorCacheEntry {
        ColorIdEntry entry;
        bool valid{false};
    };

    // Per-component color counts. Every slot holds exactly `bins` counters, one per palette
    // color, so adding a count needs no size check; a new palette color grows every slot
    // once (rare: the palette is small and kept between frames).
    struct Histograms {
        void Merge(int id_a, int id_b) {
            auto& a = data[id_a];
            const auto& b = data[id_b];
            for (std::size_t i = 0; i < bins; ++i)
                a[i] += b[i];
            Release(id_b);
        }

        // Slots are recycled: only components still being scanned hold one, so the
        // working set stays small and hot in cache.
        void Release(int id) { free_slots.push_back(id); }

        // Every slot is free at the start of a frame; storage is kept between frames.
        void Reset() {
            free_slots.resize(data.size());
            std::iota(free_slots.begin(), free_slots.end(), 0);
        }

        int AcquireSlot() {
            int slot_id;
            if (free_slots.empty()) {
                slot_id = static_cast<int>(data.size());
                data.emplace_back();
            } else {
                slot_id = free_slots.back();
                free_slots.pop_back();
            }
            data[slot_id].assign(bins, 0);
            return slot_id;
        }

        // Called when the palette gains a color.
        void AddBin() {
            ++bins;
            for (auto& slot : data)
                slot.push_back(0);
        }

        std::vector<std::uint32_t>& Get(int id) { return data[id]; }
        const std::vector<std::uint32_t>& Get(int id) const { return data[id]; }

        std::vector<std::vector<std::uint32_t>> data;
        std::vector<int> free_slots;
        std::size_t bins = 0; // == palette size
    };

    template<typename OnOverlap>
    static void ForEachOverlap(const PaintRun& current, const PaintRuns& previous, int& cursor, const int stop, OnOverlap&& on_overlap) {
        while (cursor != stop) {
            const PaintRun& prev_run = previous[cursor];
            if (prev_run.x1 < current.x0) {
                ++cursor;
                continue;
            }
            if (current.x1 < prev_run.x0)
                break;
            on_overlap(prev_run);
            if (current.x1 <= prev_run.x1)
                break;
            ++cursor;
        }
    }

    // Appends the foreground runs [x0, x1] of one mask row (ids unset). Each block yields
    // a bit per position where foreground starts or stops; set bits alternate start/stop.
    static void AppendRowRuns(const uchar* mask_ptr, int width, const cv::v_uint8& threshold,
                              std::vector<PaintRun>& runs) {
        constexpr int lanes = CV_SIMD_WIDTH;
        bool inside = false; // Foreground at x - 1; the row starts after background.
        int run_start = 0;
        const auto toggle = [&](int at) {
            if (inside)
                runs.push_back({run_start, at - 1, -1});
            else
                run_start = at;
            inside = !inside;
        };

        int x = 0;
        cv::v_uint8 previous = cv::vx_setzero_u8();
        for (; width - x >= lanes; x += lanes) {
            const auto foreground = cv::v_gt(cv::vx_load(mask_ptr + x), threshold);
            // Lane i holds foreground[i - 1]; lane 0 the last lane of the previous block.
            const auto shifted = cv::v_extract<lanes - 1>(previous, foreground);
            previous = foreground;
            for (auto bits = LaneBits(cv::v_xor(foreground, shifted)); bits; bits &= bits - 1)
                toggle(x + __builtin_ctzll(bits) / LaneBitStride);
        }
        for (; x < width; ++x)
            if ((mask_ptr[x] > 127) != inside)
                toggle(x);
        if (inside)
            runs.push_back({run_start, width - 1, -1});
    }

    // Each stripe starts on its own cache line: stripes are adjacent in m_stripes and
    // written by different threads, so neighbours must not share (false sharing).
    struct alignas(64) Stripe {
        std::size_t GetColorId(std::uint32_t color) {
            const std::uint32_t mixed = (color ^ (color >> 16)) * 0x9E3779B1u;
            auto& cached = color_cache[(mixed ^ (mixed >> 16)) & (ColorCacheSize - 1)];
            if (!cached.valid || cached.entry.color != color)
                cached = {color, FindOrInsertColorId(color), true};
            return cached.entry.id;
        }

        std::uint32_t FindOrInsertColorId(std::uint32_t color) {
            const auto entry = std::lower_bound(color_ids.begin(), color_ids.end(), color,
                [](const ColorIdEntry& candidate, std::uint32_t value) {
                    return candidate.color < value;
                });
            if (entry != color_ids.end() && entry->color == color)
                return entry->id;
            const auto id = static_cast<std::uint32_t>(palette.size());
            palette.push_back(color);
            hists.AddBin();
            color_ids.insert(entry, {color, id});
            return id;
        }

        Unions<UnionNode> uni;
        Histograms hists;
        std::array<ColorCacheEntry, ColorCacheSize> color_cache{};
        std::vector<ColorIdEntry> color_ids;
        std::vector<std::uint32_t> palette;
        std::vector<PaintRun> runs;
        std::vector<std::uint32_t> row_start;     // Stripe row -> first index in runs.
        std::vector<std::uint32_t> active_by_row; // Unfinished interior components by first row.
        std::vector<DeferredRun> deferred;
        int begin{0};

        // Indices [first, last) of a stripe row's runs.
        std::pair<int, int> RowRange(int row) const {
            return {static_cast<int>(row_start[row]), static_cast<int>(row_start[row + 1])};
        }
    };

    static void Uncount(Stripe& stripe, const UnionNode& state) {
        if (!state.boundary)
            --stripe.active_by_row[state.first_row - stripe.begin];
    }

    static void MarkBoundary(Stripe& stripe, int root) {
        auto& state = stripe.uni.nodes[root];
        Uncount(stripe, state);
        state.boundary = true;
    }

    static void MergeCompletion(Stripe& stripe, int winner, int loser) {
        auto& a = stripe.uni.nodes[winner];
        const auto& b = stripe.uni.nodes[loser];
        Uncount(stripe, a);
        Uncount(stripe, b);
        a.first_row = std::min(a.first_row, b.first_row);
        a.last_seen = std::max(a.last_seen, b.last_seen);
        a.boundary = a.boundary || b.boundary;
        if (!a.boundary)
            ++stripe.active_by_row[a.first_row - stripe.begin];
    }

    static std::uint32_t SelectColor(const Stripe& stripe, int hist_id) {
        const auto& hist = stripe.hists.Get(hist_id);
        std::size_t best = 0;
        for (std::size_t i = 1; i < hist.size(); ++i)
            if (hist[i] > hist[best] ||
                (hist[i] == hist[best] && stripe.palette[i] < stripe.palette[best]))
                best = i;
        return stripe.palette[best];
    }

    static void FinalizeComponent(Stripe& stripe, int root) {
        auto& node = stripe.uni.nodes[root];
        node.final_color = SelectColor(stripe, node.hist_id);
        stripe.hists.Release(node.hist_id);
        node.hist_id = -1;
        Uncount(stripe, stripe.uni.nodes[root]);
    }

    // Paints one complete row: zeroing it paints the black background, then colored runs
    // are filled. Zeroing a whole row lets AArch64 glibc's memset use DC ZVA, allocating
    // output cache lines without first reading them from DRAM. Stripe-edge component runs
    // are deferred; their zeroed pixels are overwritten by the final pass.
    static void PaintRow(Stripe& stripe, int y, int width, cv::Mat& output) {
        auto* out = output.ptr<cv::Vec3b>(y);
        std::memset(output.ptr<uchar>(y), 0, std::size_t(width) * sizeof(cv::Vec3b));
        const auto [first, last] = stripe.RowRange(y - stripe.begin);
        for (int k = first; k < last; ++k) {
            const auto& run = stripe.runs[k];
            const int root = stripe.uni.Root(run.id);
            if (stripe.uni.nodes[root].boundary)
                stripe.deferred.push_back({static_cast<std::uint32_t>(k), y});
            else
                FillBgrRun(out + run.x0, run.x1 - run.x0 + 1, stripe.uni.nodes[root].final_color);
        }
    }

    // Row y is scanned. Finishes the components of runs [first, last) (the previous row)
    // that row y did not continue, then paints every row whose components are all final.
    // After the last row, y = stripe end: every remaining interior component finishes.
    static void CloseRow(Stripe& stripe, int first, int last, int y, int& painted, int width, cv::Mat& output) {
        auto& uni = stripe.uni;
        for (int k = first; k < last; ++k) {
            const int root = uni.Root(stripe.runs[k].id);
            const auto& state = uni.nodes[root];
            if (state.last_seen != y && !state.boundary && state.hist_id >= 0)
                FinalizeComponent(stripe, root);
        }

        // Rows above the first unfinished interior component are final.
        const int last_row = std::min(y, stripe.begin + static_cast<int>(stripe.active_by_row.size()) - 1);
        while (painted <= last_row && stripe.active_by_row[painted - stripe.begin] == 0)
            PaintRow(stripe, painted++, width, output);
    }

    // Stripe owning a global node id; upper_bound skips stripes without nodes.
    int StripeOf(int global) const {
        return static_cast<int>(std::upper_bound(m_base.begin(), m_base.end(), global) - m_base.begin()) - 1;
    }

    UnionNode& GlobalNode(int global) {
        const int i = StripeOf(global);
        return m_stripes[i].uni.nodes[global - m_base[i]];
    }

    UnionNode& GlobalNodeWithHint(int global, int hint) {
        return m_stripes[hint].uni.nodes[global - m_base[hint]];
    }

    // Moves the loser root's histogram into the winner's, translating palette ids across stripes.
    void MergeBoundaryHistograms(int winner, int loser) {
        const int dest_id = StripeOf(winner);
        const int sorc_id = StripeOf(loser);
        
        auto& destination = m_stripes[dest_id];
        auto& source = m_stripes[sorc_id];

        const int winner_hist = GlobalNodeWithHint(winner, dest_id).hist_id;
        auto& loser_hist = GlobalNodeWithHint(loser, sorc_id).hist_id;

        if (&destination == &source) {
            destination.hists.Merge(winner_hist, loser_hist);
        } else {
            const auto& counts = source.hists.Get(loser_hist);
            for (std::size_t color_id = 0; color_id < counts.size(); ++color_id) {
                if (counts[color_id] == 0)
                    continue;
                const auto destination_id = destination.GetColorId(source.palette[color_id]);
                destination.hists.Get(winner_hist)[destination_id] += counts[color_id];
            }
            source.hists.Release(loser_hist);
        }
        loser_hist = -1;
    }

    std::vector<Stripe> m_stripes;
    std::vector<int> m_base;  // Global id of each stripe's node 0; the last entry is the total.
    Unions<LinkNode> m_links; // Cross-stripe union-find over global ids.
};
