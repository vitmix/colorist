#pragma once

#include "IAlgo.h"

#include <opencv2/core/simd_intrinsics.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>
#include <utility>

#if defined(__aarch64__) && defined(__ARM_NEON)
#include <arm_neon.h>
#endif

#ifdef COLORIST_TIMING
#include <chrono>
#include <iomanip>
#include <iostream>
#include <sstream>
#endif

// RLE CCL with watermark ("streaming") painting.
//
// Differences from RleAlgo:
// - The scan writes no output. Finished components only receive their color.
// - Every stripe keeps a per-row count of unfinished interior components, keyed by
//   each component's first row. Once all components starting at or above row W are
//   finished, row W is painted left to right in one pass: background, then colored
//   runs. Output is therefore written once, sequentially and row-ordered, a short
//   distance behind the scan, instead of scattered fills into older rows.
// - Runs of components touching internal stripe edges are skipped and recorded;
//   they are painted after reconciliation.
// - NEON lane search uses a narrowing shift instead of OpenCV's v_signmask.
// - On AArch64, each output row's whole cache lines are allocated with DC ZVA
//   before painting, so stores don't first read those lines from DRAM.
class RleStreamAlgo : public IAlgo {
public:
    explicit RleStreamAlgo(int num_threads = 1, bool zero_allocate = true)
        : m_zva_bytes(zero_allocate ? ZvaBlockBytes() : 0) { SetNumThreads(num_threads); }

    int GetOutputType() const override { return CV_8UC3; }
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

    using IAlgo::PrintTimings;
    void PrintTimings(const std::array<double, 5>& timings) const override {
        static constexpr const char* stages[] = {
            "Setup", "Scan/CCL/paint", "Reconciliation", "Color selection", "Final coloring"
        };
        const char* border = "+------------------+-----------+\n";
        std::ostringstream table;
        table << '\n' << border << "| RLE stage        |        ms |\n" << border
              << std::fixed << std::setprecision(3);
        for (std::size_t i = 0; i < timings.size(); ++i) {
            table << "| " << std::left << std::setw(16) << stages[i]
                  << " | " << std::right << std::setw(9) << timings[i] << " |\n";
        }
        table << border;
        std::cout << table.str();
    }
#endif

    void Compute(const cv::Mat &mask, const cv::Mat &color_mask, cv::Mat& output) override {
        CV_Assert(mask.type() == CV_8UC1);
        CV_Assert(color_mask.type() == CV_8UC4);
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
        ConfigureParentEncoding(stripe_count);
        const std::uint64_t node_capacity = std::uint64_t{1} << m_parent_node_bits;

        ResetWorkingBuffers(height);

#ifdef COLORIST_TIMING
        record_time(0);
#endif

        cv::parallel_for_(cv::Range(0, stripe_count),
            [&](const cv::Range& range) {
                const int mask_block_sz = cv::VTraits<cv::v_uint8>::vlanes();
                const int color_block_sz = cv::VTraits<cv::v_uint32>::vlanes();
                const cv::v_uint8 cmp_block = cv::vx_setall_u8(127);

                for (int i = range.start; i < range.end; ++i) {
                    const int stripe_begin = static_cast<int>(std::int64_t(height) * i / stripe_count);
                    const int stripe_end = static_cast<int>(std::int64_t(height) * (i + 1) / stripe_count);
                    const int stripe_rows = stripe_end - stripe_begin;

                    auto& stripe = m_stripes[i];
                    stripe.begin = stripe_begin;
                    stripe.row_start.assign(stripe_rows + 1, 0);
                    stripe.active_by_row.assign(stripe_rows, 0);

                    auto& uni = stripe.uni;
                    auto& hists = stripe.hists;

                    auto& curr_run_row = stripe.temp_row;
                    auto& prev_run_row = stripe.last_row;
                    prev_run_row.clear();
                    curr_run_row.clear();
                    stripe.first_row.clear();

                    std::uint32_t prev_color = 0;
                    std::size_t prev_color_id = 0;
                    bool cache_valid = false;
                    int painted = stripe_begin;

                    for (int y = stripe_begin; y < stripe_end; ++y) {
                        const auto* mask_ptr = mask.ptr<uchar>(y);
                        const auto* cmask_ptr = color_mask.ptr<cv::Vec4b>(y);
                        const bool edge_row = (y == stripe_begin && stripe_begin > 0) ||
                                              (y == stripe_end - 1 && stripe_end < height);
                        stripe.row_start[y - stripe_begin] = static_cast<std::uint32_t>(stripe.runs.size());

                        int c = 0;
                        int cursor = 0;

                        while (c < width) {
                            // Skip background; it is painted with the row later.
                            while (width - c >= mask_block_sz) {
                                const int n = FirstSetLane(cv::v_gt(cv::vx_load(mask_ptr + c), cmp_block));
                                c += n;
                                if (n < mask_block_sz)
                                    break;
                            }
                            while (c < width && mask_ptr[c] <= 127)
                                ++c;
                            if (c == width)
                                break;

                            const int run_start = c;
                            while (width - c >= mask_block_sz) {
                                const int n = FirstSetLane(cv::v_le(cv::vx_load(mask_ptr + c), cmp_block));
                                c += n;
                                if (n < mask_block_sz)
                                    break;
                            }
                            while (c < width && mask_ptr[c] > 127)
                                ++c;

                            const Run current{ -1, run_start, c - 1 };
                            int component_root = -1;

                            ForEachOverlap(current, prev_run_row, cursor,
                                [&](const Run& prev_run) {
                                    const auto prev_root = uni.Root(prev_run.id);
                                    if (component_root == -1) {
                                        component_root = prev_root;
                                        uni.nodes[component_root].size++;
                                    } else {
                                        const auto [winner, loser] = uni.UniteWinnerLoser(component_root, prev_root);
                                        if (loser != -1) {
                                            hists.Merge(uni.GetHistId(winner), uni.GetHistId(loser));
                                            uni.SetHistId(loser, -1);
                                            MergeCompletion(stripe, winner, loser);
                                        }
                                        component_root = winner;
                                    }
                                });

                            if (component_root == -1) {
                                if (Unlikely(uni.nodes.size() >= node_capacity))
                                    CV_Error_(cv::Error::StsOutOfRange,
                                        ("Stripe %d exceeds the tagged-parent capacity of %llu nodes (%u parent bits)",
                                         i, static_cast<unsigned long long>(node_capacity), m_parent_node_bits));
                                component_root = uni.GetId();
                                uni.nodes.push_back({static_cast<std::uint32_t>(component_root),
                                                     {1}, hists.AcquireSlot()});
                                stripe.completion.push_back({y, y, false});
                                ++stripe.active_by_row[y - stripe_begin];
                            }

                            curr_run_row.push_back({component_root, run_start, c - 1});

                            const auto hist_id = uni.GetHistId(component_root);
                            int x = run_start;
                            while (x < c) {
                                const auto color = LoadPacked(cmask_ptr[x]);
                                const auto repeated_color = cv::vx_setall_u32(color);
                                const int stretch_start = x++;

                                while (c - x >= color_block_sz) {
                                    const auto pixels = cv::v_reinterpret_as_u32(
                                        cv::vx_load(reinterpret_cast<const uchar*>(cmask_ptr + x)));
                                    const int n = FirstSetLane(cv::v_ne(pixels, repeated_color));
                                    x += n;
                                    if (n < color_block_sz)
                                        break;
                                }
                                while (x < c && LoadPacked(cmask_ptr[x]) == color)
                                    ++x;

                                if (!cache_valid || color != prev_color) {
                                    prev_color = color;
                                    prev_color_id = stripe.GetColorId(color);
                                    cache_valid = true;
                                }
                                hists.AddColorFreq(hist_id, prev_color_id,
                                                   static_cast<std::uint32_t>(x - stretch_start));
                            }

                            stripe.runs.push_back({run_start, c - run_start, component_root});
                            auto& state = stripe.completion[component_root];
                            state.last_seen = y;
                            // Components touching internal stripe edges wait for reconciliation.
                            if (edge_row)
                                MarkBoundary(stripe, component_root);
                        }

                        // Previous-row components absent from this row are complete.
                        for (const auto& run : prev_run_row) {
                            const int root = uni.Root(run.id);
                            const auto& state = stripe.completion[root];
                            if (state.last_seen != y && !state.boundary && uni.GetHistId(root) >= 0)
                                FinalizeComponent(stripe, root);
                        }
                        stripe.row_start[y - stripe_begin + 1] = static_cast<std::uint32_t>(stripe.runs.size());

                        // Rows above the first unfinished interior component are final.
                        while (painted <= y && stripe.active_by_row[painted - stripe_begin] == 0)
                            PaintRow(stripe, painted++, width, output, m_zva_bytes);

                        std::swap(prev_run_row, curr_run_row);
                        if (y == stripe_begin)
                            stripe.first_row = prev_run_row;
                        curr_run_row.clear();
                    }

                    // The image bottom is definite; internal stripe edges remain pending.
                    for (const auto& run : prev_run_row) {
                        const int root = uni.Root(run.id);
                        if (!stripe.completion[root].boundary && uni.GetHistId(root) >= 0)
                            FinalizeComponent(stripe, root);
                    }
                    while (painted < stripe_end)
                        PaintRow(stripe, painted++, width, output, m_zva_bytes);

                    // Local root resolving
                    for (std::size_t comp_id = 0; comp_id < uni.nodes.size(); ++comp_id) {
                        const auto root = uni.Root(static_cast<int>(comp_id));
                        uni.nodes[comp_id].parent = root;
                        uni.nodes[comp_id].hist_id = uni.nodes[root].hist_id;
                    }
                }
            }, m_num_threads
        );

#ifdef COLORIST_TIMING
        record_time(1);
#endif

        ValidateParentEncoding();

        // Local parents stay unchanged; only cross-stripe parents carry a tag.
        for (int i = 0; i + 1 < stripe_count; ++i) {
            auto& upper = m_stripes[i];
            auto& lower = m_stripes[i + 1];
            int cursor = 0;

            for (const auto& current : lower.first_row) {
                ForEachOverlap(current, upper.last_row, cursor,
                    [&](const Run& previous) {
                        const auto [winner, loser] = ConnectBoundary(PackComponent(i, previous.id),
                                                                      PackComponent(i + 1, current.id));
                        if (loser != InvalidComponent)
                            MergeBoundaryHistograms(winner, loser);
                    });
            }
        }

        for (int i = 0; i < stripe_count; ++i)
            for (std::size_t id = 0; id < m_stripes[i].uni.nodes.size(); ++id)
                GlobalRoot(PackComponent(i, static_cast<std::uint32_t>(id)));

#ifdef COLORIST_TIMING
        record_time(2);
#endif

        for (auto& stripe : m_stripes) {
            for (std::size_t id = 0; id < stripe.uni.nodes.size(); ++id) {
                auto& node = stripe.uni.nodes[id];
                if (node.parent != id || node.hist_id < 0)
                    continue;
                node.final_color = SelectColor(stripe, node.hist_id);
            }
        }

        for (int i = 0; i < stripe_count; ++i) {
            auto& stripe = m_stripes[i];
            for (std::size_t id = 0; id < stripe.uni.nodes.size(); ++id) {
                const auto root = ParentComponent(PackComponent(i, static_cast<std::uint32_t>(id)));
                stripe.uni.nodes[id].final_color = GetNode(root).final_color;
            }
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
                        FillBgrRun(output.ptr<cv::Vec3b>(deferred.y) + run.x0, run.length,
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
    // Index of the first lane whose comparison result is set, or the lane count.
    template<typename Vec>
    static int FirstSetLane(const Vec& comparison) {
        constexpr int lane_bytes = sizeof(typename cv::VTraits<Vec>::lane_type);
#if defined(__aarch64__) && defined(__ARM_NEON) && CV_SIMD_WIDTH == 16
        // Narrow each 0x00/0xFF byte to a nibble: a scalar bit search replaces
        // v_check_any plus v_signmask's shift/table/add-across sequence.
        const uint8x16_t bytes = cv::v_reinterpret_as_u8(comparison).val;
        const std::uint64_t nibbles = vget_lane_u64(vreinterpret_u64_u8(
            vshrn_n_u16(vreinterpretq_u16_u8(bytes), 4)), 0);
        if (nibbles == 0)
            return cv::VTraits<Vec>::vlanes();
        return __builtin_ctzll(nibbles) / (4 * lane_bytes);
#else
        (void)lane_bytes;
        return cv::v_check_any(comparison) ? cv::v_scan_forward(comparison)
                                           : cv::VTraits<Vec>::vlanes();
#endif
    }

    static void FillBgrRun(cv::Vec3b* destination, int length, std::uint32_t packed_color) {
        const int lanes = cv::VTraits<cv::v_uint8>::vlanes();
        const cv::Vec3b color = UnpackColor(packed_color);
        if (length < lanes) {
            std::fill(destination, destination + length, color);
            return;
        }

        std::array<uchar, 3 * CV_SIMD_WIDTH> pattern;
        cv::v_store_interleave(pattern.data(), cv::vx_setall_u8(color[0]),
                               cv::vx_setall_u8(color[1]), cv::vx_setall_u8(color[2]));
        const auto first = cv::vx_load(pattern.data());
        const auto second = cv::vx_load(pattern.data() + lanes);
        const auto third = cv::vx_load(pattern.data() + 2 * lanes);
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
#endif

    void ResetWorkingBuffers(int height) {
        const auto expected_runs = static_cast<std::size_t>(height) / m_stripes.size() * 10;

        for (auto& stripe : m_stripes) {
            stripe.runs.clear();
            stripe.deferred.clear();
            stripe.completion.clear();
            stripe.uni.nodes.clear();
            stripe.uni.nodes.reserve(expected_runs);

            auto& free_slots = stripe.hists.free_slots;
            auto& hist_data = stripe.hists.data;
            free_slots.clear();
            free_slots.reserve(hist_data.size());
            for (std::size_t i = 0; i < hist_data.size(); ++i)
                free_slots.push_back(static_cast<int>(i));
        }
    }

    static inline std::uint32_t LoadPacked(const cv::Vec4b& pixel) {
        std::uint32_t value;
        std::memcpy(&value, &pixel, sizeof(value));
        return value;
    }

    static cv::Vec3b UnpackColor(std::uint32_t color) {
        return cv::Vec3b(static_cast<uchar>(color), static_cast<uchar>(color >> 8),
                         static_cast<uchar>(color >> 16));
    }

    int m_num_threads = 1;
    std::size_t m_zva_bytes = 0; // DC ZVA block size; zero disables zero-allocation.
    static constexpr std::uint32_t CrossStripeParent = 0x80000000u;
    static constexpr std::uint32_t InvalidComponent = 0;
    unsigned m_parent_node_bits = 31;
    std::uint32_t m_parent_node_mask = 0x7FFFFFFFu;

    static constexpr std::size_t ColorCacheSize = 64;
    static_assert(ColorCacheSize != 0 && (ColorCacheSize & (ColorCacheSize - 1)) == 0,
        "Color cache size must be a power of two");

    struct Run {
        int id;
        int x0;
        int x1;
    };
    using Runs = std::vector<Run>;

    // Row-ordered runs; the row is implied by Stripe::row_start.
    struct PaintRun {
        int x0;
        int length;
        int id; // Node at the time of the scan; resolve with Root().
    };

    struct DeferredRun {
        std::uint32_t run;
        int y;
    };

    struct Completion {
        int first_row;  // Topmost row; keys this component in active_by_row while counted.
        int last_seen;
        bool boundary;  // Touches an internal stripe edge: not counted, painted after reconciliation.
    };

    struct UnionNode {
        std::uint32_t parent;
        union {
            std::uint32_t size;        // Used while establishing connectivity.
            std::uint32_t final_color; // Used once this component is final.
        };
        int hist_id{-1};
    };
    static_assert(sizeof(UnionNode) == 3 * sizeof(std::uint32_t),
        "Caching final colors must not increase the node size");

    struct Unions {
        int Root(int p) {
            while (p != static_cast<int>(nodes[p].parent)) {
                nodes[p].parent = nodes[nodes[p].parent].parent;
                p = nodes[p].parent;
            }
            return p;
        }

        std::pair<int, int> UniteWinnerLoser(int p, int q) {
            int root_p = Root(p);
            int root_q = Root(q);
            if (root_p == root_q)
                return {root_p, -1};
            int winner = nodes[root_p].size >= nodes[root_q].size ? root_p : root_q;
            int loser = winner == root_p ? root_q : root_p;
            nodes[winner].size += nodes[loser].size;
            nodes[loser].parent = winner;
            return {winner, loser};
        }

        int GetId() const { return static_cast<int>(nodes.size()); }
        int GetHistId(int node_id) const { return nodes[node_id].hist_id; }
        void SetHistId(int node_id, int value) { nodes[node_id].hist_id = value; }

        std::vector<UnionNode> nodes;
    };

    struct ColorIdEntry {
        std::uint32_t color;
        std::uint32_t id;
    };

    struct ColorCacheEntry {
        ColorIdEntry entry;
        bool valid{false};
    };

    struct Histograms {
        void Merge(int id_a, int id_b) {
            auto& slot_a = Get(id_a);
            const auto& slot_b = Get(id_b);
            if (slot_a.size() < slot_b.size())
                slot_a.resize(slot_b.size());
            for (std::size_t i = 0; i < slot_b.size(); ++i)
                slot_a[i] += slot_b[i];
            free_slots.push_back(id_b);
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
            auto& inner = Get(slot_id);
            std::fill(inner.begin(), inner.end(), std::uint32_t{0});
            return slot_id;
        }

        void AddColorFreq(int hist_id, std::size_t color_id, std::uint32_t freq) {
            auto& hist = Get(hist_id);
            if (hist.size() <= color_id)
                hist.resize(color_id + 1);
            hist[color_id] += freq;
        }

        std::vector<std::uint32_t>& Get(int id) { return data[id]; }
        const std::vector<std::uint32_t>& Get(int id) const { return data[id]; }

        std::vector<std::vector<std::uint32_t>> data;
        std::vector<int> free_slots;
    };

    template<typename OnOverlap>
    static void ForEachOverlap(const Run& current, const Runs& previous, int& cursor, OnOverlap&& on_overlap) {
        while (cursor != static_cast<int>(previous.size())) {
            const Run& prev_run = previous[cursor];
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

    struct Stripe {
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
            color_ids.insert(entry, {color, id});
            return id;
        }

        Unions uni;
        Histograms hists;
        std::array<ColorCacheEntry, ColorCacheSize> color_cache{};
        std::vector<ColorIdEntry> color_ids;
        std::vector<std::uint32_t> palette;
        std::vector<Run> first_row, last_row, temp_row;
        std::vector<PaintRun> runs;
        std::vector<std::uint32_t> row_start;     // Stripe row -> first index in runs.
        std::vector<std::uint32_t> active_by_row; // Unfinished interior components by first row.
        std::vector<DeferredRun> deferred;
        std::vector<Completion> completion;
        int begin{0};
    };

    static void Uncount(Stripe& stripe, const Completion& state) {
        if (!state.boundary)
            --stripe.active_by_row[state.first_row - stripe.begin];
    }

    static void MarkBoundary(Stripe& stripe, int root) {
        auto& state = stripe.completion[root];
        Uncount(stripe, state);
        state.boundary = true;
    }

    static void MergeCompletion(Stripe& stripe, int winner, int loser) {
        auto& a = stripe.completion[winner];
        const auto& b = stripe.completion[loser];
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
        stripe.hists.free_slots.push_back(node.hist_id);
        node.hist_id = -1;
        Uncount(stripe, stripe.completion[root]);
    }

    // DC ZVA block size in bytes, or zero when unavailable or prohibited.
    static std::size_t ZvaBlockBytes() {
#if defined(__aarch64__) && (defined(__GNUC__) || defined(__clang__))
        std::uint64_t dczid;
        asm volatile("mrs %0, dczid_el0" : "=r"(dczid));
        if (dczid & 0x10) // DZP: DC ZVA prohibited
            return 0;
        return std::size_t{4} << (dczid & 0xF);
#else
        return 0;
#endif
    }

    // Allocates every block lying entirely inside [begin, begin + bytes) as zeros,
    // without reading it from memory. Partial blocks at the edges are left alone,
    // so neighbouring rows (possibly owned by other threads) are never touched.
    static void ZeroAllocate(void* begin, std::size_t bytes, std::size_t block) {
#if defined(__aarch64__) && (defined(__GNUC__) || defined(__clang__))
        const auto start = reinterpret_cast<std::uintptr_t>(begin);
        const auto end = (start + bytes) & ~(block - 1);
        for (auto line = (start + block - 1) & ~(block - 1); line < end; line += block)
            asm volatile("dc zva, %0" : : "r"(line) : "memory");
#else
        (void)begin; (void)bytes; (void)block;
#endif
    }

    // Paints one complete row left to right; stripe-edge component runs are deferred.
    // Deferred pixels may be zeroed here; the final pass overwrites them.
    static void PaintRow(Stripe& stripe, int y, int width, cv::Mat& output, std::size_t zva_bytes) {
        auto* out = output.ptr<cv::Vec3b>(y);
        if (zva_bytes)
            ZeroAllocate(out, std::size_t(width) * sizeof(cv::Vec3b), zva_bytes);
        const int row = y - stripe.begin;
        int x = 0;
        for (auto k = stripe.row_start[row]; k < stripe.row_start[row + 1]; ++k) {
            const auto& run = stripe.runs[k];
            FillBgrRun(out + x, run.x0 - x, 0);
            const int root = stripe.uni.Root(run.id);
            if (stripe.completion[root].boundary)
                stripe.deferred.push_back({k, y});
            else
                FillBgrRun(out + run.x0, run.length, stripe.uni.nodes[root].final_color);
            x = run.x0 + run.length;
        }
        FillBgrRun(out + x, width - x, 0);
    }

    static bool Unlikely(bool condition) {
#if defined(__GNUC__) || defined(__clang__)
        return __builtin_expect(condition, false);
#else
        return condition;
#endif
    }

    void ConfigureParentEncoding(int stripe_count) {
        CV_Assert(stripe_count > 0);
        unsigned stripe_bits = 0;
        for (auto max_id = static_cast<std::uint32_t>(stripe_count - 1); max_id; max_id >>= 1)
            ++stripe_bits;
        m_parent_node_bits = 31 - stripe_bits;
        m_parent_node_mask = (std::uint32_t{1} << m_parent_node_bits) - 1;
    }

    void ValidateParentEncoding() const {
        const std::uint64_t capacity = std::uint64_t{1} << m_parent_node_bits;
        for (std::size_t i = 0; i < m_stripes.size(); ++i) {
            const auto count = m_stripes[i].uni.nodes.size();
            if (Unlikely(count > capacity))
                CV_Error_(cv::Error::StsOutOfRange,
                    ("Stripe %zu has %zu nodes; tagged parents allow %llu nodes (%u node bits)",
                     i, count, static_cast<unsigned long long>(capacity), m_parent_node_bits));
        }
    }

    std::uint32_t PackComponent(int stripe_id, std::uint32_t node_id) const {
        return CrossStripeParent | (static_cast<std::uint32_t>(stripe_id) << m_parent_node_bits) | node_id;
    }

    int GetStripeId(std::uint32_t component) const {
        return static_cast<int>((component & ~CrossStripeParent) >> m_parent_node_bits);
    }

    UnionNode& GetNode(std::uint32_t component) {
        return m_stripes[GetStripeId(component)].uni.nodes[component & m_parent_node_mask];
    }

    std::uint32_t ParentComponent(std::uint32_t component) {
        const auto parent = GetNode(component).parent;
        return (parent & CrossStripeParent) ? parent : PackComponent(GetStripeId(component), parent);
    }

    std::uint32_t EncodeParent(int owner_stripe, std::uint32_t parent) const {
        return GetStripeId(parent) == owner_stripe ? (parent & m_parent_node_mask) : parent;
    }

    std::uint32_t GlobalRoot(std::uint32_t component) {
        auto root = component;
        for (;;) {
            const auto parent = ParentComponent(root);
            if (parent == root)
                break;
            root = parent;
        }
        while (component != root) {
            const auto parent = ParentComponent(component);
            auto& node = GetNode(component);
            node.parent = EncodeParent(GetStripeId(component), root);
            component = parent;
        }
        return root;
    }

    void MergeBoundaryHistograms(std::uint32_t winner, std::uint32_t loser) {
        auto& destination = m_stripes[GetStripeId(winner)];
        auto& source = m_stripes[GetStripeId(loser)];
        const int winner_hist = GetNode(winner).hist_id;
        const int loser_hist = GetNode(loser).hist_id;

        if (GetStripeId(winner) == GetStripeId(loser)) {
            destination.hists.Merge(winner_hist, loser_hist);
        } else {
            const auto& counts = source.hists.Get(loser_hist);
            for (std::size_t color_id = 0; color_id < counts.size(); ++color_id) {
                if (counts[color_id] == 0)
                    continue;
                const auto destination_id = destination.GetColorId(source.palette[color_id]);
                destination.hists.AddColorFreq(winner_hist, destination_id, counts[color_id]);
            }
            source.hists.free_slots.push_back(loser_hist);
        }
    }

    std::pair<std::uint32_t, std::uint32_t> ConnectBoundary(std::uint32_t a, std::uint32_t b) {
        a = GlobalRoot(a);
        b = GlobalRoot(b);
        if (a == b)
            return {a, InvalidComponent};
        if (GetNode(a).size < GetNode(b).size)
            std::swap(a, b);
        GetNode(b).parent = EncodeParent(GetStripeId(b), a);
        GetNode(a).size += GetNode(b).size;
        return {a, b};
    }

    std::vector<Stripe> m_stripes;
};
