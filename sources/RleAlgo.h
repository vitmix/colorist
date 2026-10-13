#pragma once

#include "IAlgo.h"

#include <opencv2/core/simd_intrinsics.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <vector>

#ifdef COLORIST_TIMING
#include <chrono>
#include <iomanip>
#include <iostream>
#include <sstream>
#endif

class RleAlgo : public IAlgo {
public:
    explicit RleAlgo(int num_threads = 1) { SetNumThreads(num_threads); }

    bool SupportsParallelColoring() const override { return true; }

    void SetNumThreads(int num_threads) override {
        CV_Assert(num_threads > 0);
        m_num_threads = num_threads;
    }

    struct Run {
        int id;
        int x0;
        int x1;
    };

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
            "Setup", "Scan/CCL/hist", "Root resolution", "Color selection", "Final coloring"
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
        CV_Assert(output.type() == CV_8UC4);
        CV_Assert(output.size() == mask.size());
        // Every bin and every merged component count is bounded by image area.
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

        const cv::Vec4b black(0, 0, 0, 255);

        int height = mask.rows;
        int width = mask.cols;
        ResetWorkingBuffers(height);

        // Before the fallback boundary, markers contain both the component ID and
        // run length. After it, they contain full IDs and coloring reads the mask.
        bool packed_runs = width <= static_cast<int>(MaxRunLength);
        int packed_stop_y = packed_runs ? height : 0;
        int packed_stop_x = 0;

        int prev_row_begin = 0, prev_row_stop = 0;
        int cursor = 0;

        bool cache_valid = false;
        std::uint32_t prev_color = 0;
        std::size_t prev_color_id = 0;

#ifdef COLORIST_TIMING
        record_time(0);
#endif

        int mask_block_sz = cv::v_uint8().nlanes;
        int color_block_sz = cv::v_uint32().nlanes;
        const cv::v_uint8 cmp_block = cv::vx_setall_u8(127);
        
        std::vector<Run> prev_run_row, curr_run_row;

        // Resolve run connectivity before counting its colors directly into the root.
        for (int y = 0; y < height; ++y) {

            const auto* mask_ptr = mask.ptr<uchar>(y);
            const auto* cmask_ptr = color_mask.ptr<cv::Vec4b>(y);
            auto* out_ptr = output.ptr<cv::Vec4b>(y);

            int c = 0;

            cursor = prev_row_begin;

            while (c < width) {
                if (mask_ptr[c] > 127) {

                    const int run_start = c;
                    
                    while (width - c >= mask_block_sz) {
                        const auto batch = cv::vx_load(mask_ptr + c);
                        const auto background = cv::v_le(batch, cmp_block);

                        if (cv::v_check_any(background)) {
                            c += cv::v_scan_forward(background);
                            break;
                        }

                        c += mask_block_sz;
                    }

                    while (c < width && mask_ptr[c] > 127) {
                        ++c;
                    }

                    const int run_end = c - 1;

                    const int run_id = static_cast<int>(m_nodes.size());
                    curr_run_row.push_back({ -1, run_start, run_end });

                    int component_root = -1;

                    const Run& curr_run = curr_run_row.back();
                    while (cursor != prev_row_stop) {
                        const Run& prev_run = prev_run_row[cursor];

                        if (prev_run.x1 < curr_run.x0) {
                            ++cursor; // Previous run ends before current starts
                            continue;
                        }
                        if (curr_run.x1 < prev_run.x0) {
                            break;
                        }

                        const auto prev_root = Root(prev_run.id);

                        if (component_root == -1) {
                            component_root = prev_root;
                            m_nodes[component_root].size++;
                        } else {
                            component_root = UnionAndMergeHist(component_root, prev_root);
                        }

                        if (curr_run.x1 <= prev_run.x1) {
                            break;
                        }

                        ++cursor;
                    }

                    if (component_root == -1) {
                        component_root = run_id;
                        m_nodes.push_back({component_root, 1, AcquireHistogramSlot()});
                    }

                    curr_run_row.back().id = component_root;

                    // All unions for this run are complete; this histogram reference
                    // stays valid while the shared palette grows during color counting.
                    auto& hist = m_histograms[m_nodes[component_root].hist_id];
                    int x = run_start;
                    while (x < c) {
                        const auto color = PackColor(cmask_ptr[x]);
                        const auto repeated_color = cv::vx_setall_u32(color);
                        const int stretch_start = x++;

                        while (c - x >= color_block_sz) {
                            const auto pixels = cv::v_reinterpret_as_u32(
                                cv::vx_load(reinterpret_cast<const uchar*>(cmask_ptr + x)));
                            
                            const auto mismatches = cv::v_ne(pixels, repeated_color);
                            if (cv::v_check_any(mismatches)) {
                                x += cv::v_scan_forward(mismatches);
                                break;
                            }

                            x += color_block_sz;
                        }

                        while (x < c && PackColor(cmask_ptr[x]) == color)
                            ++x;

                        if (!cache_valid || color != prev_color) {
                            prev_color = color;
                            prev_color_id = GetColorId(color);
                            cache_valid = true;
                        }
                        if (hist.size() <= prev_color_id) {
                            hist.resize(prev_color_id + 1);
                        }
                        hist[prev_color_id] += static_cast<std::uint32_t>(x - stretch_start);
                    }

                    if (packed_runs) {
                        const auto metadata = PackRunMetadata(component_root, c - run_start);
                        if (static_cast<std::uint32_t>(component_root) > ComponentIdMask ||
                            metadata == BackgroundMarker) {
                            packed_runs = false;
                            packed_stop_y = y;
                            packed_stop_x = run_start;
                        } else {
                            out_ptr[run_start] = UnpackColor(metadata);
                        }
                    }
                    if (!packed_runs) {
                        out_ptr[run_start] = PackComponentId(component_root);
                    }
                } else {
                    out_ptr[c] = black;
                    ++c;
                }
            }
            std::swap(prev_run_row, curr_run_row);
            curr_run_row.clear();
            prev_row_begin = 0;
            prev_row_stop = static_cast<int>(prev_run_row.size());
        }
#ifdef COLORIST_TIMING
        record_time(1);
#endif

        // Histograms are already aggregated; resolve every run's final root for coloring.
        for (std::size_t comp_id = 0; comp_id < m_nodes.size(); ++comp_id) {
            const auto root = Root(static_cast<int>(comp_id));
            m_nodes[comp_id].parent = root; // Cache the final root for selection and coloring.
            m_nodes[comp_id].hist_id = m_nodes[root].hist_id;
        }
#ifdef COLORIST_TIMING
        record_time(2);
#endif

        // Select a winner once per root, including valid packed color zero.
        for (std::size_t run_id = 0; run_id < m_nodes.size(); ++run_id) {
            if (m_nodes[run_id].parent != static_cast<int>(run_id))
                continue;

            auto& hist = m_histograms[m_nodes[run_id].hist_id];
            std::size_t best = 0;
            for (std::size_t h = 1; h < hist.size(); ++h) {
                if (hist[h] > hist[best] ||
                    (hist[h] == hist[best] && m_palette[h] < m_palette[best])) {
                    best = h;
                }
            }
            hist[0] = m_palette[best];
        }
#ifdef COLORIST_TIMING
        record_time(3);
#endif

        // Jump over unwritten run interiors using their packed lengths. Every
        // position examined is either a run-start marker or initialized background.

        cv::parallel_for_(cv::Range(0, height),
            [&](const cv::Range& range) {
                for (int row_id = range.start; row_id < range.end; ++row_id) {
                    auto* out_ptr = output.ptr<cv::Vec4b>(row_id);
                    const int packed_width = row_id < packed_stop_y ? width :
                                    row_id == packed_stop_y ? packed_stop_x : 0;
                    int c = 0;

                    while (c < packed_width) {
                        while (c < packed_width && PackColor(out_ptr[c]) == BackgroundMarker)
                            ++c;
                        if (c == packed_width)
                            break;

                        const int run_start = c;
                        const auto metadata = PackColor(out_ptr[run_start]);
                        const int component_id = static_cast<int>(metadata & ComponentIdMask);
                        c += static_cast<int>(metadata >> ComponentIdBits);
                        const cv::Vec4b color = UnpackColor(m_histograms[m_nodes[component_id].hist_id][0]);
                        std::fill(out_ptr + run_start, out_ptr + c, color);
                    }

                    // Wide images, excessive component IDs, or a reserved-marker collision
                    // use full-ID markers and mask boundaries, without clearing interiors.
                    if (c < width) {
                        const auto* mask_ptr = mask.ptr<uchar>(row_id);
                        while (c < width) {
                            while (c < width && mask_ptr[c] <= 127)
                                ++c;
                            if (c == width)
                                break;

                            const int run_start = c;
                            const int component_id = UnpackComponentId(out_ptr[run_start]);
                            const cv::Vec4b color = UnpackColor(m_histograms[m_nodes[component_id].hist_id][0]);
                            while (c < width && mask_ptr[c] > 127)
                                ++c;
                            std::fill(out_ptr + run_start, out_ptr + c, color);
                        }
                    }
                }
            }, m_num_threads
        );

#ifdef COLORIST_TIMING
        record_time(4);
#endif
    }

private:
    int m_num_threads = 1;

    // Resolve recurring colors through the cache before searching the sorted lookup.
    static constexpr bool UseColorCache = true;
    static constexpr std::size_t ColorCacheSize = 64;
    static_assert(ColorCacheSize != 0 && (ColorCacheSize & (ColorCacheSize - 1)) == 0,
        "Color cache size must be a power of two");
    struct ColorCacheEntry {
        std::uint32_t color{};
        std::uint32_t id{};
        bool valid{false};
    };

    std::size_t GetColorId(std::uint32_t color) {
        if constexpr (UseColorCache) {
            // Mix all BGRA bytes into a slot. A full-key check makes collisions safe.
            const std::uint32_t mixed = (color ^ (color >> 16)) * 0x9E3779B1u;
            auto& cached = m_color_cache[(mixed ^ (mixed >> 16)) & (ColorCacheSize - 1)];
            if (!cached.valid || cached.color != color)
                cached = {color, FindOrInsertColorId(color), true};
            return cached.id;
        }
        return FindOrInsertColorId(color);
    }

    struct ColorIdEntry {
        std::uint32_t color;
        std::uint32_t id;
    };

    std::uint32_t FindOrInsertColorId(std::uint32_t color) {
        const auto entry = std::lower_bound(m_color_ids.begin(), m_color_ids.end(), color,
            [](const ColorIdEntry& candidate, std::uint32_t value) {
                return candidate.color < value;
            });
        if (entry != m_color_ids.end() && entry->color == color)
            return entry->id;

        // Sorted positions may change; histogram IDs index the append-only palette.
        // Packed BGRA has at most 2^32 colors, so every palette ID fits in 32 bits.
        const auto id = static_cast<std::uint32_t>(m_palette.size());
        m_palette.push_back(color);
        m_color_ids.insert(entry, {color, id});
        return id;
    }

    static constexpr unsigned ComponentIdBits = 20;
    static constexpr std::uint32_t ComponentIdMask = (std::uint32_t{1} << ComponentIdBits) - 1;
    static constexpr std::uint32_t MaxRunLength = (std::uint32_t{1} << (32 - ComponentIdBits)) - 1;
    static constexpr std::uint32_t BackgroundMarker = 0xFF000000u;

    static std::uint32_t PackRunMetadata(int component_id, int length) {
        return static_cast<std::uint32_t>(component_id) |
            (static_cast<std::uint32_t>(length) << ComponentIdBits);
    }

    struct UnionNode {
        int parent;
        std::uint32_t size;
        int hist_id{-1};
    };

#ifdef COLORIST_TIMING
    std::array<double, 5> m_timings_ms{};
#endif

    void ResetWorkingBuffers(int height) {
        // Flat vectors retain capacity; each new node starts with its own parent and size 1.
        m_nodes.clear();
        const auto expected_runs = static_cast<std::size_t>(height) * 10;
        m_nodes.reserve(expected_runs);

        // Previous roots no longer exist; every retained slot is available exactly once.
        // AcquireHistogramSlot zeroes counts when a slot is actually reused.
        m_free_hist_slots.clear();
        m_free_hist_slots.reserve(m_histograms.size());
        for (std::size_t i = 0; i < m_histograms.size(); ++i)
            m_free_hist_slots.push_back(static_cast<int>(i));

        // Sorted lookup, palette and optional cache retain their shared, stable IDs.
    }

    int AcquireHistogramSlot() {
        int hist_id;
        if (m_free_hist_slots.empty()) {
            hist_id = static_cast<int>(m_histograms.size());
            m_histograms.emplace_back();
        } else {
            hist_id = m_free_hist_slots.back();
            m_free_hist_slots.pop_back();
        }
        auto& hist = m_histograms[hist_id];
        std::fill(hist.begin(), hist.end(), std::uint32_t{0});
        return hist_id;
    }

    int Root(int p) {
        while (p != m_nodes[p].parent) {
            m_nodes[p].parent = m_nodes[m_nodes[p].parent].parent;
            p = m_nodes[p].parent;
        }
        return p;
    }

    void Union(int p, int q) {
        int root_p = Root(p);
        int root_q = Root(q);

        if (root_p == root_q)
            return;

        auto& node_p = m_nodes[root_p];
        auto& node_q = m_nodes[root_q];
        if (node_p.size < node_q.size) {
            node_p.parent = root_q;
            node_q.size += node_p.size;
        } else {
            node_q.parent = root_p;
            node_p.size += node_q.size;
        }
    }

    void MergeHist(std::vector<std::uint32_t>& a, const std::vector<std::uint32_t>& b) {
        if (a.size() < b.size()) {
            a.resize(b.size());
        }

        for (std::size_t i = 0; i < b.size(); ++i) {
            a[i] += b[i];
        }
    }

    int UnionAndMergeHist(int p, int q) {
        int root_p = Root(p);
        int root_q = Root(q);

        if (root_p == root_q)
            return root_p;

        auto& node_p = m_nodes[root_p];
        auto& node_q = m_nodes[root_q];

        int winner = node_p.size >= node_q.size ? root_p : root_q;
        int loser = winner == root_p ? root_q : root_p;

        MergeHist(m_histograms[m_nodes[winner].hist_id],
                  m_histograms[m_nodes[loser].hist_id]);

        m_nodes[loser].parent = winner;
        m_nodes[winner].size += m_nodes[loser].size;

        m_free_hist_slots.push_back(m_nodes[loser].hist_id);
        m_nodes[loser].hist_id = -1;

        return winner;
    }

    static std::uint32_t PackColor(const cv::Vec4b& color)
    {
        return std::uint32_t(color[0]) | (std::uint32_t(color[1]) << 8) |
            (std::uint32_t(color[2]) << 16) | (std::uint32_t(color[3]) << 24);
    }

    static cv::Vec4b UnpackColor(std::uint32_t color)
    {
        return cv::Vec4b(static_cast<uchar>(color), static_cast<uchar>(color >> 8),
                        static_cast<uchar>(color >> 16), static_cast<uchar>(color >> 24));
    }

    static cv::Vec4b PackComponentId(int component_id) {
        return cv::Vec4b(static_cast<uchar>(component_id), static_cast<uchar>(component_id >> 8),
                         static_cast<uchar>(component_id >> 16), static_cast<uchar>(component_id>> 24));
    }

    static int UnpackComponentId(const cv::Vec4b& packed_root) {
        return int(packed_root[0]) | (int(packed_root[1]) << 8) |
              (int(packed_root[2]) << 16) | (int(packed_root[3]) << 24);
    }

    std::vector<UnionNode> m_nodes;
    std::array<ColorCacheEntry, ColorCacheSize> m_color_cache{};
    std::vector<ColorIdEntry> m_color_ids;
    std::vector<std::uint32_t> m_palette;
    std::vector<std::vector<std::uint32_t>> m_histograms;
    std::vector<int> m_free_hist_slots;
};
