#pragma once

#include "IAlgo.h"

#include <opencv2/core/simd_intrinsics.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <vector>
#include <utility>

#if defined(COLORIST_DEBUG) || defined(COLORIST_TIMING)
#include <iostream>
#endif

#ifdef COLORIST_TIMING
#include <chrono>
#include <iomanip>
#include <sstream>
#endif

class RleAlgo : public IAlgo {
public:
    explicit RleAlgo(int num_threads = 1) { SetNumThreads(num_threads); }

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
            "Setup", "Scan/CCL/hist", "Reconciliation", "Color selection", "Final coloring"
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

        const int mask_block_sz = cv::v_uint8().nlanes;
        const int color_block_sz = cv::v_uint32().nlanes;
        const cv::v_uint8 cmp_block = cv::vx_setall_u8(127);

        cv::parallel_for_(cv::Range(0, stripe_count),
            [&](const cv::Range& range) {
                for (int i = range.start; i < range.end; ++i) {
                    const int stripe_begin = static_cast<int>(
                        std::int64_t(height) * i / stripe_count);
                    const int stripe_end = static_cast<int>(
                        std::int64_t(height) * (i + 1) / stripe_count);

                    auto& stripe = m_stripes[i];

                    stripe.first_row_id = stripe_begin;
                    stripe.last_row_id = stripe_end - 1;

                    auto& uni = stripe.uni;
                    auto& hists = stripe.hists;

                    // Reuse the rolling row buffers; last_row retains the final row.
                    auto& curr_run_row = stripe.temp_row;
                    auto& prev_run_row = stripe.last_row;
                    prev_run_row.clear();
                    curr_run_row.clear();
                    stripe.first_row.clear();

                    // Before the fallback boundary, markers contain both the component ID and
                    // run length. After it, they contain full IDs and coloring reads the mask.
                    bool packed_runs = width <= static_cast<int>(MaxRunLength);
                    stripe.packed_stop_x = 0;
                    stripe.packed_stop_y = packed_runs ? height : 0;

                    std::uint32_t prev_color = 0;
                    std::size_t prev_color_id = 0;
                    int cursor = 0;
                    bool cache_valid = false;

                    for (int y = stripe_begin; y < stripe_end; ++y) {
                        const auto* mask_ptr = mask.ptr<uchar>(y);
                        const auto* cmask_ptr = color_mask.ptr<cv::Vec4b>(y);
                        auto* out_ptr = output.ptr<cv::Vec4b>(y);

                        int c = 0;

                        cursor = 0;

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

                                curr_run_row.push_back({ -1, run_start, run_end });

                                int component_root = -1;

                                ForEachOverlap(curr_run_row.back(), prev_run_row, cursor,
                                    [&](const Run& prev_run) {
                                        const auto prev_root = uni.Root(prev_run.id);

                                        if (component_root == -1) {
                                            component_root = prev_root;
                                            uni.nodes[component_root].size++;
                                        } else {
                                            const auto&[winner, looser] = uni.UniteWinnerLooser(component_root, prev_root);
                                            if (looser != -1) {
                                                hists.Merge(uni.GetHistId(winner), uni.GetHistId(looser));
                                                uni.SetHistId(looser, -1);
                                            }
                                            component_root = winner;
                                        }
                                });

                                if (component_root == -1) {
                                    if (Unlikely(uni.nodes.size() >= node_capacity))
                                        CV_Error_(cv::Error::StsOutOfRange,
                                            ("Stripe %d exceeds the tagged-parent capacity of %llu nodes (%u node bits)",
                                             i, static_cast<unsigned long long>(node_capacity), m_parent_node_bits));
                                    component_root = uni.GetId();
                                    uni.nodes.push_back({static_cast<std::uint32_t>(component_root),
                                                         1, hists.AcquireSlot()});
                                }

                                curr_run_row.back().id = component_root;

                                // All unions for this run are complete; this histogram reference
                                // stays valid while the shared palette grows during color counting.
                                const auto hist_id = uni.GetHistId(component_root);
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
                                        prev_color_id = stripe.GetColorId(color);
                                        cache_valid = true;
                                    }

                                    hists.AddColorFreq(hist_id, prev_color_id, static_cast<std::uint32_t>(x - stretch_start));
                                }

                                if (packed_runs) {
                                    const auto metadata = PackRunMetadata(component_root, c - run_start);
                                    if (static_cast<std::uint32_t>(component_root) > ComponentIdMask ||
                                        metadata == BackgroundMarker) {
                                        packed_runs = false;
                                        stripe.packed_stop_y = y;
                                        stripe.packed_stop_x = run_start;
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
                        if (y == stripe_begin)
                            stripe.first_row = prev_run_row;
                        curr_run_row.clear();
                    }

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

        // Resolve once before selection; coloring never traverses cross-stripe parents.
        for (int i = 0; i < stripe_count; ++i)
            for (std::size_t id = 0; id < m_stripes[i].uni.nodes.size(); ++id)
                GlobalRoot(PackComponent(i, static_cast<std::uint32_t>(id)));

#ifdef COLORIST_TIMING
        record_time(2);
#endif

        // Select a winner once per root, including valid packed color zero.
        for (auto& stripe : m_stripes) {
            for (std::size_t id = 0; id < stripe.uni.nodes.size(); ++id) {
                const auto& node = stripe.uni.nodes[id];
                if (node.parent != id)
                    continue;

                auto& hist = stripe.hists.Get(node.hist_id);
                std::size_t best = 0;
                for (std::size_t h = 1; h < hist.size(); ++h) {
                    if (hist[h] > hist[best] ||
                        (hist[h] == hist[best] && stripe.palette[h] < stripe.palette[best]))
                        best = h;
                }
                hist[0] = stripe.palette[best];
            }
        }

        // Original local slots remain allocated after boundary unions. Cache their
        // global winner here; live root colors were all selected before any copying.
        // No histogram slot is acquired again until the next Compute/reset.
        for (int i = 0; i < stripe_count; ++i) {
            auto& stripe = m_stripes[i];
            for (std::size_t id = 0; id < stripe.uni.nodes.size(); ++id) {
                const auto root = ParentComponent(PackComponent(i, static_cast<std::uint32_t>(id)));
                const auto& owner = m_stripes[GetStripeId(root)];
                const auto& root_node = GetNode(root);
                if (GetStripeId(root) != i || root_node.hist_id != stripe.uni.nodes[id].hist_id)
                    stripe.hists.Get(stripe.uni.nodes[id].hist_id)[0] = owner.hists.data[root_node.hist_id][0];
            }
        }
#ifdef COLORIST_TIMING
        record_time(3);
#endif

        // Jump over unwritten run interiors using their packed lengths. Every
        // position examined is either a run-start marker or initialized background.

        cv::parallel_for_(cv::Range(0, stripe_count),
            [&](const cv::Range& range) {
                for (int i = range.start; i < range.end; ++i) {
                    const auto& stripe = m_stripes[i];
                    for (int row_id = stripe.first_row_id; row_id <= stripe.last_row_id; ++row_id) {
                        auto* out_ptr = output.ptr<cv::Vec4b>(row_id);
                        const int packed_width = row_id < stripe.packed_stop_y ? width :
                                        row_id == stripe.packed_stop_y ? stripe.packed_stop_x : 0;
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
                            const cv::Vec4b color = UnpackColor(stripe.hists.data[stripe.uni.nodes[component_id].hist_id][0]);
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
                                const cv::Vec4b color = UnpackColor(stripe.hists.data[stripe.uni.nodes[component_id].hist_id][0]);
                                while (c < width && mask_ptr[c] > 127)
                                    ++c;
                                std::fill(out_ptr + run_start, out_ptr + c, color);
                            }
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

    static std::uint32_t PackRunMetadata(int component_id, int length) {
        return static_cast<std::uint32_t>(component_id) |
            (static_cast<std::uint32_t>(length) << ComponentIdBits);
    }

#ifdef COLORIST_TIMING
    std::array<double, 5> m_timings_ms{};
#endif

    void ResetWorkingBuffers(int height) {
        // Flat vectors retain capacity; each new node starts with its own parent and size 1.
        const auto expected_runs = static_cast<std::size_t>(height) / m_stripes.size() * 10;

        for (auto& stripe : m_stripes) {
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

private:
    int m_num_threads = 1;
    static constexpr std::uint32_t CrossStripeParent = 0x80000000u;
    // Component references always carry the tag, so zero is an invalid reference.
    static constexpr std::uint32_t InvalidComponent = 0;
    unsigned m_parent_node_bits = 31;
    std::uint32_t m_parent_node_mask = 0x7FFFFFFFu;

    // Resolve recurring colors through the cache before searching the sorted lookup.
    static constexpr bool UseColorCache = true;
    static constexpr std::size_t ColorCacheSize = 64;
    static_assert(ColorCacheSize != 0 && (ColorCacheSize & (ColorCacheSize - 1)) == 0,
        "Color cache size must be a power of two");

    static constexpr unsigned ComponentIdBits = 20;
    static constexpr std::uint32_t ComponentIdMask = (std::uint32_t{1} << ComponentIdBits) - 1;
    static constexpr std::uint32_t MaxRunLength = (std::uint32_t{1} << (32 - ComponentIdBits)) - 1;
    static constexpr std::uint32_t BackgroundMarker = 0xFF000000u;

    struct Run {
        int id;
        int x0;
        int x1;
    };

    using Runs = std::vector<Run>;

    struct UnionNode {
        std::uint32_t parent;
        std::uint32_t size;
        int hist_id{-1};
    };

    struct Unions {
        // Local traversal is valid during scanning, before cross-stripe unions.
        int Root(int p) {
            while (p != nodes[p].parent) {
                nodes[p].parent = nodes[nodes[p].parent].parent;
                p = nodes[p].parent;
            }
            return p;
        }

        void Unite(int p, int q) {
            int root_p = Root(p);
            int root_q = Root(q);

            if (root_p == root_q)
                return;

            auto& node_p = nodes[root_p];
            auto& node_q = nodes[root_q];
            if (node_p.size < node_q.size) {
                node_p.parent = root_q;
                node_q.size += node_p.size;
            } else {
                node_q.parent = root_p;
                node_p.size += node_q.size;
            }
        }

        std::pair<int, int> UniteWinnerLooser(int p, int q) {
            int root_p = Root(p);
            int root_q = Root(q);

            if (root_p == root_q)
                return {root_p, -1};

            auto& node_p = nodes[root_p];
            auto& node_q = nodes[root_q];

            int winner = node_p.size >= node_q.size ? root_p : root_q;
            int loser = winner == root_p ? root_q : root_p;

            nodes[winner].size += nodes[loser].size;
            nodes[loser].parent = winner;

            return {winner, loser};
        }

        const int GetId() const { return static_cast<int>(nodes.size()); }

        const int GetHistId(int node_id) const { return nodes[node_id].hist_id; }

        void SetHistId(int node_id, int value) {
            nodes[node_id].hist_id = value;
        }

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
            auto& slot_a = data[id_a];
            auto& slot_b = data[id_b];

            if (slot_a.size() < slot_b.size()) {
                slot_a.resize(slot_b.size());
            }
            for (std::size_t i = 0; i < slot_b.size(); ++i) {
                slot_a[i] += slot_b[i];
            }

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
            auto& inner = data[slot_id];
            std::fill(inner.begin(), inner.end(), std::uint32_t{0});
            return slot_id;
        }

        void AddColorFreq(int hist_id, std::size_t color_id, std::uint32_t freq) {
            auto& hist = Get(hist_id);
            if (hist.size() <= color_id) {
                hist.resize(color_id + 1);
            }
            hist[color_id] += freq;
        }

        std::vector<std::uint32_t>& Get(int id) { return data[id]; }

        std::vector<std::vector<std::uint32_t>> data;
        std::vector<int> free_slots;
    };

    template<typename OnOverlap>
    static void ForEachOverlap(const Run& current, const Runs& previous, int& cursor, OnOverlap&& on_overlap) {

        while (cursor != previous.size()) {
            const Run& prev_run = previous[cursor];

            if (prev_run.x1 < current.x0) {
                ++cursor; // Previous run ends before current starts
                continue;
            }
            if (current.x1 < prev_run.x0) {
                break;
            }

            on_overlap(prev_run);

            if (current.x1 <= prev_run.x1) {
                break;
            }
            ++cursor;
        }
    }

    struct Stripe {
    public:
        std::size_t GetColorId(std::uint32_t color) {
            if constexpr (UseColorCache) {
                // Mix all BGRA bytes into a slot. A full-key check makes collisions safe.
                const std::uint32_t mixed = (color ^ (color >> 16)) * 0x9E3779B1u;
                auto& cached = color_cache[(mixed ^ (mixed >> 16)) & (ColorCacheSize - 1)];
                if (!cached.valid || cached.entry.color != color)
                    cached = {color, FindOrInsertColorId(color), true};
                return cached.entry.id;
            }
            return FindOrInsertColorId(color);
        }

        std::uint32_t FindOrInsertColorId(std::uint32_t color) {
            const auto entry = std::lower_bound(color_ids.begin(), color_ids.end(), color,
                [](const ColorIdEntry& candidate, std::uint32_t value) {
                    return candidate.color < value;
                });
            if (entry != color_ids.end() && entry->color == color)
                return entry->id;

            // Sorted positions may change; histogram IDs index the append-only palette.
            // Packed BGRA has at most 2^32 colors, so every palette ID fits in 32 bits.
            const auto id = static_cast<std::uint32_t>(palette.size());
            palette.push_back(color);
            color_ids.insert(entry, {color, id});
            return id;
        }

    public:
        int first_row_id{-1}, last_row_id{-1};
        int packed_stop_x{0}, packed_stop_y{0};
        Unions uni;
        Histograms hists;
        std::array<ColorCacheEntry, ColorCacheSize> color_cache{};
        std::vector<ColorIdEntry> color_ids;
        std::vector<std::uint32_t> palette;
        std::vector<Run> first_row, last_row, temp_row;
    };

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

#ifdef COLORIST_DEBUG
        std::cout << "Parent encoding: " << stripe_count << " stripes (IDs 0.."
                  << stripe_count - 1 << "), " << stripe_bits << " stripe bits, "
                  << m_parent_node_bits << " node bits (IDs 0.." << m_parent_node_mask << ")\n";
#endif
    }

    void ValidateParentEncoding() const {
        const std::uint64_t capacity = std::uint64_t{1} << m_parent_node_bits;
        for (std::size_t i = 0; i < m_stripes.size(); ++i) {
            const auto count = m_stripes[i].uni.nodes.size();
            if (Unlikely(count > capacity))
                CV_Error_(cv::Error::StsOutOfRange,
                    ("Stripe %zu has %zu nodes; tagged parents allow %llu nodes (%u node bits)",
                     i, count, static_cast<unsigned long long>(capacity), m_parent_node_bits));
#ifdef COLORIST_DEBUG
            std::cout << "Stripe " << i << ": " << count << '/' << capacity << " nodes -- fits\n";
#endif
        }
    }

    // Used only with stripe/node IDs whose ranges were checked during scanning.
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

        // Compress the complete path, including links into other stripes.
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
        // Keep local hist_id values as color-cache addresses, including the loser's.
        // Freed slots cannot be reused until the next computation begins.
    }

    // Returns the winner and loser so each successful union merges counts once.
    // An invalid loser means the components were already connected.
    std::pair<std::uint32_t, std::uint32_t> ConnectBoundary(std::uint32_t a, std::uint32_t b) {
        a = GlobalRoot(a);
        b = GlobalRoot(b);
        if (a == b)
            return {a, InvalidComponent};

        if (GetNode(a).size < GetNode(b).size)
            std::swap(a, b);

        GetNode(b).parent = EncodeParent(GetStripeId(b), a);
        GetNode(a).size += GetNode(b).size;
        // Histograms retain their stripe-local IDs for the final color cache.
        return {a, b};
    }

    std::vector<Stripe> m_stripes;
};
