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

        const cv::Vec3b black(0, 0, 0);

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
                    const int stripe_begin = static_cast<int>(
                        std::int64_t(height) * i / stripe_count);
                    const int stripe_end = static_cast<int>(
                        std::int64_t(height) * (i + 1) / stripe_count);

                    auto& stripe = m_stripes[i];

                    auto& uni = stripe.uni;
                    auto& hists = stripe.hists;

                    // Reuse the rolling row buffers; last_row retains the final row.
                    auto& curr_run_row = stripe.temp_row;
                    auto& prev_run_row = stripe.last_row;
                    prev_run_row.clear();
                    curr_run_row.clear();
                    stripe.first_row.clear();

                    std::uint32_t prev_color = 0;
                    std::size_t prev_color_id = 0;
                    int cursor = 0;
                    bool cache_valid = false;

                    for (int y = stripe_begin; y < stripe_end; ++y) {
                        const auto* mask_ptr = mask.ptr<uchar>(y);
                        const auto* cmask_ptr = color_mask.ptr<cv::Vec4b>(y);
                        auto* out_ptr = output.ptr<cv::Vec3b>(y);

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
                                                MergeRunLists(stripe, winner, looser);
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
                                    stripe.completion.emplace_back();
                                }

                                curr_run_row.back().id = component_root;

                                // All unions for this run are complete; this histogram reference
                                // stays valid while the shared palette grows during color counting.
                                const auto hist_id = uni.GetHistId(component_root);
                                int x = run_start;
                                while (x < c) {
                                    const auto color = LoadPacked(cmask_ptr[x]);
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

                                stripe.paint_runs.push_back({y, run_start, c - run_start});
                                AppendComponentRun(stripe, component_root, y);
                                // Components touching internal stripe edges must wait for reconciliation.
                                if ((y == stripe_begin && stripe_begin > 0) ||
                                    (y == stripe_end - 1 && stripe_end < height))
                                    stripe.completion[component_root].boundary = true;
                            } else {
                                const int run_start = c;
                                while (width - c >= mask_block_sz) {
                                    const auto batch = cv::vx_load(mask_ptr + c);
                                    const auto foreground = cv::v_gt(batch, cmp_block);
                                    if (cv::v_check_any(foreground)) {
                                        c += cv::v_scan_forward(foreground);
                                        break;
                                    }
                                    c += mask_block_sz;
                                }
                                while (c < width && mask_ptr[c] <= 127)
                                    ++c;

                                std::fill(out_ptr + run_start, out_ptr + c, black);
                            }
                        }
                        // Resolve row IDs after all unions. A released histogram marks
                        // roots already painted, even if several previous runs reference them.
                        for (const auto& run : prev_run_row) {
                            const int root = uni.Root(run.id);
                            const auto& state = stripe.completion[root];
                            if (state.last_seen != y && !state.boundary && uni.GetHistId(root) >= 0)
                                FinalizeComponent(stripe, root, output);
                        }
                        std::swap(prev_run_row, curr_run_row);
                        if (y == stripe_begin)
                            stripe.first_row = prev_run_row;
                        curr_run_row.clear();
                    }

                    // The image bottom is definite; internal stripe edges remain pending.
                    for (const auto& run : prev_run_row) {
                        const int root = uni.Root(run.id);
                        if (!stripe.completion[root].boundary && uni.GetHistId(root) >= 0)
                            FinalizeComponent(stripe, root, output);
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

        // Unions are finished: reuse each root's size storage for its final color.
        // Select a winner once per root, including valid packed color zero.
        for (auto& stripe : m_stripes) {
            for (std::size_t id = 0; id < stripe.uni.nodes.size(); ++id) {
                auto& node = stripe.uni.nodes[id];
                if (node.parent != id || node.hist_id < 0)
                    continue;
                node.final_color = SelectColor(stripe, node.hist_id);
            }
        }

        // Every parent now points directly to a global root. All root colors were
        // selected above; cache them in the remaining nodes for direct coloring.
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

        // Only boundary component lists remain; interior components were painted in the scan.
        cv::parallel_for_(cv::Range(0, stripe_count),
            [&](const cv::Range& range) {
                for (int i = range.start; i < range.end; ++i) {
                    auto& stripe = m_stripes[i];
                    for (std::size_t id = 0; id < stripe.completion.size(); ++id)
                        if (stripe.completion[id].head != InvalidRun)
                            PaintComponentRuns(stripe, static_cast<int>(id), output,
                                               stripe.uni.nodes[id].final_color);
                }
            }, m_num_threads
        );

#ifdef COLORIST_TIMING
        record_time(4);
#endif
    }

private:
    static void FillBgrRun(cv::Vec3b* destination, int length, std::uint32_t packed_color) {
        const int lanes = cv::VTraits<cv::v_uint8>::vlanes();
        const cv::Vec3b color = UnpackColor(packed_color);
        if (length >= lanes) {
            const auto blue = cv::vx_setall_u8(color[0]);
            const auto green = cv::vx_setall_u8(color[1]);
            const auto red = cv::vx_setall_u8(color[2]);
            do {
                cv::v_store_interleave(reinterpret_cast<uchar*>(destination), blue, green, red);
                destination += lanes;
                length -= lanes;
            } while (length >= lanes);
        }
        std::fill(destination, destination + length, color);
    }

#ifdef COLORIST_TIMING
    std::array<double, 5> m_timings_ms{};
#endif

    void ResetWorkingBuffers(int height) {
        // Flat vectors retain capacity; each new node starts with its own parent and size 1.
        const auto expected_runs = static_cast<std::size_t>(height) / m_stripes.size() * 10;

        for (auto& stripe : m_stripes) {
            stripe.paint_runs.clear();
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

    static inline std::uint32_t LoadPacked(const cv::Vec4b& pixel)
    {
        std::uint32_t value;
        std::memcpy(&value, &pixel, sizeof(value));
        return value;
    }

    static cv::Vec3b UnpackColor(std::uint32_t color)
    {
        return cv::Vec3b(static_cast<uchar>(color), static_cast<uchar>(color >> 8),
                        static_cast<uchar>(color >> 16));
    }

    int m_num_threads = 1;
    static constexpr std::uint32_t CrossStripeParent = 0x80000000u;
    // Component references always carry the tag, so zero is an invalid reference.
    static constexpr std::uint32_t InvalidComponent = 0;
    unsigned m_parent_node_bits = 31;
    std::uint32_t m_parent_node_mask = 0x7FFFFFFFu;

    // Resolve recurring colors through the cache before searching the sorted lookup.
    static constexpr std::size_t ColorCacheSize = 64;
    static_assert(ColorCacheSize != 0 && (ColorCacheSize & (ColorCacheSize - 1)) == 0,
        "Color cache size must be a power of two");

    struct Run {
        int id;
        int x0;
        int x1;
    };

    using Runs = std::vector<Run>;

    static constexpr std::uint32_t InvalidRun = std::numeric_limits<std::uint32_t>::max();
    struct PaintRun {
        int y;
        int x0;
        int length;
        std::uint32_t next{InvalidRun};
    };
    static_assert(sizeof(PaintRun) == 4 * sizeof(std::uint32_t), "Linked paint runs must stay compact");

    struct Completion {
        std::uint32_t head{InvalidRun}, tail{InvalidRun};
        int last_seen{-1};
        bool boundary{false};
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
            auto& slot_a = Get(id_a);
            const auto& slot_b = Get(id_b);

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
            auto& inner = Get(slot_id);
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
        const std::vector<std::uint32_t>& Get(int id) const { return data[id]; }

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
            // Mix all BGRA bytes into a slot. A full-key check makes collisions safe.
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

            // Sorted positions may change; histogram IDs index the append-only palette.
            // Packed BGRA has at most 2^32 colors, so every palette ID fits in 32 bits.
            const auto id = static_cast<std::uint32_t>(palette.size());
            palette.push_back(color);
            color_ids.insert(entry, {color, id});
            return id;
        }

    public:
        Unions uni;
        Histograms hists;
        std::array<ColorCacheEntry, ColorCacheSize> color_cache{};
        std::vector<ColorIdEntry> color_ids;
        std::vector<std::uint32_t> palette;
        std::vector<Run> first_row, last_row, temp_row;
        std::vector<PaintRun> paint_runs;
        // Node IDs stay stable; only histogram slots are recycled.
        std::vector<Completion> completion;
    };

    static void AppendComponentRun(Stripe& stripe, int root, int row) {
        const auto index = static_cast<std::uint32_t>(stripe.paint_runs.size() - 1);
        auto& state = stripe.completion[root];
        if (state.tail == InvalidRun)
            state.head = index;
        else
            stripe.paint_runs[state.tail].next = index;
        state.tail = index;
        state.last_seen = row;
    }

    static void MergeRunLists(Stripe& stripe, int winner, int loser) {
        auto& a = stripe.completion[winner];
        auto& b = stripe.completion[loser];
        if (b.head != InvalidRun) {
            if (a.tail == InvalidRun)
                a.head = b.head;
            else
                stripe.paint_runs[a.tail].next = b.head;
            a.tail = b.tail;
        }
        a.last_seen = std::max(a.last_seen, b.last_seen);
        a.boundary = a.boundary || b.boundary;
        b.head = b.tail = InvalidRun;
    }

    static void PaintComponentRuns(Stripe& stripe, int root, cv::Mat& output, std::uint32_t color) {
        auto& state = stripe.completion[root];
        for (auto index = state.head; index != InvalidRun; index = stripe.paint_runs[index].next) {
            const auto& run = stripe.paint_runs[index];
            FillBgrRun(output.ptr<cv::Vec3b>(run.y) + run.x0, run.length, color);
        }
        state.head = state.tail = InvalidRun;
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

    static void FinalizeComponent(Stripe& stripe, int root, cv::Mat& output) {
        auto& node = stripe.uni.nodes[root];
        node.final_color = SelectColor(stripe, node.hist_id);
        PaintComponentRuns(stripe, root, output, node.final_color);
        stripe.hists.free_slots.push_back(node.hist_id);
        node.hist_id = -1;
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
        // Histogram IDs remain local to their owning stripes for the caller's merge.
        return {a, b};
    }

    std::vector<Stripe> m_stripes;
};
