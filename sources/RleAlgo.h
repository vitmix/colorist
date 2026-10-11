#pragma once

#include "IAlgo.h"

#include <algorithm>
#include <cstdint>
#include <vector>
#include <unordered_map>

#ifdef COLORIST_TIMING
#include <array>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <sstream>
#endif

class RleAlgo : public IAlgo {
public:
    struct Run {
        int y;
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

#ifdef COLORIST_TIMING
        using Clock = std::chrono::steady_clock;
        auto stage_start = Clock::now();
        const auto record_time = [this, &stage_start](std::size_t stage) {
            const auto now = Clock::now();
            m_timings_ms[stage] = std::chrono::duration<double, std::milli>(now - stage_start).count();
            stage_start = now;
        };
#endif

        int height = mask.rows;
        int width = mask.cols;
        ResetWorkingBuffers(height);

        int prev_row_begin = 0, prev_row_stop = 0;
        int cursor = 0;

        bool cache_valid = false;
        std::uint32_t prev_color = 0;
        std::size_t prev_color_id = 0;

#ifdef COLORIST_TIMING
        record_time(0);
#endif

        std::vector<std::size_t> temp_hist;

        // Scan run statistics and combine component histograms during connectivity updates.
        for (int y = 0; y < height; ++y) {

            const auto* mask_ptr = mask.ptr<uchar>(y);
            const auto* cmask_ptr = color_mask.ptr<cv::Vec4b>(y);

            int c = 0;
            int curr_row_start = static_cast<int>(m_runs.size());

            cursor = prev_row_begin;

            while (c < width) {
                if (mask_ptr[c] > 127) {

                    std::fill(temp_hist.begin(), temp_hist.end(), 0);

                    int run_start = c;
                    while (c < width && mask_ptr[c] > 127) {
                        auto color = PackColor(cmask_ptr[c]);

                        if (!cache_valid || color != prev_color) {
                            const auto [entry, inserted] = m_color_ids.try_emplace(color, m_palette.size());
                            if (inserted) {
                                m_palette.push_back(color);
                            }
                            prev_color = color;
                            prev_color_id = entry->second;
                            cache_valid = true;
                        }
                        if (temp_hist.size() <= prev_color_id) {
                            temp_hist.resize(prev_color_id + 1);
                        }
                        ++temp_hist[prev_color_id];
                        ++c;
                    }
                    int run_end = c - 1;

                    const int run_id = static_cast<int>(m_runs.size());
                    m_runs.push_back({ y, run_start, run_end });
                    m_nodes.push_back({run_id, 1, -1});

                    int component_root = -1;

                    const Run& curr_run = m_runs.back();
                    while (cursor != prev_row_stop) {
                        const Run& prev_run = m_runs[cursor];

                        if (prev_run.x1 < curr_run.x0) {
                            ++cursor; // Previous run ends before current starts
                            continue;
                        }
                        if (curr_run.x1 < prev_run.x0) {
                            break;
                        }

                        const auto prev_root = Root(cursor);

                        if (component_root == -1) {
                            component_root = prev_root;
                            m_nodes[run_id].parent = component_root;
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
                        m_nodes[component_root].hist_id = AcquireHistogramSlot();
                    }

                    MergeHist(m_histograms[m_nodes[component_root].hist_id], temp_hist);

                } else {
                    ++c;
                }
            }
            prev_row_begin = curr_row_start;
            prev_row_stop = static_cast<int>(m_runs.size());
        }
#ifdef COLORIST_TIMING
        record_time(1);
#endif

        // Histograms are already aggregated; resolve every run's final root for coloring.
        for (std::size_t run_id = 0; run_id < m_nodes.size(); ++run_id) {
            const auto root = Root(static_cast<int>(run_id));
            m_nodes[run_id].parent = root; // Cache the final root for selection and coloring.
        }
#ifdef COLORIST_TIMING
        record_time(2);
#endif

        // Select a winner once per root, including valid packed color zero.
        std::vector<cv::Vec4b> final_colors(m_runs.size());
        for (std::size_t run_id = 0; run_id < m_nodes.size(); ++run_id) {
            if (m_nodes[run_id].parent != static_cast<int>(run_id))
                continue;

            const auto& hist = m_histograms[m_nodes[run_id].hist_id];
            std::size_t best = 0;
            for (std::size_t h = 1; h < hist.size(); ++h) {
                if (hist[h] > hist[best] ||
                    (hist[h] == hist[best] && m_palette[h] < m_palette[best])) {
                    best = h;
                }
            }
            final_colors[run_id] = UnpackColor(m_palette[best]);
        }
#ifdef COLORIST_TIMING
        record_time(3);
#endif

        // Runs are ordered by row and x. Paint gaps and runs once, without a whole-image clear.
        const cv::Vec4b black(0, 0, 0, 255);
        std::size_t run_index = 0;
        for (int y = 0; y < height; ++y) {
            auto* row = output.ptr<cv::Vec4b>(y);
            int next_x = 0;
            while (run_index < m_runs.size() && m_runs[run_index].y == y) {
                const auto& run = m_runs[run_index];
                std::fill(row + next_x, row + run.x0, black);
                const auto color = final_colors[m_nodes[run_index].parent];
                std::fill(row + run.x0, row + run.x1 + 1, color);
                next_x = run.x1 + 1;
                ++run_index;
            }
            // Also covers completely empty rows and pixels after the last run.
            std::fill(row + next_x, row + width, black);
        }
#ifdef COLORIST_TIMING
        record_time(4);
#endif
    }

private:
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
        m_runs.clear();
        m_nodes.clear();
        const auto expected_runs = static_cast<std::size_t>(height) * 10;
        m_runs.reserve(expected_runs);
        m_nodes.reserve(expected_runs);

        // Previous roots no longer exist; every retained slot is available exactly once.
        // AcquireHistogramSlot zeroes counts when a slot is actually reused.
        m_free_hist_slots.clear();
        m_free_hist_slots.reserve(m_histograms.size());
        for (std::size_t i = 0; i < m_histograms.size(); ++i)
            m_free_hist_slots.push_back(static_cast<int>(i));

        // m_color_ids and m_palette intentionally retain their shared, stable color IDs.
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
        std::fill(hist.begin(), hist.end(), std::size_t{0});
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

    void MergeHist(std::vector<std::size_t>& a, const std::vector<std::size_t>& b) {
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

    std::vector<Run> m_runs;
    std::vector<UnionNode> m_nodes;
    std::unordered_map<std::uint32_t, std::size_t> m_color_ids;
    std::vector<std::uint32_t> m_palette;
    std::vector<std::vector<std::size_t>> m_histograms;
    std::vector<int> m_free_hist_slots;
};
