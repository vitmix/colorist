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
        int id;
        int y;
        int x0;
        int x1;
    };

#ifdef COLORIST_TIMING
    const std::array<double, 5>& GetTimings() const override { return m_timings_ms; }

    const std::array<const char*, 5>& GetTimingKeys() const override {
        static constexpr std::array<const char*, 5> keys = {
            "setup_ms", "scan_ccl_histogram_ms", "histogram_aggregation_ms",
            "color_selection_ms", "final_coloring_ms"
        };
        return keys;
    }

    void PrintTimings() const override {
        static constexpr const char* stages[] = {
            "Setup", "Scan/CCL/hist", "Hist aggregation", "Color selection", "Final coloring"
        };
        const char* border = "+------------------+-----------+\n";
        std::ostringstream table;
        table << '\n' << border << "| RLE stage        |        ms |\n" << border
              << std::fixed << std::setprecision(3);
        for (std::size_t i = 0; i < m_timings_ms.size(); ++i) {
            table << "| " << std::left << std::setw(16) << stages[i]
                  << " | " << std::right << std::setw(9) << m_timings_ms[i] << " |\n";
        }
        table << border;
        std::cout << table.str();
    }
#endif

    cv::Mat Compute(const cv::Mat &mask, const cv::Mat &color_mask) override {
        CV_Assert(mask.type() == CV_8UC1);
        CV_Assert(color_mask.type() == CV_8UC4);
        CV_Assert(mask.size() == color_mask.size());
        CV_Assert(!mask.empty());

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

        int prev_row_run_id = 0, prev_row_run_id_stop = 0;
        int prr = 0;

        bool cache_valid = false;
        std::uint32_t prev_color = 0;
        std::size_t prev_color_id = 0;

#ifdef COLORIST_TIMING
        record_time(0);
#endif
        // Scanning, connectivity and per-run histograms share the same pass.
        for (int y = 0; y < height; ++y) {
            
            const auto* mask_ptr = mask.ptr<uchar>(y);
            const auto* cmask_ptr = color_mask.ptr<cv::Vec4b>(y);

            int c = 0;
            int curr_row_run_id = static_cast<int>(runs.size());
            prr = prev_row_run_id;
            
            while (c < width) {
                if (mask_ptr[c] > 127) {
                    const int run_id = static_cast<int>(runs.size());
                    if (histogram.size() == runs.size())
                        histogram.emplace_back();
                    auto& hist = histogram[run_id];

                    int run_start = c;
                    while (c < width && mask_ptr[c] > 127) {
                        auto color = PackColor(cmask_ptr[c]);

                        if (!cache_valid || color != prev_color) {
                            const auto [entry, inserted] = color_ids.try_emplace(color, palette.size());
                            if (inserted) {
                                palette.push_back(color);
                            }
                            prev_color = color;
                            prev_color_id = entry->second;
                            cache_valid = true;
                        }
                        if (hist.size() <= prev_color_id) {
                            hist.resize(prev_color_id + 1);
                        }
                        ++hist[prev_color_id];
                        ++c;
                    }
                    int run_end = c - 1;

                    runs.push_back({ run_id, y, run_start, run_end });
                    connectivity.push_back(run_id);
                    sizes.push_back(1);
                    
                    const Run& curr_run = runs.back();
                    while (prr != prev_row_run_id_stop) {
                        const Run& prev_run = runs[prr];
    
                        if (prev_run.x1 < curr_run.x0) {
                            ++prr; // Previous run ends before current starts
                        }
                        else if (curr_run.x1 < prev_run.x0) {
                            break;
                        }
                        else {
                            // Runs overlap: same connected component
                            Union(prev_run.id, curr_run.id);
    
                            // Advance whichever run ends first
                            if (prev_run.x1 < curr_run.x1)
                                ++prr;
                            else
                                break;
                        }
                    }
                } else {
                    ++c;
                }
            }
            prev_row_run_id = curr_row_run_id;
            prev_row_run_id_stop = static_cast<int>(runs.size());
        }
#ifdef COLORIST_TIMING
        record_time(1);
#endif

        // Aggregate each run's histogram into its final component root.
        for (const auto & run : runs) {
            const auto root = Root(run.id);
            connectivity[run.id] = root; // Cache the final root for selection and coloring.
            if (root != run.id) {
                auto& root_hist = histogram[root];
                auto& run_hist = histogram[run.id];
                
                if (root_hist.size() < run_hist.size()) {
                    root_hist.resize(run_hist.size());
                }

                for (std::size_t i = 0; i < run_hist.size(); ++i) {
                    root_hist[i] += run_hist[i];
                }
            }
        }
#ifdef COLORIST_TIMING
        record_time(2);
#endif

        // Select a winner once per root, including valid packed color zero.
        std::vector<cv::Vec4b> final_colors(runs.size());
        for (const auto& run : runs) {
            if (connectivity[run.id] != run.id)
                continue;

            const auto& hist = histogram[run.id];
            std::size_t best = 0;
            for (std::size_t h = 1; h < hist.size(); ++h) {
                if (hist[h] > hist[best] ||
                    (hist[h] == hist[best] && palette[h] < palette[best])) {
                    best = h;
                }
            }
            final_colors[run.id] = UnpackColor(palette[best]);
        }
#ifdef COLORIST_TIMING
        record_time(3);
#endif

        // Background is opaque black; each run includes both endpoints.
        cv::Mat output(height, width, CV_8UC4, cv::Scalar(0, 0, 0, 255));
        for (const auto& run : runs) {
            auto* row = output.ptr<cv::Vec4b>(run.y);
            std::fill(row + run.x0, row + run.x1 + 1,
                      final_colors[connectivity[run.id]]);
        }
#ifdef COLORIST_TIMING
        record_time(4);
#endif
        return output;
    }

private:
#ifdef COLORIST_TIMING
    std::array<double, 5> m_timings_ms{};
#endif

    void ResetWorkingBuffers(int height) {
        // Flat vectors retain capacity; parents and union sizes are recreated per run.
        runs.clear();
        connectivity.clear();
        sizes.clear();
        const auto expected_runs = static_cast<std::size_t>(height) * 10;
        runs.reserve(expected_runs);
        connectivity.reserve(expected_runs);
        sizes.reserve(expected_runs);

        // Keep every histogram's allocation and reset all counts, including root totals.
        // Only histogram entries indexed by the current runs are used by Compute().
        for (auto& hist : histogram)
            std::fill(hist.begin(), hist.end(), std::size_t{0});

        // color_ids and palette intentionally retain their shared, stable color IDs.
    }

    int Root(int p) {
        while (p != connectivity[p]) {
            connectivity[p] = connectivity[connectivity[p]];
            p = connectivity[p];
        }
        return p;
    }

    void Union(int p, int q) {
        int root_p = Root(p);
        int root_q = Root(q);

        if (root_p == root_q)
            return;

        if (sizes[root_p] < sizes[root_q]) {
            connectivity[root_p] = root_q;
            sizes[root_q] += sizes[root_p];
        } else {
            connectivity[root_q] = root_p;
            sizes[root_p] += sizes[root_q];
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

    std::vector<Run> runs;
    std::vector<int> connectivity;
    std::vector<std::uint32_t> sizes;
    std::unordered_map<std::uint32_t, std::size_t> color_ids;
    std::vector<std::uint32_t> palette;
    std::vector<std::vector<std::size_t>> histogram;
};
