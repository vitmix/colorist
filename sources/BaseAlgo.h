#pragma once

#include "IAlgo.h"

#include <algorithm>
#include <cstdint>
#include <unordered_map>
#include <vector>

#ifdef COLORIST_TIMING
#include <array>
#include <chrono>
#endif

#if defined(COLORIST_DEBUG) || defined(COLORIST_TIMING)
#include <iomanip>
#include <iostream>
#include <sstream>
#endif

using CclAlgoType = cv::ConnectedComponentsAlgorithmsTypes;

class BaseAlgo : public IAlgo
{
public:
    static constexpr int Connectivity = 4;

    void SetAlgoType(CclAlgoType type)
    {
        m_type = type;
    }

#ifdef COLORIST_TIMING
    const std::array<double, 5>& GetTimings() const { return m_timings_ms; }

    void PrintTimings() const
    {
        static constexpr const char* stages[] = {
            "Threshold", "CCL", "Histogram", "Color selection", "Final coloring"
        };
        const char* border = "+-----------------+-----------+\n";
        std::ostringstream table;
        table << '\n' << border << "| Stage           |        ms |\n" << border
              << std::fixed << std::setprecision(3);
        for (std::size_t i = 0; i < m_timings_ms.size(); ++i)
        {
            table << "| " << std::left << std::setw(15) << stages[i]
                  << " | " << std::right << std::setw(9) << m_timings_ms[i] << " |\n";
        }
        table << border;
        std::cout << table.str();
    }
#endif

    cv::Mat Compute(const cv::Mat &mask, const cv::Mat &color_mask) override
    {
        CV_Assert(mask.type() == CV_8UC1);
        CV_Assert(color_mask.type() == CV_8UC4);
        CV_Assert(mask.size() == color_mask.size());
        CV_Assert(!mask.empty());

        const int height = mask.rows;
        const int width = mask.cols;

#ifdef COLORIST_TIMING
        using Clock = std::chrono::steady_clock;
        auto stage_start = Clock::now();
        const auto record_time = [this, &stage_start](std::size_t stage)
        {
            m_timings_ms[stage] = std::chrono::duration<double, std::milli>(
                Clock::now() - stage_start).count();
        };
#endif
        cv::Mat binary;
        cv::threshold(mask, binary, 127, 255, cv::THRESH_BINARY);
#ifdef COLORIST_TIMING
        record_time(0);
        stage_start = Clock::now();
#endif

        cv::Mat labels;
        const int num_components = cv::connectedComponents(binary, labels, Connectivity, CV_32S, m_type);
#ifdef COLORIST_TIMING
        record_time(1);
#endif

#ifdef COLORIST_DEBUG
        std::cout << "Found " << num_components - 1 << " polygons" << '\n';
#endif

#ifdef COLORIST_TIMING
        stage_start = Clock::now();
#endif
        std::vector<std::unordered_map<std::uint32_t, std::size_t>> hist(num_components);

#ifdef COLORIST_DEBUG
        constexpr std::uint32_t BLACK = 0xFF000000;
        std::vector<std::size_t> ids_of_interest;
#endif

        for (int y = 0; y < height; ++y)
        {
            const auto* lbl_base = labels.ptr<int>(y);
            const auto* cmask_base = color_mask.ptr<cv::Vec4b>(y);

            for (int x = 0; x < width; ++x)
            {
                const auto id = *(lbl_base + x);
                if (id == 0)
                    continue;

                const auto color = PackColor(cmask_base[x]);
#ifdef COLORIST_DEBUG
                if (color == BLACK) {
                    ids_of_interest.push_back(id);
                }
#endif

                ++hist[id][color];
            }
        }

#ifdef COLORIST_TIMING
        record_time(2);
        stage_start = Clock::now();
#endif

        // Label 0 is the background: keep it opaque black, without a histogram.
        std::vector<cv::Vec4b> final_colors(num_components, cv::Vec4b(0, 0, 0, 255));

        for (int comp_id = 1; comp_id < num_components; ++comp_id)
        {
            const auto& comp = hist[comp_id];
            const auto best = std::max_element(comp.begin(), comp.end(),
                                              [](const auto& p1, const auto& p2)
                                              {
                                                  // Resolve ties consistently across labeling algorithms.
                                                  return p1.second < p2.second ||
                                                      (p1.second == p2.second && p1.first > p2.first);
                                              });
            if (best != comp.end()) {
                final_colors[comp_id] = UnpackColor(best->first);
            }
        }
#ifdef COLORIST_TIMING
        record_time(3);
#endif

#ifdef COLORIST_DEBUG
        std::sort(ids_of_interest.begin(), ids_of_interest.end());
        ids_of_interest.erase(std::unique(ids_of_interest.begin(), ids_of_interest.end()),
                              ids_of_interest.end());
        for (const auto id : ids_of_interest)
        {
            std::vector<std::pair<std::uint32_t, std::size_t>> rows(hist[id].begin(), hist[id].end());
            std::sort(rows.begin(), rows.end(), [](const auto& a, const auto& b)
            {
                return a.second > b.second || (a.second == b.second && a.first < b.first);
            });

            const char* border = "+-----+-----+-----+-----+--------------+-----------+----------+\n";
            std::ostringstream table;
            table << "\nHistogram for polygon " << id << '\n'
                  << border
                  << "|   R |   G |   B |   A |        Color | Frequency | Selected |\n"
                  << border;
            for (const auto& [packed, frequency] : rows)
            {
                const auto color = UnpackColor(packed);
                table << "| " << std::setw(3) << int(color[2])
                      << " | " << std::setw(3) << int(color[1])
                      << " | " << std::setw(3) << int(color[0])
                      << " | " << std::setw(3) << int(color[3])
                      << " | " << std::setw(12) << ColorName(color)
                      << " | " << std::setw(9) << frequency
                      << " | " << std::setw(8) << (packed == PackColor(final_colors[id]) ? "yes" : "no")
                      << " |\n";
            }
            table << border;
            std::cout << table.str();
        }
#endif

#ifdef COLORIST_TIMING
        stage_start = Clock::now();
#endif
        cv::Mat output(height, width, CV_8UC4);
        for (int y = 0; y < height; ++y)
        {
            const auto* lbl_base = labels.ptr<int>(y);
            auto* out_base = output.ptr<cv::Vec4b>(y);

            for (int x = 0; x < width; ++x)
            {
                const auto id = lbl_base[x];
                out_base[x] = final_colors[id];
            }
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
#ifdef COLORIST_DEBUG
    // Approximate debug names using the nearest RGB color; alpha is shown separately.
    static const char* ColorName(const cv::Vec4b& color)
    {
        struct NamedColor { const char* name; int r, g, b; };
        static constexpr NamedColor palette[] = {
            {"black", 0, 0, 0}, {"white", 255, 255, 255}, {"gray", 128, 128, 128},
            {"red", 255, 0, 0}, {"orange", 255, 165, 0}, {"yellow", 255, 255, 0},
            {"yellow-green", 128, 255, 0}, {"green", 0, 255, 0}, {"cyan", 0, 255, 255},
            {"blue", 0, 0, 255}, {"purple", 128, 0, 128}, {"magenta", 255, 0, 255}
        };
        const char* name = "black";
        int best_distance = 3 * 255 * 255 + 1;
        for (const auto& candidate : palette)
        {
            const int dr = int(color[2]) - candidate.r;
            const int dg = int(color[1]) - candidate.g;
            const int db = int(color[0]) - candidate.b;
            const int distance = dr * dr + dg * dg + db * db;
            if (distance < best_distance)
            {
                best_distance = distance;
                name = candidate.name;
            }
        }
        return name;
    }
#endif

    // Explicit packing avoids alignment, aliasing and host-byte-order assumptions.
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

    CclAlgoType m_type{CclAlgoType::CCL_DEFAULT};
};
