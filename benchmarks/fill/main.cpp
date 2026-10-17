// Memory roofline for colorist: the unavoidable traffic of one frame, without labeling.
//   read     - read every mask byte and BGR color byte (cheap SIMD sum keeps the loads)
//   write    - zero each output row, then fill its foreground runs with one color
//   roofline - both, row by row: the same traffic as `rle`, i.e. the floor it can reach
// Rows are split into stripes, one per thread, as in RleAlgo. The *_cols variants split
// columns instead: every thread walks all rows over its own column band, so the threads
// stream through neighboring addresses at the same time rather than far-apart stripes.
#include <opencv2/core.hpp>
#include <opencv2/core/simd_intrinsics.hpp>
#include <opencv2/imgcodecs.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

struct Span { int begin, end; };
enum Kernel { Read = 1, Write = 2, Roofline = Read | Write, Columns = 4 };
static std::atomic<std::uint64_t> observed{0}; // Keeps the read checksums alive.
static constexpr uchar fill_byte = 0x55;
static const cv::Vec3b fill_color(fill_byte, fill_byte, fill_byte);

static std::uint64_t ReadRow(const uchar* mask, const uchar* bgr, int width) {
    const int lanes = cv::VTraits<cv::v_uint8>::vlanes();
    auto sum = cv::vx_setzero_u8();
    int x = 0;
    for (; x + lanes <= width; x += lanes)
        sum = cv::v_add_wrap(sum, cv::vx_load(mask + x));
    const int bytes = 3 * width;
    int b = 0;
    for (; b + lanes <= bytes; b += lanes)
        sum = cv::v_add_wrap(sum, cv::vx_load(bgr + b));
    cv::v_uint16 low, high;
    cv::v_expand(sum, low, high);
    std::uint64_t tail = cv::v_reduce_sum(low) + cv::v_reduce_sum(high);
    for (; x < width; ++x) tail += mask[x];
    for (; b < bytes; ++b) tail += bgr[b];
    return tail;
}

// Same bytes as RleAlgo writes (zero the row, then overwrite every foreground run), with
// the cheapest possible fill: a gray run is a plain memset, no color pattern to build.
// Writes pixels [x0, x1) of one row; runs are clipped to that range.
static void WriteRow(uchar* out, int x0, int x1, const std::vector<Span>& runs) {
    std::memset(out + 3 * x0, 0, std::size_t(x1 - x0) * 3);
    for (const auto& run : runs) {
        const int begin = std::max(run.begin, x0), end = std::min(run.end, x1);
        if (begin < end)
            std::memset(out + 3 * begin, fill_byte, std::size_t(end - begin) * 3);
    }
}

// Rows [y0, y1) x columns [x0, x1).
static std::uint64_t RunBlock(int kernel, int y0, int y1, int x0, int x1, const cv::Mat& mask,
                              const cv::Mat& bgr, cv::Mat& output, const std::vector<std::vector<Span>>& runs) {
    std::uint64_t checksum = 0;
    for (int y = y0; y < y1; ++y) {
        if (kernel & Read)
            checksum += ReadRow(mask.ptr<uchar>(y) + x0, bgr.ptr<uchar>(y) + 3 * x0, x1 - x0);
        if (kernel & Write)
            WriteRow(output.ptr<uchar>(y), x0, x1, runs[y]);
    }
    return checksum;
}

static void RunFrame(int kernel, int threads, const cv::Mat& mask, const cv::Mat& bgr, cv::Mat& output,
                     const std::vector<std::vector<Span>>& runs) {
    cv::parallel_for_(cv::Range(0, threads), [&](const cv::Range& range) {
        std::uint64_t checksum = 0;
        for (int part = range.start; part < range.end; ++part) {
            if (kernel & Columns) {
                // Band edges on 64-pixel multiples: 64 BGR pixels are three whole cache lines,
                // so neighboring bands never share an input or output line.
                const int blocks = (mask.cols + 63) / 64;
                const int x0 = std::min(mask.cols, blocks * part / threads * 64);
                const int x1 = std::min(mask.cols, blocks * (part + 1) / threads * 64);
                checksum += RunBlock(kernel, 0, mask.rows, x0, x1, mask, bgr, output, runs);
            } else {
                const int y0 = static_cast<int>(std::int64_t(mask.rows) * part / threads);
                const int y1 = static_cast<int>(std::int64_t(mask.rows) * (part + 1) / threads);
                checksum += RunBlock(kernel, y0, y1, 0, mask.cols, mask, bgr, output, runs);
            }
        }
        observed.fetch_add(checksum, std::memory_order_relaxed);
    }, threads);
}

static std::vector<int> ParseList(const std::string& text) {
    std::vector<int> values;
    std::stringstream stream(text);
    for (std::string item; std::getline(stream, item, ',');)
        values.push_back(std::stoi(item));
    return values;
}

int main(int argc, char** argv) {
    try {
        if (argc < 4 || argc > 7)
            throw std::runtime_error("Usage: fill_bench <mask.png> <color_mask.png> <output.csv> "
                                     "[runs=50] [kernels=read,write,roofline,roofline_cols] [threads=1,2,3,4]\n"
                                     "kernels: read, write, roofline and their *_cols variants");
        const int runs_count = argc > 4 ? std::stoi(argv[4]) : 50;
        const std::string kernel_arg = argc > 5 ? argv[5] : "read,write,roofline,roofline_cols";
        const auto thread_counts = ParseList(argc > 6 ? argv[6] : "1,2,3,4");
        if (runs_count < 1) throw std::runtime_error("runs must be >= 1");

        std::vector<std::pair<std::string, int>> kernels;
        std::stringstream names(kernel_arg);
        for (std::string name; std::getline(names, name, ',');) {
            const bool columns = name.size() > 5 && name.compare(name.size() - 5, 5, "_cols") == 0;
            const std::string base = columns ? name.substr(0, name.size() - 5) : name;
            int kernel = base == "read" ? Read : base == "write" ? Write : base == "roofline" ? Roofline : 0;
            if (!kernel) throw std::runtime_error("Unknown kernel: " + name);
            if (columns) kernel |= Columns;
            kernels.emplace_back(name, kernel);
        }

        // Decoded exactly as colorist does for rle; outside all timing.
        const cv::Mat mask = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);
        const cv::Mat bgr = cv::imread(argv[2], cv::IMREAD_COLOR);
        if (mask.empty() || bgr.empty() || mask.size() != bgr.size())
            throw std::runtime_error("Expected equal-size mask and color images");

        std::vector<std::vector<Span>> runs(mask.rows);
        for (int y = 0; y < mask.rows; ++y) {
            const auto* row = mask.ptr<uchar>(y);
            for (int x = 0; x < mask.cols;) {
                if (row[x] <= 127) { ++x; continue; }
                const int start = x;
                while (x < mask.cols && row[x] > 127) ++x;
                runs[y].push_back({start, x});
            }
        }
        cv::Mat output(mask.size(), CV_8UC3);
        const double read_mb = (mask.total() + bgr.total() * 3) / 1e6;
        const double write_mb = output.total() * 3 / 1e6;
        std::cout << "Image " << mask.cols << 'x' << mask.rows << ", SIMD " << CV_SIMD_WIDTH
                  << " bytes; per frame: read " << read_mb << " MB, write " << write_mb << " MB\n\n";

        std::ofstream csv(argv[3]);
        if (!csv) throw std::runtime_error("Cannot open output CSV");
        csv << "kernel,threads,runs,median_ms,p95_ms,min_ms,mb_per_frame,gb_per_s\n" << std::fixed << std::setprecision(4);
        std::cout << std::left << std::setw(10) << "kernel" << std::right << std::setw(8) << "threads"
                  << std::setw(11) << "median ms" << std::setw(9) << "p95 ms" << std::setw(9) << "min ms"
                  << std::setw(10) << "MB/frame" << std::setw(8) << "GB/s" << '\n' << std::fixed << std::setprecision(3);

        for (const auto& [name, kernel] : kernels) {
            for (const int threads : thread_counts) {
                cv::setNumThreads(threads);
                std::vector<double> times;
                for (int iteration = -5; iteration < runs_count; ++iteration) {
                    const auto start = std::chrono::steady_clock::now();
                    RunFrame(kernel, threads, mask, bgr, output, runs);
                    const double ms = std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - start).count();
                    if (iteration >= 0) times.push_back(ms);
                }
                if (kernel & Write) // Every pixel: fill color inside runs, black elsewhere.
                    for (int y = 0; y < mask.rows; ++y)
                        for (int x = 0; x < mask.cols; ++x)
                            if (output.at<cv::Vec3b>(y, x) != (mask.at<uchar>(y, x) > 127 ? fill_color : cv::Vec3b()))
                                throw std::runtime_error("Output verification failed");
                std::sort(times.begin(), times.end());
                const double median = times[times.size() / 2];
                const double p95 = times[(times.size() * 95 + 99) / 100 - 1];
                const double mb = (kernel & Read ? read_mb : 0) + (kernel & Write ? write_mb : 0);
                const double gbps = mb / median; // MB per ms == GB per s
                std::cout << std::left << std::setw(10) << name << std::right << std::setw(8) << threads
                          << std::setw(11) << median << std::setw(9) << p95 << std::setw(9) << times.front()
                          << std::setw(10) << mb << std::setw(8) << gbps << '\n';
                csv << name << ',' << threads << ',' << runs_count << ',' << median << ',' << p95 << ','
                    << times.front() << ',' << mb << ',' << gbps << '\n';
            }
        }
        std::cout << "\nOutput verified for write/roofline. CSV: " << argv[3] << '\n';
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
