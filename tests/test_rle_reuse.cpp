// Standalone regression check: reuse one RLE object across changing images.
#include "BaseAlgo.h"
#include "RleAlgo.h"

#include <array>
#include <iostream>
#include <random>
#include <stdexcept>

// RLE reads the color mask as BGR. Returns a BGR copy as a non-contiguous view
// (inside a one-pixel border), like the caller-provided views below.
static cv::Mat ToBgr(const cv::Mat& bgra) {
    cv::Mat storage(bgra.rows + 2, bgra.cols + 2, CV_8UC3, cv::Scalar(1, 2, 3));
    cv::Mat view = storage(cv::Rect(1, 1, bgra.cols, bgra.rows));
    cv::cvtColor(bgra, view, cv::COLOR_BGRA2BGR);
    return view;
}

// The reference sees the same colors as RLE: alpha made opaque, so it cannot split a color.
static void ComputeReference(BaseAlgo& reference, const cv::Mat& mask,
                             const cv::Mat& colors, cv::Mat& expected) {
    cv::Mat opaque, bgra(mask.size(), CV_8UC4);
    cv::cvtColor(ToBgr(colors), opaque, cv::COLOR_BGR2BGRA);
    reference.Compute(mask, opaque, bgra);
    cv::cvtColor(bgra, expected, cv::COLOR_BGRA2BGR);
}

static void CheckRle() {
    BaseAlgo reference;
    RleAlgo rle;
    std::mt19937 random(42);
    cv::setNumThreads(1);

    for (int test = 0; test < 100; ++test) {
        const int height = 1 + random() % 25;
        const int width = 1 + random() % 25;
        // Use non-contiguous views as well as changing sizes, palettes and component counts.
        cv::Mat mask_storage(height + 2, width + 2, CV_8UC1);
        cv::Mat color_storage(height + 2, width + 2, CV_8UC4);
        auto mask = mask_storage(cv::Rect(1, 1, width, height));
        auto colors = color_storage(cv::Rect(1, 1, width, height));
        const std::array<cv::Vec4b, 5> palette = {
            cv::Vec4b(0, 0, 0, 0), cv::Vec4b(0, 0, 0, 255),
            cv::Vec4b(0, 0, 255, 255), cv::Vec4b(255, 0, 0, 255),
            cv::Vec4b(random() % 256, random() % 256, random() % 256, random() % 256)
        };
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                static constexpr uchar mask_values[] = {0, 1, 127, 128, 255};
                mask.at<uchar>(y, x) = test % 5 == 0 ? 0 :
                    test % 5 == 1 ? 255 :
                    test % 5 == 2 ? ((x + y) % 2 ? 0 : 255) : mask_values[random() % 5];
                colors.at<cv::Vec4b>(y, x) = palette[random() % palette.size()];
            }
        }
        cv::Mat expected;
        ComputeReference(reference, mask, colors, expected);
        const cv::Vec3b marker(17, 23, 41);
        cv::Mat output_storage(height + 2, width + 2, CV_8UC3);
        auto actual = output_storage(cv::Rect(1, 1, width, height));
        const auto* output_data = actual.data;
        for (int threads : {1, 2, 4, 6, 12}) {
            cv::setNumThreads(threads);
            rle.SetNumThreads(cv::getNumThreads());
            output_storage.setTo(cv::Scalar(marker[0], marker[1], marker[2]));
            rle.Compute(mask, ToBgr(colors), actual);
            if (actual.data != output_data)
                throw std::runtime_error("RLE replaced caller-owned output storage");
            if (actual.size() != expected.size() || actual.type() != expected.type())
                throw std::runtime_error("RLE output shape/type mismatch");
            for (int y = 0; y < height; ++y) {
                for (int x = 0; x < width; ++x) {
                    if (actual.at<cv::Vec3b>(y, x) != expected.at<cv::Vec3b>(y, x))
                        throw std::runtime_error("RLE retained stale counts or color IDs");
                }
            }
            for (int y = 0; y < output_storage.rows; ++y) {
                for (int x = 0; x < output_storage.cols; ++x) {
                    if ((y == 0 || y == height + 1 || x == 0 || x == width + 1) &&
                        output_storage.at<cv::Vec3b>(y, x) != marker)
                        throw std::runtime_error("RLE painted outside the output view");
                }
            }
        }
    }
    std::cout << "PASS: 500 reused-object computations across 100 changing images\n";

    // Reuse run vectors across wide images and empty rows.
    for (int width : {4080, 4095, 4096, 8193}) {
        constexpr int height = 7;
        cv::Mat mask(height, width, CV_8UC1, cv::Scalar(0));
        mask.row(0).setTo(255);
        mask.row(2).colRange(17, width - 6).setTo(255);
        mask.at<uchar>(3, 0) = mask.at<uchar>(3, width - 1) = 255;
        mask.row(4).setTo(255);
        cv::Mat colors(height, width, CV_8UC4, cv::Scalar(0, 0, 0, 0));
        colors.colRange(width / 2, width).setTo(cv::Scalar(7, 8, 9, 255));
        cv::Mat expected;
        ComputeReference(reference, mask, colors, expected);
        const cv::Vec3b marker(17, 23, 41);
        cv::Mat storage(height + 2, width + 2, CV_8UC3);
        auto actual = storage(cv::Rect(1, 1, width, height));
        for (int threads : {1, 2, 4, 12}) {
            cv::setNumThreads(threads);
            rle.SetNumThreads(threads);
            storage.setTo(cv::Scalar(marker[0], marker[1], marker[2]));
            rle.Compute(mask, ToBgr(colors), actual);
            if (cv::norm(actual, expected, cv::NORM_INF) != 0)
                throw std::runtime_error("RLE wide-run or empty-row coloring mismatch");
            for (int y = 0; y < storage.rows; ++y)
                for (int x = 0; x < storage.cols; ++x)
                    if ((y == 0 || y == height + 1 || x == 0 || x == width + 1) &&
                        storage.at<cv::Vec3b>(y, x) != marker)
                        throw std::runtime_error("RLE painted outside the wide output view");
        }
    }
    std::cout << "PASS: 16 wide-image computations with mixed runs and empty rows\n";

    // Background painting must cover leading/trailing and entirely empty rows,
    // including SIMD tails. BGR output drops alpha after selecting the BGRA winner.
    for (int width : {1, 3, 4, 7, 8, 9, 15, 16, 17, 31, 32, 33, 127, 128, 4095, 4096}) {
        constexpr int height = 5;
        cv::Mat mask(height, width, CV_8UC1, cv::Scalar(127));
        mask.at<uchar>(1, 0) = 128;
        mask.row(2).setTo(255);
        mask.at<uchar>(3, width - 1) = 255;
        cv::Mat colors(height, width, CV_8UC4, cv::Scalar(0, 0, 0, 0));
        cv::Mat expected;
        ComputeReference(reference, mask, colors, expected);
        cv::Mat storage(height + 2, width + 2, CV_8UC3);
        auto actual = storage(cv::Rect(1, 1, width, height));
        const cv::Vec3b marker(17, 23, 41);
        for (int threads : {1, 2, 4, 12}) {
            cv::setNumThreads(threads);
            rle.SetNumThreads(threads);
            storage.setTo(cv::Scalar(marker[0], marker[1], marker[2]));
            rle.Compute(mask, ToBgr(colors), actual);
            if (cv::norm(actual, expected, cv::NORM_INF) != 0)
                throw std::runtime_error("RLE background-run or transparent-black mismatch");
            for (int y = 0; y < storage.rows; ++y)
                for (int x = 0; x < storage.cols; ++x)
                    if ((y == 0 || y == height + 1 || x == 0 || x == width + 1) &&
                        storage.at<cv::Vec3b>(y, x) != marker)
                        throw std::runtime_error("RLE background fill escaped the output view");
        }
    }
    std::cout << "PASS: 64 background-run computations with SIMD tails and output guards\n";

    // More than 4096 isolated singletons must retain their independent components.
    // Use a fresh palette so this stress case needs only one histogram bin per node.
    RleAlgo overflow_rle;
    overflow_rle.SetNumThreads(cv::getNumThreads());
    const int height = 129, width = 129;
    cv::Mat mask(height, width, CV_8UC1);
    const cv::Vec3b foreground(7, 8, 9), background(0, 0, 0);
    cv::Mat colors(height, width, CV_8UC4, cv::Scalar(7, 8, 9, 0));
    cv::Mat actual(height, width, CV_8UC3);
    for (int y = 0; y < height; ++y) {
        auto* row = mask.ptr<uchar>(y);
        for (int x = 0; x < width; ++x)
            row[x] = (x + y) % 2 == 0 ? 255 : 0;
    }
    for (int repetition = 0; repetition < 2; ++repetition) {
        cv::setNumThreads(repetition == 0 ? 1 : 6);
        overflow_rle.SetNumThreads(cv::getNumThreads());
        actual.setTo(cv::Scalar(255, 255, 255, 255));
        overflow_rle.Compute(mask, ToBgr(colors), actual);
        for (int y = 0; y < height; ++y) {
            const auto* row = actual.ptr<cv::Vec3b>(y);
            for (int x = 0; x < width; ++x) {
                if (row[x] != ((x + y) % 2 == 0 ? foreground : background))
                    throw std::runtime_error("RLE extended component-ID coloring mismatch");
            }
        }
    }
    std::cout << "PASS: 2 computations above 4096 components with sequential/striped reuse\n";

    // Two-pixel runs next to one-pixel gaps/singletons, including a run occupying
    // the final two pixels of a non-contiguous row.
    for (int width : {2, 3, 4, 5, 7, 8, 17, 33, 513}) {
        constexpr int height = 129;
        cv::Mat mask(height, width, CV_8UC1, cv::Scalar(0));
        for (int y = 0; y < height; y += 2) {
            auto* row = mask.ptr<uchar>(y);
            for (int x = 0; x < width; ++x)
                row[x] = x % 4 < 2 ? 255 : 0;
        }
        mask.row(1).colRange(width - 2, width).setTo(255);
        mask.at<uchar>(3, width - 1) = 255;
        cv::Mat colors(height, width, CV_8UC4, cv::Scalar(17, 23, 41, 255));
        colors.colRange(width / 2, width).setTo(cv::Scalar(7, 8, 9, 0));
        cv::Mat expected;
        ComputeReference(reference, mask, colors, expected);
        cv::Mat storage(height + 2, width + 2, CV_8UC3);
        auto actual = storage(cv::Rect(1, 1, width, height));
        const cv::Vec3b marker(101, 103, 107);
        for (int threads : {1, 2, 4, 12}) {
            cv::setNumThreads(threads);
            rle.SetNumThreads(threads);
            storage.setTo(cv::Scalar(marker[0], marker[1], marker[2]));
            rle.Compute(mask, ToBgr(colors), actual);
            if (cv::norm(actual, expected, cv::NORM_INF) != 0)
                throw std::runtime_error("RLE multipixel header or large component-ID mismatch");
            for (int y = 0; y < storage.rows; ++y)
                for (int x = 0; x < storage.cols; ++x)
                    if ((y == 0 || y == height + 1 || x == 0 || x == width + 1) &&
                        storage.at<cv::Vec3b>(y, x) != marker)
                        throw std::runtime_error("RLE multipixel header escaped its row");
        }
    }
    std::cout << "PASS: 36 short-run computations with row guards and large IDs\n";

    // Long runs are represented directly, without the former 22-bit encoding limit.
    for (int width : {(1 << 22) - 1, (1 << 22) + 1}) {
        cv::Mat mask(1, width, CV_8UC1, cv::Scalar(255));
        cv::Mat colors(1, width, CV_8UC4, cv::Scalar(17, 23, 41, 255));
        cv::Mat storage(1, width + 2, CV_8UC3, cv::Scalar(101, 103, 107));
        auto actual = storage.colRange(1, width + 1);
        cv::setNumThreads(1);
        rle.SetNumThreads(1);
        rle.Compute(mask, ToBgr(colors), actual);
        for (int x = 0; x < width; ++x)
            if (actual.at<cv::Vec3b>(0, x) != cv::Vec3b(17, 23, 41))
                throw std::runtime_error("RLE 22-bit length limit or fallback mismatch");
        if (storage.at<cv::Vec3b>(0, 0) != cv::Vec3b(101, 103, 107) ||
            storage.at<cv::Vec3b>(0, width + 1) != cv::Vec3b(101, 103, 107))
            throw std::runtime_error("RLE long run escaped its buffer");
    }
    std::cout << "PASS: 2 computations at/beyond the 22-bit run-length limit\n";

    // Mostly-correct SIMD batches, minority colors inside a batch, and black runs.
    // The input/output views have guards and different non-contiguous row strides.
    for (int width : {15, 16, 17, 31, 32, 33, 63, 64, 65, 129}) {
        constexpr int height = 7;
        cv::Mat mask(height, width, CV_8UC1, cv::Scalar(255));
        mask.row(2).setTo(0);
        mask.col(width - 1).setTo(0);
        cv::Mat source(height + 2, width + 4, CV_8UC4, cv::Scalar(17, 23, 41, 0));
        auto colors = source(cv::Rect(1, 1, width, height));
        colors.at<cv::Vec4b>(1, width / 2) = cv::Vec4b(7, 8, 9, 255);
        colors.row(4).setTo(cv::Scalar(0, 0, 0, 255));
        const cv::Mat input = ToBgr(colors);
        const auto original = input.clone();
        cv::Mat expected;
        ComputeReference(reference, mask, colors, expected);
        const cv::Vec3b guard(101, 103, 107);
        cv::Mat storage(height + 2, width + 2, CV_8UC3, cv::Scalar(101, 103, 107));
        auto actual = storage(cv::Rect(1, 1, width, height));
        const auto* data = actual.data;
        for (int threads : {1, 2, 4}) {
            cv::setNumThreads(threads);
            rle.SetNumThreads(threads);
            // Repeated calls share the same output without resetting it in the caller.
            for (int repeat = 0; repeat < 3; ++repeat) {
                rle.Compute(mask, input, actual);
                if (cv::norm(actual, expected, cv::NORM_INF) != 0 || actual.data != data)
                    throw std::runtime_error("RLE repeated SIMD recoloring mismatch");
                if (cv::norm(input, original, cv::NORM_INF) != 0)
                    throw std::runtime_error("RLE modified the input color image");
                for (int y = 0; y < storage.rows; ++y)
                    for (int x = 0; x < storage.cols; ++x)
                        if ((y == 0 || y == height + 1 || x == 0 || x == width + 1) &&
                            storage.at<cv::Vec3b>(y, x) != guard)
                            throw std::runtime_error("RLE recoloring escaped its output view");
            }
        }
    }
    std::cout << "PASS: 90 repeated SIMD computations with unchanged input and output guards\n";

    // Input colors change across rows. The otherwise poisoned output has one
    // correct pixel already: completion must still paint the whole polygon.
    for (int threads : {1, 2, 4}) {
        constexpr int width = 97, height = 4;
        cv::Mat mask(height, width, CV_8UC1, cv::Scalar(255));
        cv::Mat colors(height, width, CV_8UC4, cv::Scalar(0, 255, 0, 255));
        colors.row(0).setTo(cv::Scalar(0, 0, 255, 255));
        for (int x = 0; x < width; x += 2)
            colors.at<cv::Vec4b>(1, x) = cv::Vec4b(0, 0, 255, 255);
        cv::Mat expected;
        ComputeReference(reference, mask, colors, expected);
        cv::Mat actual(height, width, CV_8UC3, cv::Scalar(101, 103, 107));
        actual.at<cv::Vec3b>(1, 0) = cv::Vec3b(0, 255, 0);
        cv::setNumThreads(threads);
        rle.SetNumThreads(threads);
        rle.Compute(mask, ToBgr(colors), actual);
        if (cv::norm(actual, expected, cv::NORM_INF) != 0)
            throw std::runtime_error("RLE failed to paint the complete polygon");
    }
    std::cout << "PASS: 3 mixed-color polygons with poisoned output\n";
}

static void CheckFillBoundaries() {
    RleAlgo algo(2);
    cv::setNumThreads(2);
    const cv::Vec3b guard(101, 103, 107), color(17, 23, 41);
    // Exercise every remainder/alignment around multiple SSE/NEON/AVX widths.
    // Empty separating rows keep neighboring runs as independent components.
    for (int length = 1; length <= 129; ++length) {
        const int width = length + 15, height = 32;
        cv::Mat mask(height, width, CV_8UC1, cv::Scalar(0));
        for (int offset = 0; offset < 16; ++offset)
            mask.row(2 * offset).colRange(offset, offset + length).setTo(255);
        cv::Mat colors(height, width, CV_8UC4, cv::Scalar(color[0], color[1], color[2], 255));
        cv::Mat storage(height + 2, width + 2, CV_8UC3,
                        cv::Scalar(guard[0], guard[1], guard[2]));
        auto output = storage(cv::Rect(1, 1, width, height));
        algo.Compute(mask, ToBgr(colors), output);
        for (int y = 0; y < storage.rows; ++y)
            for (int x = 0; x < storage.cols; ++x) {
                const bool outside = y == 0 || y == height + 1 || x == 0 || x == width + 1;
                const auto expected = outside ? guard :
                    mask.at<uchar>(y - 1, x - 1) > 127 ? color : cv::Vec3b(0, 0, 0);
                if (storage.at<cv::Vec3b>(y, x) != expected)
                    throw std::runtime_error("BGR fill crossed a run/row boundary or left pixels unwritten");
            }
    }
    std::cout << "PASS: 129 lengths x 16 offsets with poisoned output and row guards\n";
}

int main() {
    CheckFillBoundaries();
    CheckRle();
}
