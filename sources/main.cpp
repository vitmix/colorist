#include <opencv2/imgcodecs.hpp>

#include <iostream>
#include <stdexcept>
#include <string>

int main(int argc, char* argv[]) {
    constexpr auto usage =
        "Usage: colorist --mask <mask.png> --color_mask <color_mask.png> --output <result.png>\n";

    if (argc == 2 && (std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h")) {
        std::cout << usage;
        return 0;
    }

    try {
        std::string mask_path, color_mask_path, output_path;
        for (int i = 1; i < argc; ++i) {
            const std::string option = argv[i];
            std::string* path = option == "--mask" ? &mask_path
                              : option == "--color_mask" ? &color_mask_path
                              : option == "--output" ? &output_path
                              : nullptr;
            if (!path) {
                throw std::runtime_error("Unknown argument: " + option);
            }
            if (i + 1 == argc) {
                throw std::runtime_error("Missing value for " + option);
            }
            if (!path->empty()) {
                throw std::runtime_error("Duplicate argument: " + option);
            }
            *path = argv[++i];
        }
        if (mask_path.empty() || color_mask_path.empty() || output_path.empty()) {
            throw std::runtime_error("--mask, --color_mask and --output are required");
        }

        const cv::Mat mask = cv::imread(mask_path, cv::IMREAD_UNCHANGED);
        if (mask.empty()) {
            throw std::runtime_error("Cannot read mask: " + mask_path);
        }
        const cv::Mat color_mask = cv::imread(color_mask_path, cv::IMREAD_UNCHANGED);
        if (color_mask.empty()) {
            throw std::runtime_error("Cannot read color_mask: " + color_mask_path);
        }

        // Replace this copy with the algorithm using mask and color_mask.
        const cv::Mat result = color_mask.clone();
        if (!cv::imwrite(output_path, result)) {
            throw std::runtime_error("Cannot write output: " + output_path);
        }
        std::cout << "Saved " << output_path << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n' << usage;
        return 1;
    }
}
