#include "cli/args.hpp"
#include <stdexcept>
#include <string>

CLIOptions parse_args(int argc, char* argv[]) {
    CLIOptions opts;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "--play") {
            opts.mode = Mode::Play;
        }
        else if (arg == "--selfplay") {
            opts.mode = Mode::Selfplay;
        }
        else if (arg == "--depth") {
            if (i + 1 >= argc) {
                throw std::runtime_error("--depth requires a value");
            }

            opts.depth = std::stoi(argv[++i]);
        }
        else if (arg == "--color") {
            if (i + 1 >= argc) {
                throw std::runtime_error("--color requires a value");
            }

            std::string color = argv[++i];

            if (color == "white") {
                opts.color = Color::White;
                opts.engine_color = Color::Black;
            }
            else if (color == "black") {
                opts.color = Color::Black;
                opts.engine_color = Color::White;
            }
            else {
                throw std::runtime_error(
                    "Invalid color: " + color +
                    " (expected 'white' or 'black')"
                );
            }
        }
        else {
            throw std::runtime_error("Unknown argument: " + arg);
        }
    }

    return opts;
}


GameConfig make_game_config(const CLIOptions& opts) {
    return {
        .color = opts.color,
        .engine_color = opts.engine_color,
        .depth = opts.depth,
        .evaluator = opts.evaluator
    };
}