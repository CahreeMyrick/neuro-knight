#pragma once

#include <string>

#include "../core/types.hpp"
#include "../eval/eval_strategy.hpp"

enum class Mode {
    Play,
    Selfplay
};

struct CLIOptions {
    Mode mode = Mode::Play;
    int depth = 3;
    Color color = Color::White;
    Color engine_color = Color::Black;
    MaterialEvaluator evaluator;
};

struct GameConfig {
    Color color = Color::White;
    Color engine_color = Color::Black;
    int depth = 3;
    MaterialEvaluator evaluator;
};

CLIOptions parse_args(int argc, char* argv[]);

GameConfig make_game_config(const CLIOptions& opts);