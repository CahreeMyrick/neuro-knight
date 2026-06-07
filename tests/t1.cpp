#include <iostream>
#include "../src/game/game.hpp"
#include <iostream>
#include "../src/position/bbposition.hpp"
#include "../src/parsing/parse.hpp"
#include "../src/render/render.hpp"
#include "../src/core/movelist.hpp"
#include "../src/movegen/bb_movegen.hpp"
#include "../src/search/search_strategy.hpp"
#include "../src/eval/eval_strategy.hpp"
#include "../src/cli/args.hpp"

void run_selfplay(const GameConfig& config) {
    std::cout << "Running selfplay at depth "
              << config.depth << "\n";
}

int main(int argc, char* argv[]) {
    try {
        CLIOptions opts = parse_args(argc, argv);
        Game game(opts);

        switch (opts.mode) {
            case Mode::Play: {
                game.run();
                break;
            }
            
            case Mode::Selfplay: {
                // game.run_selfplay();
                break;
            }
        }
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}