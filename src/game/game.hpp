#include <iostream>
#include "../src/position/bbposition.hpp"
#include "../src/parsing/parse.hpp"
#include "../src/render/render.hpp"
#include "../src/core/movelist.hpp"
#include "../src/movegen/bb_movegen.hpp"
#include "../src/search/search_strategy.hpp"
#include "../src/core/movelist.hpp"
#include "../src/cli/args.hpp"


class Game {
public:
    // constructor
    explicit Game(const CLIOptions & opts) : _opts(opts) {}

    // member functions
    void run();
    void run_selfplay();
private:
    CLIOptions _opts;
};


