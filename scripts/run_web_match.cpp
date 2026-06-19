#include <iostream>
#include <string>

#include "selfplay/match_web.hpp"

int main(int argc, char* argv[]) {
    std::string white = (argc > 1) ? argv[1] : "neural";
    std::string black = (argc > 2) ? argv[2] : "material";

    int depth = (argc > 3) ? std::stoi(argv[3]) : 3;
    int max_plies = (argc > 4) ? std::stoi(argv[4]) : 120;

    auto result = chess::play_web_match(
        chess::parse_web_engine_kind(white),
        chess::parse_web_engine_kind(black),
        depth,
        max_plies
    );

    std::cout << chess::match_result_to_json(result) << "\n";

    return 0;
}
