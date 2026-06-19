#pragma once

#include <string>
#include <vector>

namespace chess {

enum class WebEngineKind {
    Material,
    Neural,
};

struct WebMatchPly {
    int ply = 0;
    std::string side;       // "W" or "B"
    std::string engine;     // "material", "neural", or "lasso"
    std::string move;       // UCI move, example: "e2e4"
    int score = 0;
    std::string fen_after;  // FEN after the move
};

struct WebMatchResult {
    std::string white_engine;
    std::string black_engine;

    int depth = 0;
    int max_plies = 0;

    std::string start_fen;
    std::string result;     // "1-0", "0-1", or "1/2-1/2"

    std::vector<WebMatchPly> plies;
};

WebEngineKind parse_web_engine_kind(const std::string& name);

std::string web_engine_name(WebEngineKind engine);

WebMatchResult play_web_match(
    WebEngineKind white_engine,
    WebEngineKind black_engine,
    int depth,
    int max_plies
);

std::string match_result_to_json(const WebMatchResult& match);

} // namespace chess
