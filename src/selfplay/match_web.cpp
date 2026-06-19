#include "selfplay/match_web.hpp"

#include "position/bbposition.hpp"
#include "search/search_strategy.hpp"
#include "core/uci.hpp"
#include "openings/opening_book.h"

#include "eval/eval_strategy.hpp"
#include "eval/eval_nn.hpp"
// #include "eval/eval_lasso.hpp"

#include <sstream>
#include <stdexcept>
#include <string>

#include <iostream>

namespace chess {

WebEngineKind parse_web_engine_kind(const std::string& name) {
    if (name == "material") return WebEngineKind::Material;
    if (name == "neural")  return WebEngineKind::Neural;
    // if (name == "lasso")   return WebEngineKind::Lasso;

    throw std::runtime_error("Unknown engine kind: " + name);
}

std::string web_engine_name(WebEngineKind engine) {
    switch (engine) {
        case WebEngineKind::Material:
            return "material";

        case WebEngineKind::Neural:
            return "neural";

        // case WebEngineKind::Lasso:
        //     return "lasso";
    }

    return "unknown";
}

static SearchResult choose_web_move(
    BitboardPosition& pos,
    WebEngineKind engine,
    int depth,
    Search& material_search,
    Search& neural_search
) {
    switch (engine) {
        case WebEngineKind::Neural:
            return neural_search.minimax(pos, depth);

        case WebEngineKind::Material:
        default:
            return material_search.minimax(pos, depth);
    }
}

static BitboardPosition starting_position_from_book() {
    const std::string book_path =
        "/Users/cahree/dev/systems/chess/engines/engine_4/src/openings/popularpos_lichess_v3.epd";

    // Suppress opening book logs so stdout remains pure JSON.
    std::ostringstream suppressed_output;
    std::streambuf* old_cout_buffer = std::cout.rdbuf(suppressed_output.rdbuf());

    PositionBook book = loadEPD(book_path);

    std::cout.rdbuf(old_cout_buffer);

    if (!book.empty()) {
        std::string fen = pickRandomPosition(book);

        if (!fen.empty()) {
            return BitboardPosition::from_fen(fen);
        }
    }

    return BitboardPosition::startpos();
}

static std::string result_string(float result) {
    if (result == 1.0f) return "1-0";
    if (result == 0.0f) return "0-1";
    return "1/2-1/2";
}

static std::string json_escape(const std::string& s) {
    std::ostringstream out;

    for (char c : s) {
        switch (c) {
            case '"':  out << "\\\""; break;
            case '\\': out << "\\\\"; break;
            case '\n': out << "\\n";  break;
            case '\r': out << "\\r";  break;
            case '\t': out << "\\t";  break;
            default:   out << c;      break;
        }
    }

    return out.str();
}

WebMatchResult play_web_match(
    WebEngineKind white_engine,
    WebEngineKind black_engine,
    int depth,
    int max_plies
) {
    WebMatchResult match;

    match.white_engine = web_engine_name(white_engine);
    match.black_engine = web_engine_name(black_engine);
    match.depth = depth;
    match.max_plies = max_plies;

    BitboardPosition pos = starting_position_from_book();

    MaterialEvaluator material_eval;
    NeuralEvaluator neural_eval;
    // LassoEvaluator lasso_eval;

    Search material_search(material_eval);
    Search neural_search(neural_eval);
    // Search lasso_search(lasso_eval);

    material_search.clear_tt();
    neural_search.clear_tt();
    // lasso_search.clear_tt();

    match.start_fen = pos.to_fen();

    float final_result = 0.5f;

    for (int ply = 0; ply < max_plies; ++ply) {
        if (pos.is_draw_by_fifty() || pos.is_draw_by_repetition()) {
            final_result = 0.5f;
            break;
        }

        MoveList legal;
        BitboardMoveGen::generate_legal(pos, legal);

        if (legal.size == 0) {
            if (BitboardMoveGen::in_check(pos, pos.side_to_move())) {
                final_result =
                    (pos.side_to_move() == Color::White)
                        ? 0.0f
                        : 1.0f;
            } else {
                final_result = 0.5f;
            }

            break;
        }

        bool white_to_move = pos.side_to_move() == Color::White;

        WebEngineKind engine_to_move =
            white_to_move ? white_engine : black_engine;

        SearchResult result = choose_web_move(
            pos,
            engine_to_move,
            depth,
            material_search,
            neural_search
        );

        WebMatchPly row;
        row.ply = ply;
        row.side = white_to_move ? "W" : "B";
        row.engine = web_engine_name(engine_to_move);
        row.move = move_to_uci(result.best);
        row.score = result.score;

        pos.make_move(result.best);

        row.fen_after = pos.to_fen();

        match.plies.push_back(row);
    }

    match.result = result_string(final_result);

    return match;
}

std::string match_result_to_json(const WebMatchResult& match) {
    std::ostringstream out;

    out << "{";
    out << "\"white_engine\":\"" << json_escape(match.white_engine) << "\",";
    out << "\"black_engine\":\"" << json_escape(match.black_engine) << "\",";
    out << "\"depth\":" << match.depth << ",";
    out << "\"max_plies\":" << match.max_plies << ",";
    out << "\"result\":\"" << json_escape(match.result) << "\",";
    out << "\"start_fen\":\"" << json_escape(match.start_fen) << "\",";
    out << "\"plies\":[";

    for (size_t i = 0; i < match.plies.size(); ++i) {
        const WebMatchPly& p = match.plies[i];

        if (i > 0) {
            out << ",";
        }

        out << "{";
        out << "\"ply\":" << p.ply << ",";
        out << "\"side\":\"" << json_escape(p.side) << "\",";
        out << "\"engine\":\"" << json_escape(p.engine) << "\",";
        out << "\"move\":\"" << json_escape(p.move) << "\",";
        out << "\"score\":" << p.score << ",";
        out << "\"fen_after\":\"" << json_escape(p.fen_after) << "\"";
        out << "}";
    }

    out << "]";
    out << "}";

    return out.str();
}

} // namespace chess
