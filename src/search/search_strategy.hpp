#pragma once

#include "core/move.hpp"
#include "core/movelist.hpp"
#include "position/bbposition.hpp"
#include "movegen/bb_movegen.hpp"
#include "eval/eval_strategy.hpp"

struct SearchResult {
    Move best = {};
    int score = 0;
};

class Search {
public:
    explicit Search(const IEvaluator& evaluator);

    SearchResult minimax(BitboardPosition& pos, int depth);
    void clear_tt();

private:
    const IEvaluator& evaluator_;

    int alphabeta(
        BitboardPosition& pos,
        int depth,
        int alpha,
        int beta,
        int ply
    );
};