#include "eval/eval_strategy.hpp"
#include "eval/eval_nn.hpp"

#include <bit>

int MaterialEvaluator::evaluate(const BitboardPosition& pos) const {
    int score = 0;

    score += 900 * std::popcount(pos.pieces(Color::White, 1)); // Queen
    score += 300 * std::popcount(pos.pieces(Color::White, 2)); // Knight
    score += 300 * std::popcount(pos.pieces(Color::White, 3)); // Bishop
    score += 100 * std::popcount(pos.pieces(Color::White, 4)); // Pawn
    score += 500 * std::popcount(pos.pieces(Color::White, 5)); // Rook

    score -= 900 * std::popcount(pos.pieces(Color::Black, 1));
    score -= 300 * std::popcount(pos.pieces(Color::Black, 2));
    score -= 300 * std::popcount(pos.pieces(Color::Black, 3));
    score -= 100 * std::popcount(pos.pieces(Color::Black, 4));
    score -= 500 * std::popcount(pos.pieces(Color::Black, 5));

    return score;
}

int NeuralEvaluator::evaluate(const BitboardPosition& pos) const {
    return NNEval::evaluate(pos);
}

int LassoEvaluator::evaluate(const BitboardPosition&) const {
    return 0;
}