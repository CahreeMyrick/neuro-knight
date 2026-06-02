#include "search/search_strategy.hpp"
#include "eval/eval_strategy.hpp"
// #include "eval/eval_nn.hpp"
// #include "eval/eval_lasso.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>

static constexpr int CONTEMPT = 25;

enum class TTFlag : uint8_t {
    EXACT,
    LOWERBOUND,
    UPPERBOUND
};

struct TTEntry {
    uint64_t key = 0;
    int score = 0;
    int depth = -1;
    TTFlag flag = TTFlag::EXACT;
    Move best = {};
};

static constexpr size_t TT_SIZE = 1 << 22;
static std::array<TTEntry, TT_SIZE> tt;

static void tt_clear() {
    tt.fill(TTEntry{});
}

static TTEntry* tt_probe(uint64_t key) {
    TTEntry& entry = tt[key & (TT_SIZE - 1)];
    return (entry.key == key) ? &entry : nullptr;
}

static void tt_store(
    uint64_t key,
    int score,
    int depth,
    TTFlag flag,
    Move best
) {
    TTEntry& entry = tt[key & (TT_SIZE - 1)];

    if (entry.key == 0 || depth >= entry.depth) {
        entry = {key, score, depth, flag, best};
    }
}

static int piece_val(Piece p) {
    switch (p) {
        case Piece::WQ:
        case Piece::BQ:
            return 900;

        case Piece::WR:
        case Piece::BR:
            return 500;

        case Piece::WB:
        case Piece::BB:
            return 300;

        case Piece::WN:
        case Piece::BN:
            return 300;

        case Piece::WP:
        case Piece::BP:
            return 100;

        default:
            return 0;
    }
}

static bool same_move(const Move& a, const Move& b) {
    return a.from == b.from &&
           a.to == b.to &&
           a.promo == b.promo;
}

static void order_moves(
    BitboardPosition& pos,
    MoveList& moves,
    Move tt_best = {}
) {
    auto move_score = [&](const Move& m) -> int {
        if (same_move(m, tt_best)) {
            return 100000;
        }

        int score = 0;

        Piece captured = pos.piece_at(m.to);
        Piece moving = pos.piece_at(m.from);

        if (captured != Piece::Empty) {
            score += 10000 + piece_val(captured) - piece_val(moving);
        }

        if (m.promo != PROMO_NONE) {
            score += 8000;
        }

        return score;
    };

    std::sort(
        moves.moves.begin(),
        moves.moves.begin() + moves.size,
        [&](const Move& a, const Move& b) {
            return move_score(a) > move_score(b);
        }
    );
}

Search::Search(const IEvaluator& evaluator)
    : evaluator_(evaluator) {}

void Search::clear_tt() {
    tt_clear();
}

SearchResult Search::minimax(BitboardPosition& pos, int depth) {
    MoveList moves;
    BitboardMoveGen::generate_legal(pos, moves);

    SearchResult res;

    if (moves.size == 0) {
        res.best = {};
        res.score = 0;
        return res;
    }

    Move tt_best = {};
    TTEntry* entry = tt_probe(pos.zobrist_key());

    if (entry) {
        tt_best = entry->best;
    }

    order_moves(pos, moves, tt_best);

    Color root_stm = pos.side_to_move();

    res.score = (root_stm == Color::White)
        ? std::numeric_limits<int>::min()
        : std::numeric_limits<int>::max();

    res.best = moves.moves[0];

    int alpha = std::numeric_limits<int>::min();
    int beta = std::numeric_limits<int>::max();

    for (int i = 0; i < moves.size; ++i) {
        pos.make_move(moves.moves[i]);

        int score = alphabeta(
            pos,
            depth - 1,
            alpha,
            beta,
            1
        );

        pos.undo_move();

        if (root_stm == Color::White) {
            if (score > res.score) {
                res.score = score;
                res.best = moves.moves[i];
            }

            alpha = std::max(alpha, score);
        } else {
            if (score < res.score) {
                res.score = score;
                res.best = moves.moves[i];
            }

            beta = std::min(beta, score);
        }

        if (beta <= alpha) {
            break;
        }
    }

    return res;
}

int Search::alphabeta(
    BitboardPosition& pos,
    int depth,
    int alpha,
    int beta,
    int ply
) {
    if (pos.is_draw_by_repetition()) {
        return (pos.side_to_move() == Color::White)
            ? -CONTEMPT
            : CONTEMPT;
    }

    if (pos.is_draw_by_fifty()) {
        return 0;
    }

    uint64_t key = pos.zobrist_key();

    int orig_alpha = alpha;
    int orig_beta = beta;

    Move tt_best = {};

    TTEntry* entry = tt_probe(key);

    if (entry && entry->depth >= depth) {
        switch (entry->flag) {
            case TTFlag::EXACT:
                return entry->score;

            case TTFlag::LOWERBOUND:
                alpha = std::max(alpha, entry->score);
                break;

            case TTFlag::UPPERBOUND:
                beta = std::min(beta, entry->score);
                break;
        }

        if (alpha >= beta) {
            return entry->score;
        }

        tt_best = entry->best;
    } else if (entry) {
        tt_best = entry->best;
    }

    if (depth == 0) {
        return evaluator_.evaluate(pos);
    }

    MoveList moves;
    BitboardMoveGen::generate_legal(pos, moves);
    order_moves(pos, moves, tt_best);

    if (moves.size == 0) {
        if (BitboardMoveGen::in_check(pos, pos.side_to_move())) {
            return (pos.side_to_move() == Color::White)
                ? -100000 + ply
                : 100000 - ply;
        }

        return 0;
    }

    Move best_move = moves.moves[0];

    if (pos.side_to_move() == Color::White) {
        int best = std::numeric_limits<int>::min();

        for (int i = 0; i < moves.size; ++i) {
            pos.make_move(moves.moves[i]);

            int score = alphabeta(
                pos,
                depth - 1,
                alpha,
                beta,
                ply + 1
            );

            pos.undo_move();

            if (score > best) {
                best = score;
                best_move = moves.moves[i];
            }

            alpha = std::max(alpha, score);

            if (beta <= alpha) {
                break;
            }
        }

        TTFlag flag =
            (best <= orig_alpha) ? TTFlag::UPPERBOUND :
            (best >= orig_beta)  ? TTFlag::LOWERBOUND :
                                   TTFlag::EXACT;

        tt_store(key, best, depth, flag, best_move);

        return best;
    }

    int best = std::numeric_limits<int>::max();

    for (int i = 0; i < moves.size; ++i) {
        pos.make_move(moves.moves[i]);

        int score = alphabeta(
            pos,
            depth - 1,
            alpha,
            beta,
            ply + 1
        );

        pos.undo_move();

        if (score < best) {
            best = score;
            best_move = moves.moves[i];
        }

        beta = std::min(beta, score);

        if (beta <= alpha) {
            break;
        }
    }

    TTFlag flag =
        (best >= orig_beta)  ? TTFlag::LOWERBOUND :
        (best <= orig_alpha) ? TTFlag::UPPERBOUND :
                               TTFlag::EXACT;

    tt_store(key, best, depth, flag, best_move);

    return best;
}