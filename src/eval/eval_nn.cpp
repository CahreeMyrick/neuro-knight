#include "eval/eval_nn.hpp"
#include "eval/nn_weights.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>

static constexpr int INPUT_SIZE = 769;
static constexpr int HIDDEN_SIZE = 32;

// Must match Python encoder:
// index = square * 12 + color_offset + piece_type_index
// Python piece order:
// PAWN=0, KNIGHT=1, BISHOP=2, ROOK=3, QUEEN=4, KING=5
static int feature_index(int square, Color color, int bitboard_piece_type) {
    int piece_type_index = 0;

    // BitboardPosition type indices:
    // 0=K, 1=Q, 2=N, 3=B, 4=P, 5=R
    switch (bitboard_piece_type) {
        case 4: piece_type_index = 0; break; // Pawn
        case 2: piece_type_index = 1; break; // Knight
        case 3: piece_type_index = 2; break; // Bishop
        case 5: piece_type_index = 3; break; // Rook
        case 1: piece_type_index = 4; break; // Queen
        case 0: piece_type_index = 5; break; // King
        default: piece_type_index = 0; break;
    }

    int color_offset = (color == Color::White) ? 0 : 6;
    return square * 12 + color_offset + piece_type_index;
}

int NNEval::evaluate(const BitboardPosition& pos) {
    float hidden[HIDDEN_SIZE];

    for (int h = 0; h < HIDDEN_SIZE; ++h) {
        hidden[h] = nn_weights::b1[h];
    }

    // Sparse accumulation over active board features
    for (int ti = 0; ti < 6; ++ti) {
        uint64_t white_bb = pos.pieces(Color::White, ti);

        while (white_bb) {
            int sq = __builtin_ctzll(white_bb);
            white_bb &= (white_bb - 1);

            int idx = feature_index(sq, Color::White, ti);

            for (int h = 0; h < HIDDEN_SIZE; ++h) {
                hidden[h] += nn_weights::W1[h][idx];
            }
        }

        uint64_t black_bb = pos.pieces(Color::Black, ti);

        while (black_bb) {
            int sq = __builtin_ctzll(black_bb);
            black_bb &= (black_bb - 1);

            int idx = feature_index(sq, Color::Black, ti);

            for (int h = 0; h < HIDDEN_SIZE; ++h) {
                hidden[h] += nn_weights::W1[h][idx];
            }
        }
    }

    // Side-to-move feature at index 768
    if (pos.side_to_move() == Color::White) {
        constexpr int stm_idx = 768;
        for (int h = 0; h < HIDDEN_SIZE; ++h) {
            hidden[h] += nn_weights::W1[h][stm_idx];
        }
    }

    for (int h = 0; h < HIDDEN_SIZE; ++h) {
        if (hidden[h] < 0.0f) {
            hidden[h] = 0.0f;
        }
    }

    float out = nn_weights::b2[0];

    for (int h = 0; h < HIDDEN_SIZE; ++h) {
        out += nn_weights::W2[0][h] * hidden[h];
    }

    float normalized = std::tanh(out);

    // Training used tanh(clipped_cp / 400)
    normalized = std::max(-0.999f, std::min(0.999f, normalized));

    float cp = 400.0f * std::atanh(normalized);

    return static_cast<int>(cp);
}