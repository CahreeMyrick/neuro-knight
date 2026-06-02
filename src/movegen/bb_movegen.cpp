#include "movegen/bb_movegen.hpp"
#include "movegen/bb_attacks.hpp"

#include <cstdint>

namespace {

static constexpr uint64_t FILE_A = 0x0101010101010101ULL;
static constexpr uint64_t FILE_H = 0x8080808080808080ULL;

static constexpr uint64_t RANK_1 = 0x00000000000000FFULL;
static constexpr uint64_t RANK_3 = 0x0000000000FF0000ULL;
static constexpr uint64_t RANK_6 = 0x0000FF0000000000ULL;
static constexpr uint64_t RANK_8 = 0xFF00000000000000ULL;

static inline int pop_lsb(uint64_t& b) {
    int sq = __builtin_ctzll(b);
    b &= b - 1;
    return sq;
}

static inline void push_promos(MoveList& out, uint8_t from, uint8_t to) {
    out.push({from, to, PROMO_Q});
    out.push({from, to, PROMO_R});
    out.push({from, to, PROMO_B});
    out.push({from, to, PROMO_N});
}

} // namespace

bool BitboardMoveGen::square_attacked(const BitboardPosition& pos, int sq, Color by) {
    int ci = static_cast<int>(by);

    if (attacks::pawn(other(by), sq) & pos.pieces(by, 4)) return true;
    if (attacks::knight(sq) & pos.pieces(by, 2)) return true;
    if (attacks::king(sq)   & pos.pieces(by, 0)) return true;

    uint64_t bishops_queens = pos.pieces(by, 3) | pos.pieces(by, 1);
    uint64_t rooks_queens   = pos.pieces(by, 5) | pos.pieces(by, 1);

    if (attacks::bishop_otf(sq, pos.occ_all()) & bishops_queens) return true;
    if (attacks::rook_otf(sq, pos.occ_all())   & rooks_queens)   return true;

    return false;
}

bool BitboardMoveGen::in_check(const BitboardPosition& pos, Color who) {
    return square_attacked(pos, pos.king_square(who), other(who));
}

void BitboardMoveGen::generate_pseudo(const BitboardPosition& pos, MoveList& out) {
    out.clear();

    gen_pawns(pos, out);
    gen_knights(pos, out);
    gen_bishops(pos, out);
    gen_rooks(pos, out);
    gen_queens(pos, out);
    gen_king(pos, out);
}

void BitboardMoveGen::generate_legal(BitboardPosition& pos, MoveList& out) {
    MoveList pseudo;
    generate_pseudo(pos, pseudo);

    out.clear();

    Color mover = pos.side_to_move();

    for (int i = 0; i < pseudo.size; ++i) {
        Move m = pseudo.moves[i];

        pos.make_move(m);

        bool legal = !square_attacked(
            pos,
            pos.king_square(mover),
            pos.side_to_move()
        );

        pos.undo_move();

        if (legal) {
            out.push(m);
        }
    }
}

bool BitboardMoveGen::has_any_legal_move(BitboardPosition& pos) {
    MoveList pseudo;
    generate_pseudo(pos, pseudo);

    Color mover = pos.side_to_move();

    for (int i = 0; i < pseudo.size; ++i) {
        Move m = pseudo.moves[i];

        pos.make_move(m);

        bool legal = !square_attacked(
            pos,
            pos.king_square(mover),
            pos.side_to_move()
        );

        pos.undo_move();

        if (legal) return true;
    }

    return false;
}

void BitboardMoveGen::gen_pawns(const BitboardPosition& pos, MoveList& out) {
    Color stm = pos.side_to_move();
    int ci = static_cast<int>(stm);

    uint64_t pawns = pos.pieces(stm, 4);
    uint64_t them_occ = pos.occ(other(stm));
    uint64_t occ_all = pos.occ_all();

    if (stm == Color::White) {
        uint64_t single = (pawns << 8) & ~occ_all;
        uint64_t dbl = ((single & RANK_3) << 8) & ~occ_all;

        for (uint64_t b = single & ~RANK_8; b;) {
            int to = pop_lsb(b);
            out.push({static_cast<uint8_t>(to - 8), static_cast<uint8_t>(to), PROMO_NONE});
        }

        for (uint64_t b = single & RANK_8; b;) {
            int to = pop_lsb(b);
            push_promos(out, static_cast<uint8_t>(to - 8), static_cast<uint8_t>(to));
        }

        for (uint64_t b = dbl; b;) {
            int to = pop_lsb(b);
            out.push({static_cast<uint8_t>(to - 16), static_cast<uint8_t>(to), PROMO_NONE});
        }

        uint64_t caps_nw = ((pawns & ~FILE_A) << 7) & them_occ;
        uint64_t caps_ne = ((pawns & ~FILE_H) << 9) & them_occ;

        for (uint64_t b = caps_nw & ~RANK_8; b;) {
            int to = pop_lsb(b);
            out.push({static_cast<uint8_t>(to - 7), static_cast<uint8_t>(to), PROMO_NONE});
        }

        for (uint64_t b = caps_nw & RANK_8; b;) {
            int to = pop_lsb(b);
            push_promos(out, static_cast<uint8_t>(to - 7), static_cast<uint8_t>(to));
        }

        for (uint64_t b = caps_ne & ~RANK_8; b;) {
            int to = pop_lsb(b);
            out.push({static_cast<uint8_t>(to - 9), static_cast<uint8_t>(to), PROMO_NONE});
        }

        for (uint64_t b = caps_ne & RANK_8; b;) {
            int to = pop_lsb(b);
            push_promos(out, static_cast<uint8_t>(to - 9), static_cast<uint8_t>(to));
        }

        int ep = pos.ep_square();
        if (ep >= 0) {
            uint64_t ep_bb = 1ULL << ep;

            if (file_of(ep) > 0 && (pawns & (ep_bb >> 9))) {
                out.push({static_cast<uint8_t>(ep - 9), static_cast<uint8_t>(ep), PROMO_NONE});
            }

            if (file_of(ep) < 7 && (pawns & (ep_bb >> 7))) {
                out.push({static_cast<uint8_t>(ep - 7), static_cast<uint8_t>(ep), PROMO_NONE});
            }
        }

    } else {
        uint64_t single = (pawns >> 8) & ~occ_all;
        uint64_t dbl = ((single & RANK_6) >> 8) & ~occ_all;

        for (uint64_t b = single & ~RANK_1; b;) {
            int to = pop_lsb(b);
            out.push({static_cast<uint8_t>(to + 8), static_cast<uint8_t>(to), PROMO_NONE});
        }

        for (uint64_t b = single & RANK_1; b;) {
            int to = pop_lsb(b);
            push_promos(out, static_cast<uint8_t>(to + 8), static_cast<uint8_t>(to));
        }

        for (uint64_t b = dbl; b;) {
            int to = pop_lsb(b);
            out.push({static_cast<uint8_t>(to + 16), static_cast<uint8_t>(to), PROMO_NONE});
        }

        uint64_t caps_se = ((pawns & ~FILE_H) >> 7) & them_occ;
        uint64_t caps_sw = ((pawns & ~FILE_A) >> 9) & them_occ;

        for (uint64_t b = caps_se & ~RANK_1; b;) {
            int to = pop_lsb(b);
            out.push({static_cast<uint8_t>(to + 7), static_cast<uint8_t>(to), PROMO_NONE});
        }

        for (uint64_t b = caps_se & RANK_1; b;) {
            int to = pop_lsb(b);
            push_promos(out, static_cast<uint8_t>(to + 7), static_cast<uint8_t>(to));
        }

        for (uint64_t b = caps_sw & ~RANK_1; b;) {
            int to = pop_lsb(b);
            out.push({static_cast<uint8_t>(to + 9), static_cast<uint8_t>(to), PROMO_NONE});
        }

        for (uint64_t b = caps_sw & RANK_1; b;) {
            int to = pop_lsb(b);
            push_promos(out, static_cast<uint8_t>(to + 9), static_cast<uint8_t>(to));
        }

        int ep = pos.ep_square();
        if (ep >= 0) {
            uint64_t ep_bb = 1ULL << ep;

            if (file_of(ep) < 7 && (pawns & (ep_bb << 9))) {
                out.push({static_cast<uint8_t>(ep + 9), static_cast<uint8_t>(ep), PROMO_NONE});
            }

            if (file_of(ep) > 0 && (pawns & (ep_bb << 7))) {
                out.push({static_cast<uint8_t>(ep + 7), static_cast<uint8_t>(ep), PROMO_NONE});
            }
        }
    }
}

void BitboardMoveGen::gen_knights(const BitboardPosition& pos, MoveList& out) {
    Color stm = pos.side_to_move();

    for (uint64_t b = pos.pieces(stm, 2); b;) {
        int from = pop_lsb(b);
        uint64_t moves = attacks::knight(from) & ~pos.occ(stm);

        while (moves) {
            int to = pop_lsb(moves);
            out.push({static_cast<uint8_t>(from), static_cast<uint8_t>(to), PROMO_NONE});
        }
    }
}

void BitboardMoveGen::gen_bishops(const BitboardPosition& pos, MoveList& out) {
    Color stm = pos.side_to_move();

    for (uint64_t b = pos.pieces(stm, 3); b;) {
        int from = pop_lsb(b);
        uint64_t moves = attacks::bishop_otf(from, pos.occ_all()) & ~pos.occ(stm);

        while (moves) {
            int to = pop_lsb(moves);
            out.push({static_cast<uint8_t>(from), static_cast<uint8_t>(to), PROMO_NONE});
        }
    }
}

void BitboardMoveGen::gen_rooks(const BitboardPosition& pos, MoveList& out) {
    Color stm = pos.side_to_move();

    for (uint64_t b = pos.pieces(stm, 5); b;) {
        int from = pop_lsb(b);
        uint64_t moves = attacks::rook_otf(from, pos.occ_all()) & ~pos.occ(stm);

        while (moves) {
            int to = pop_lsb(moves);
            out.push({static_cast<uint8_t>(from), static_cast<uint8_t>(to), PROMO_NONE});
        }
    }
}

void BitboardMoveGen::gen_queens(const BitboardPosition& pos, MoveList& out) {
    Color stm = pos.side_to_move();

    for (uint64_t b = pos.pieces(stm, 1); b;) {
        int from = pop_lsb(b);

        uint64_t moves =
            (attacks::bishop_otf(from, pos.occ_all()) |
             attacks::rook_otf(from, pos.occ_all())) & ~pos.occ(stm);

        while (moves) {
            int to = pop_lsb(moves);
            out.push({static_cast<uint8_t>(from), static_cast<uint8_t>(to), PROMO_NONE});
        }
    }
}

void BitboardMoveGen::gen_king(const BitboardPosition& pos, MoveList& out) {
    Color stm = pos.side_to_move();
    Color them = other(stm);

    int from = pos.king_square(stm);

    uint64_t moves = attacks::king(from) & ~pos.occ(stm);

    while (moves) {
        int to = pop_lsb(moves);
        out.push({static_cast<uint8_t>(from), static_cast<uint8_t>(to), PROMO_NONE});
    }

    if (stm == Color::White) {
        if ((pos.castling_rights() & CR_WK) &&
            !(pos.occ_all() & ((1ULL << make_sq(5, 0)) | (1ULL << make_sq(6, 0)))) &&
            !square_attacked(pos, make_sq(4, 0), them) &&
            !square_attacked(pos, make_sq(5, 0), them)) {
            out.push({static_cast<uint8_t>(make_sq(4, 0)), static_cast<uint8_t>(make_sq(6, 0)), PROMO_NONE});
        }

        if ((pos.castling_rights() & CR_WQ) &&
            !(pos.occ_all() & ((1ULL << make_sq(3, 0)) | (1ULL << make_sq(2, 0)) | (1ULL << make_sq(1, 0)))) &&
            !square_attacked(pos, make_sq(4, 0), them) &&
            !square_attacked(pos, make_sq(3, 0), them)) {
            out.push({static_cast<uint8_t>(make_sq(4, 0)), static_cast<uint8_t>(make_sq(2, 0)), PROMO_NONE});
        }
    } else {
        if ((pos.castling_rights() & CR_BK) &&
            !(pos.occ_all() & ((1ULL << make_sq(5, 7)) | (1ULL << make_sq(6, 7)))) &&
            !square_attacked(pos, make_sq(4, 7), them) &&
            !square_attacked(pos, make_sq(5, 7), them)) {
            out.push({static_cast<uint8_t>(make_sq(4, 7)), static_cast<uint8_t>(make_sq(6, 7)), PROMO_NONE});
        }

        if ((pos.castling_rights() & CR_BQ) &&
            !(pos.occ_all() & ((1ULL << make_sq(3, 7)) | (1ULL << make_sq(2, 7)) | (1ULL << make_sq(1, 7)))) &&
            !square_attacked(pos, make_sq(4, 7), them) &&
            !square_attacked(pos, make_sq(3, 7), them)) {
            out.push({static_cast<uint8_t>(make_sq(4, 7)), static_cast<uint8_t>(make_sq(2, 7)), PROMO_NONE});
        }
    }
}