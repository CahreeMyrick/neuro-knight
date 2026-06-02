#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "core/types.hpp"
#include "core/move.hpp"

// BitboardPosition
// Owns chess position state only.
// Does NOT generate moves.
// Does NOT choose moves.
// Does NOT evaluate positions.

class BitboardPosition {
public:
    BitboardPosition();

    static BitboardPosition startpos();
    static BitboardPosition from_fen(const std::string& fen);

    // ----- Board state -----
    Piece   piece_at(int sq) const;
    Color   side_to_move() const;
    uint8_t castling_rights() const;
    int     ep_square() const;

    uint64_t pieces(Color c, int pt_idx) const;
    uint64_t occ(Color c) const;
    uint64_t occ_all() const;

    int king_square(Color c) const;

    // ----- State mutation -----
    void make_move(Move m);
    void undo_move();

    // ----- Draw / game state -----
    int  halfmove_clock() const;
    void set_halfmove_clock(int n);
    bool is_draw_by_fifty() const;

    uint64_t zobrist_key() const;
    bool     is_draw_by_repetition() const;

    void set_winner(int w);
    int  winner() const;

    std::string to_fen() const;

private:
    // Piece-type indices:
    // 0 = King
    // 1 = Queen
    // 2 = Knight
    // 3 = Bishop
    // 4 = Pawn
    // 5 = Rook
    uint64_t pcs_[2][6] = {};
    uint64_t occ_[2]    = {};
    uint64_t occ_all_   = 0;

    // Optional convenience cache.
    // The position is still bitboard-owned.
    Piece mailbox_[64] = {};

    Color   stm_      = Color::White;
    uint8_t cr_       = CR_NONE;
    int     ep_       = -1;
    int     halfmove_ = 0;
    int     winner_   = -1;

    std::vector<uint64_t> history_;

    struct State {
        uint8_t cr;
        int     ep;
        int     halfmove;
        Piece   captured;
        Piece   moving;
        bool    was_ep;
        Move    m;
    };

    std::vector<State> stack_;

private:
    static int   piece_ci(Piece p);
    static int   piece_ti(Piece p);
    static Piece promo_piece(Color c, Promotion pr);

    void put_piece(Piece p, int sq);
    void remove_piece(Piece p, int sq);
    void move_piece(Piece p, int from, int to);
};