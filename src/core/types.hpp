#pragma once
#include <cstdint>
#include <array>

using piece_size = uint8_t;
using color_size = uint8_t;

constexpr int ROWS = 8;
constexpr int COLS = 8;

enum class Piece : piece_size {
    Empty = 0,
    WK, WQ, WN, WB, WP, WR,
    BK, BQ, BN, BB, BP, BR
};

enum class Color : color_size {
    White = 0,
    Black = 1
};

enum CastlingRights : uint8_t {
    CR_NONE = 0,
    CR_WK   = 1 << 0,
    CR_WQ   = 1 << 1,
    CR_BK   = 1 << 2,
    CR_BQ   = 1 << 3
};

struct Square {
    int row;
    int col;
};

constexpr int make_sq(int file, int rank) {
    return rank * 8 + file;
}

constexpr int file_of(int sq) {
    return sq % 8;
}

constexpr int rank_of(int sq) {
    return sq / 8;
}

constexpr Square to_square(int sq) {
    return Square{rank_of(sq), file_of(sq)};
}

constexpr bool is_empty(Piece p) {
    return p == Piece::Empty;
}

constexpr bool is_white(Piece p) {
    return p == Piece::WK || p == Piece::WQ || p == Piece::WN ||
           p == Piece::WB || p == Piece::WP || p == Piece::WR;
}

constexpr bool is_black(Piece p) {
    return p == Piece::BK || p == Piece::BQ || p == Piece::BN ||
           p == Piece::BB || p == Piece::BP || p == Piece::BR;
}

constexpr Color other(Color c) {
    return c == Color::White ? Color::Black : Color::White;
}

struct PiecePlacement {
    int sq;
    Piece pc;
};

inline constexpr std::array<PiecePlacement, 32> STARTPOS_PLACEMENTS{{
    {make_sq(0,0), Piece::WR},
    {make_sq(1,0), Piece::WN},
    {make_sq(2,0), Piece::WB},
    {make_sq(3,0), Piece::WQ},
    {make_sq(4,0), Piece::WK},
    {make_sq(5,0), Piece::WB},
    {make_sq(6,0), Piece::WN},
    {make_sq(7,0), Piece::WR},

    {make_sq(0,1), Piece::WP},
    {make_sq(1,1), Piece::WP},
    {make_sq(2,1), Piece::WP},
    {make_sq(3,1), Piece::WP},
    {make_sq(4,1), Piece::WP},
    {make_sq(5,1), Piece::WP},
    {make_sq(6,1), Piece::WP},
    {make_sq(7,1), Piece::WP},

    {make_sq(0,6), Piece::BP},
    {make_sq(1,6), Piece::BP},
    {make_sq(2,6), Piece::BP},
    {make_sq(3,6), Piece::BP},
    {make_sq(4,6), Piece::BP},
    {make_sq(5,6), Piece::BP},
    {make_sq(6,6), Piece::BP},
    {make_sq(7,6), Piece::BP},

    {make_sq(0,7), Piece::BR},
    {make_sq(1,7), Piece::BN},
    {make_sq(2,7), Piece::BB},
    {make_sq(3,7), Piece::BQ},
    {make_sq(4,7), Piece::BK},
    {make_sq(5,7), Piece::BB},
    {make_sq(6,7), Piece::BN},
    {make_sq(7,7), Piece::BR}
}};

