#include "movegen/bb_attacks.hpp"

namespace attacks {

// File and rank edge masks used for boundary checks during table init and ray cast.
static constexpr uint64_t FILE_A = 0x0101010101010101ULL;
static constexpr uint64_t FILE_B = 0x0202020202020202ULL;
static constexpr uint64_t FILE_G = 0x4040404040404040ULL;
static constexpr uint64_t FILE_H = 0x8080808080808080ULL;
static constexpr uint64_t RANK_1 = 0x00000000000000FFULL;
static constexpr uint64_t RANK_8 = 0xFF00000000000000ULL;

static uint64_t KNIGHT_TBL[64];
static uint64_t KING_TBL[64];
static uint64_t WPAWN_TBL[64];
static uint64_t BPAWN_TBL[64];

static bool s_initialized = false;

void init() {
    if (s_initialized) return;

    for (int s = 0; s < 64; ++s) {
        uint64_t b = 1ULL << s;

        // Knight: all 8 L-shaped deltas with file-wrap guards
        uint64_t k = 0;
        k |= (b & ~FILE_H)           << 17; // +2r +1f
        k |= (b & ~FILE_A)           << 15; // +2r -1f
        k |= (b & ~FILE_H)           >> 15; // -2r +1f
        k |= (b & ~FILE_A)           >> 17; // -2r -1f
        k |= (b & ~(FILE_G | FILE_H)) << 10; // +1r +2f
        k |= (b & ~(FILE_A | FILE_B)) << 6;  // +1r -2f
        k |= (b & ~(FILE_G | FILE_H)) >> 6;  // -1r +2f
        k |= (b & ~(FILE_A | FILE_B)) >> 10; // -1r -2f
        KNIGHT_TBL[s] = k;

        // King: all 8 adjacent squares with file-wrap guards
        uint64_t kg = 0;
        kg |= b << 8;                // N
        kg |= b >> 8;                // S
        kg |= (b & ~FILE_H) << 1;   // E
        kg |= (b & ~FILE_A) >> 1;   // W
        kg |= (b & ~FILE_H) << 9;   // NE
        kg |= (b & ~FILE_A) << 7;   // NW
        kg |= (b & ~FILE_H) >> 7;   // SE
        kg |= (b & ~FILE_A) >> 9;   // SW
        KING_TBL[s] = kg;

        // Pawn captures (squares a pawn of that color attacks FROM sq)
        WPAWN_TBL[s] = ((b & ~FILE_H) << 9) | ((b & ~FILE_A) << 7); // NE | NW
        BPAWN_TBL[s] = ((b & ~FILE_H) >> 7) | ((b & ~FILE_A) >> 9); // SE | SW
    }

    s_initialized = true;
}

// Auto-initialize before main via static constructor
namespace { struct AutoInit { AutoInit() { init(); } } s_auto_init; }

uint64_t knight(int sq) { return KNIGHT_TBL[sq]; }
uint64_t king(int sq)   { return KING_TBL[sq]; }
uint64_t pawn(Color c, int sq) {
    return (c == Color::White) ? WPAWN_TBL[sq] : BPAWN_TBL[sq];
}

// --- On-the-fly sliding rays ---
// Each function steps one square at a time along its ray until it hits the
// board edge or a blocker (the blocker square is included in the result).

static inline uint64_t ray_north(int sq, uint64_t occ) {
    uint64_t mask = 0, b = 1ULL << sq;
    while (!(b & RANK_8)) { b <<= 8;  mask |= b; if (b & occ) break; }
    return mask;
}
static inline uint64_t ray_south(int sq, uint64_t occ) {
    uint64_t mask = 0, b = 1ULL << sq;
    while (!(b & RANK_1)) { b >>= 8;  mask |= b; if (b & occ) break; }
    return mask;
}
static inline uint64_t ray_east(int sq, uint64_t occ) {
    uint64_t mask = 0, b = 1ULL << sq;
    while (!(b & FILE_H)) { b <<= 1;  mask |= b; if (b & occ) break; }
    return mask;
}
static inline uint64_t ray_west(int sq, uint64_t occ) {
    uint64_t mask = 0, b = 1ULL << sq;
    while (!(b & FILE_A)) { b >>= 1;  mask |= b; if (b & occ) break; }
    return mask;
}
static inline uint64_t ray_ne(int sq, uint64_t occ) {
    uint64_t mask = 0, b = 1ULL << sq;
    while (!(b & FILE_H) && !(b & RANK_8)) { b <<= 9; mask |= b; if (b & occ) break; }
    return mask;
}
static inline uint64_t ray_nw(int sq, uint64_t occ) {
    uint64_t mask = 0, b = 1ULL << sq;
    while (!(b & FILE_A) && !(b & RANK_8)) { b <<= 7; mask |= b; if (b & occ) break; }
    return mask;
}
static inline uint64_t ray_se(int sq, uint64_t occ) {
    uint64_t mask = 0, b = 1ULL << sq;
    while (!(b & FILE_H) && !(b & RANK_1)) { b >>= 7; mask |= b; if (b & occ) break; }
    return mask;
}
static inline uint64_t ray_sw(int sq, uint64_t occ) {
    uint64_t mask = 0, b = 1ULL << sq;
    while (!(b & FILE_A) && !(b & RANK_1)) { b >>= 9; mask |= b; if (b & occ) break; }
    return mask;
}

uint64_t bishop_otf(int sq, uint64_t occ) {
    return ray_ne(sq, occ) | ray_nw(sq, occ) | ray_se(sq, occ) | ray_sw(sq, occ);
}
uint64_t rook_otf(int sq, uint64_t occ) {
    return ray_north(sq, occ) | ray_south(sq, occ) | ray_east(sq, occ) | ray_west(sq, occ);
}

} // namespace attacks

