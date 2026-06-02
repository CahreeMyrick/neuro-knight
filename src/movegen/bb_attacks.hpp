#pragma once

#include <cstdint>
#include "core/types.hpp"

namespace attacks {

// Populates all lookup tables. Called automatically at program startup.
// Safe to call multiple times (idempotent after first call).
void init();

// Precomputed attack masks for non-sliding pieces.
// Returns a bitboard of all squares the given piece type attacks from sq.
uint64_t knight(int sq);
uint64_t king(int sq);

// Returns squares a pawn of color c on sq attacks (its capture targets).
// To find which squares can attack sq, use pawn(other(by), sq).
uint64_t pawn(Color c, int sq);

// On-the-fly sliding piece attack masks.
// occ = full occupancy bitboard (both colors combined).
// Returns all squares reachable along each ray, including the first blocker.
// Future Phase: replace these with magic bitboard lookups without changing callers.
uint64_t bishop_otf(int sq, uint64_t occ);
uint64_t rook_otf(int sq, uint64_t occ);

} // namespace attacks
