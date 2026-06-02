#pragma once

#include "core/types.hpp"
#include "core/move.hpp"
#include "core/movelist.hpp"
#include "../position/bbposition.hpp"

class BitboardMoveGen {
public:
    static bool square_attacked(const BitboardPosition& pos, int sq, Color by);
    static bool in_check(const BitboardPosition& pos, Color who);

    static void generate_pseudo(const BitboardPosition& pos, MoveList& out);
    static void generate_legal(BitboardPosition& pos, MoveList& out);

    static bool has_any_legal_move(BitboardPosition& pos);

private:
    static void gen_pawns(const BitboardPosition& pos, MoveList& out);
    static void gen_knights(const BitboardPosition& pos, MoveList& out);
    static void gen_bishops(const BitboardPosition& pos, MoveList& out);
    static void gen_rooks(const BitboardPosition& pos, MoveList& out);
    static void gen_queens(const BitboardPosition& pos, MoveList& out);
    static void gen_king(const BitboardPosition& pos, MoveList& out);
};