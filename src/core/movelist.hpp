#pragma once

#include <array>
#include "core/move.hpp"


struct MoveList {
    std::array<Move, 256> moves{};
    int size = 0;

    void clear() {
        size = 0;
    }

    void push(const Move& m) {
        if (size < static_cast<int>(moves.size())) {
            moves[size++] = m;
        }
    }
};
