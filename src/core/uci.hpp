// utils/uci.hpp
#pragma once
#include "core/move.hpp"
#include <string>

namespace chess {

inline std::string move_to_uci(Move m) {
    static const char files[] = "abcdefgh";
    static const char ranks[] = "12345678";
    static const char promos[] = " qrbn"; // index matches PROMO_Q/R/B/N

    std::string s;
    s.reserve(5);
    s += files[m.from % 8];
    s += ranks[m.from / 8];
    s += files[m.to   % 8];
    s += ranks[m.to   / 8];

    if (m.promo != PROMO_NONE)
        s += promos[m.promo]; // e.g. "e7e8q"

    return s;
}

} // namespace chess