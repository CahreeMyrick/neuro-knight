#pragma once

#include <string>
#include "core/types.hpp"
#include "../position/bbposition.hpp"



class Render {
public:
    static char piece_char(Piece p);
    static std::string board_ascii(const BitboardPosition& pos);

};

