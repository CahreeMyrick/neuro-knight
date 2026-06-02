#pragma once

#include <cstdint>


enum Promotion : uint8_t {
    PROMO_NONE = 0,
    PROMO_Q,
    PROMO_R,
    PROMO_B,
    PROMO_N
};

struct Move {
    uint8_t from = 0;
    uint8_t to = 0;
    uint8_t promo = PROMO_NONE;
};
