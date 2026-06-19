#pragma once
#include <vector>
#include <string>

namespace chess {
// A book is just a list of FEN strings — one per starting position
using PositionBook = std::vector<std::string>;

// Load all positions from an EPD file into a PositionBook
PositionBook loadEPD(const std::string& filepath);

// Pick a random FEN from the book to use as the game starting position
std::string pickRandomPosition(const PositionBook& book);

}