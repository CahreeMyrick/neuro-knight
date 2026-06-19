#include "opening_book.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <random>

namespace chess {
// ── Loader ────────────────────────────────────────────────────────────────────
PositionBook loadEPD(const std::string& filepath) {
    PositionBook positions;
    std::ifstream file(filepath);

    if (!file.is_open()) {
        std::cerr << "[book] ERROR: Could not open " << filepath << "\n";
        return positions;
    }

    std::string line;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;

        std::istringstream ss(line);
        std::vector<std::string> tokens;
        std::string part;
        while (ss >> part) tokens.push_back(part);

        // EPD needs at least 4 fields: placement, side, castling, en passant
        if (tokens.size() < 4) continue;

        // Reconstruct a valid FEN (EPD has no move counters — add defaults)
        std::string fen = tokens[0] + " " + tokens[1] + " " +
                          tokens[2] + " " + tokens[3] + " 0 1";

        positions.push_back(fen);
    }

    std::cout << "[book] Loaded " << positions.size()
              << " positions from " << filepath << "\n";
    return positions;
}

// ── Random picker ─────────────────────────────────────────────────────────────

std::string pickRandomPosition(const PositionBook& book) {
    if (book.empty()) return "";

    static std::mt19937 rng(std::random_device{}());
    std::uniform_int_distribution<size_t> dist(0, book.size() - 1);
    return book[dist(rng)];
}

}
