#include "render.hpp"
#include <sstream>


// ── Shared helpers ────────────────────────────────────────────────────────────

char Render::piece_char(Piece p) {
    switch (p) {
        case Piece::WP: return 'P'; case Piece::WN: return 'N';
        case Piece::WB: return 'B'; case Piece::WR: return 'R';
        case Piece::WQ: return 'Q'; case Piece::WK: return 'K';
        case Piece::BP: return 'p'; case Piece::BN: return 'n';
        case Piece::BB: return 'b'; case Piece::BR: return 'r';
        case Piece::BQ: return 'q'; case Piece::BK: return 'k';
        default:        return '.';
    }
}

// Shared rendering logic — accepts any callable piece_at(sq) -> Piece
template <typename PieceAtFn>
static std::string render_board(PieceAtFn piece_at, Color stm) {
    std::ostringstream out;

    out << "\n";
    out << "    a b c d e f g h\n";
    out << "  +-----------------+\n";

    for (int r = 7; r >= 0; --r) {
        out << (r + 1) << " | ";
        for (int f = 0; f < 8; ++f)
            out << Render::piece_char(piece_at(make_sq(f, r))) << ' ';
        out << "| " << (r + 1) << "\n";
    }

    out << "  +-----------------+\n";
    out << "    a b c d e f g h\n";
    out << "\nSide to move: " << (stm == Color::White ? "White" : "Black") << "\n";

    return out.str();
}

// ── Overloads ─────────────────────────────────────────────────────────────────

/*std::string Render::board_ascii(const Position& pos) {
    return render_board(
        [&](int sq) { return pos.at(sq); },
        pos.side_to_move()
    );
}*/


std::string Render::board_ascii(const BitboardPosition& pos) {
    return render_board(
        [&](int sq) { return pos.piece_at(sq); },
        pos.side_to_move()
    );
}

