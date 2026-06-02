#include "bbposition.hpp"

#include <cassert>
#include <cstdlib>
#include <sstream>

struct ZobristTable {
    uint64_t pieces[64][13];
    uint64_t side_to_move;
    uint64_t castling[16];
    uint64_t en_passant[8];

    ZobristTable() {
        uint64_t s = 6364136223846793005ULL;

        auto next = [&]() {
            s ^= s >> 12;
            s ^= s << 25;
            s ^= s >> 27;
            return s * 2685821657736338717ULL;
        };

        for (auto& sq : pieces)
            for (auto& p : sq)
                p = next();

        side_to_move = next();

        for (auto& c : castling)   c = next();
        for (auto& e : en_passant) e = next();
    }
};

const ZobristTable ZOBRIST;

int piece_index(Piece p) {
    return static_cast<int>(p);
}

BitboardPosition::BitboardPosition() {
    for (int i = 0; i < 64; ++i)
        mailbox_[i] = Piece::Empty;
}

BitboardPosition BitboardPosition::startpos() {
    return from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
}

BitboardPosition BitboardPosition::from_fen(const std::string& fen) {
    BitboardPosition pos;

    std::istringstream ss(fen);
    std::string token;

    ss >> token;

    int rank = 7;
    int file = 0;

    for (char c : token) {
        if (c == '/') {
            --rank;
            file = 0;
            continue;
        }

        if (c >= '1' && c <= '8') {
            file += c - '0';
            continue;
        }

        Piece p = Piece::Empty;

        switch (c) {
            case 'K': p = Piece::WK; break;
            case 'Q': p = Piece::WQ; break;
            case 'N': p = Piece::WN; break;
            case 'B': p = Piece::WB; break;
            case 'P': p = Piece::WP; break;
            case 'R': p = Piece::WR; break;

            case 'k': p = Piece::BK; break;
            case 'q': p = Piece::BQ; break;
            case 'n': p = Piece::BN; break;
            case 'b': p = Piece::BB; break;
            case 'p': p = Piece::BP; break;
            case 'r': p = Piece::BR; break;

            default: break;
        }

        if (p != Piece::Empty)
            pos.put_piece(p, make_sq(file, rank));

        ++file;
    }

    ss >> token;
    pos.stm_ = (token == "b") ? Color::Black : Color::White;

    ss >> token;
    pos.cr_ = CR_NONE;

    if (token != "-") {
        for (char c : token) {
            switch (c) {
                case 'K': pos.cr_ |= CR_WK; break;
                case 'Q': pos.cr_ |= CR_WQ; break;
                case 'k': pos.cr_ |= CR_BK; break;
                case 'q': pos.cr_ |= CR_BQ; break;
            }
        }
    }

    ss >> token;
    if (token == "-") {
        pos.ep_ = -1;
    } else {
        int f = token[0] - 'a';
        int r = token[1] - '1';
        pos.ep_ = make_sq(f, r);
    }

    int fullmove = 1;
    ss >> pos.halfmove_;
    ss >> fullmove;

    pos.history_.push_back(pos.zobrist_key());

    return pos;
}

Piece BitboardPosition::piece_at(int sq) const {
    return mailbox_[sq];
}

Color BitboardPosition::side_to_move() const {
    return stm_;
}

uint8_t BitboardPosition::castling_rights() const {
    return cr_;
}

int BitboardPosition::ep_square() const {
    return ep_;
}

uint64_t BitboardPosition::pieces(Color c, int pt_idx) const {
    return pcs_[static_cast<int>(c)][pt_idx];
}

uint64_t BitboardPosition::occ(Color c) const {
    return occ_[static_cast<int>(c)];
}

uint64_t BitboardPosition::occ_all() const {
    return occ_all_;
}

int BitboardPosition::king_square(Color c) const {
    uint64_t k = pcs_[static_cast<int>(c)][0];
    assert(k != 0);
    return __builtin_ctzll(k);
}

int BitboardPosition::halfmove_clock() const {
    return halfmove_;
}

void BitboardPosition::set_halfmove_clock(int n) {
    halfmove_ = n;
}

bool BitboardPosition::is_draw_by_fifty() const {
    return halfmove_ >= 100;
}

void BitboardPosition::set_winner(int w) {
    winner_ = w;
}

int BitboardPosition::winner() const {
    return winner_;
}

uint64_t BitboardPosition::zobrist_key() const {
    uint64_t key = 0;

    for (int sq = 0; sq < 64; ++sq) {
        Piece p = mailbox_[sq];

        if (p != Piece::Empty)
            key ^= ZOBRIST.pieces[sq][piece_index(p)];
    }

    if (stm_ == Color::Black)
        key ^= ZOBRIST.side_to_move;

    key ^= ZOBRIST.castling[cr_];

    if (ep_ >= 0)
        key ^= ZOBRIST.en_passant[file_of(ep_)];

    return key;
}

bool BitboardPosition::is_draw_by_repetition() const {
    if (history_.empty())
        return false;

    uint64_t key = history_.back();
    int count = 0;

    for (int i = static_cast<int>(history_.size()) - 1; i >= 0; i -= 2) {
        if (history_[i] == key)
            ++count;

        if (count >= 3)
            return true;
    }

    return false;
}

std::string BitboardPosition::to_fen() const {
    static const char piece_chars[] = {
        ' ',
        'K', 'Q', 'N', 'B', 'P', 'R',
        'k', 'q', 'n', 'b', 'p', 'r'
    };

    std::string fen;

    for (int rank = 7; rank >= 0; --rank) {
        int empty = 0;

        for (int file = 0; file < 8; ++file) {
            int sq = make_sq(file, rank);
            Piece p = mailbox_[sq];

            if (p == Piece::Empty) {
                ++empty;
            } else {
                if (empty > 0) {
                    fen += static_cast<char>('0' + empty);
                    empty = 0;
                }

                fen += piece_chars[static_cast<int>(p)];
            }
        }

        if (empty > 0)
            fen += static_cast<char>('0' + empty);

        if (rank > 0)
            fen += '/';
    }

    fen += (stm_ == Color::White) ? " w " : " b ";

    if (cr_ == CR_NONE) {
        fen += '-';
    } else {
        if (cr_ & CR_WK) fen += 'K';
        if (cr_ & CR_WQ) fen += 'Q';
        if (cr_ & CR_BK) fen += 'k';
        if (cr_ & CR_BQ) fen += 'q';
    }

    fen += ' ';

    if (ep_ < 0) {
        fen += '-';
    } else {
        fen += static_cast<char>('a' + file_of(ep_));
        fen += static_cast<char>('1' + rank_of(ep_));
    }

    int fullmove = 1 + static_cast<int>(history_.size()) / 2;

    fen += ' ';
    fen += std::to_string(halfmove_);
    fen += ' ';
    fen += std::to_string(fullmove);

    return fen;
}

void BitboardPosition::make_move(Move m) {
    int from = m.from;
    int to   = m.to;

    Piece moving = mailbox_[from];

    State st;
    st.cr       = cr_;
    st.ep       = ep_;
    st.halfmove = halfmove_;
    st.captured = Piece::Empty;
    st.moving   = moving;
    st.was_ep   = false;
    st.m        = m;

    assert(moving != Piece::Empty);

    bool pawn_move = moving == Piece::WP || moving == Piece::BP;

    if (pawn_move && ep_ >= 0 && to == ep_ && mailbox_[to] == Piece::Empty) {
        st.was_ep = true;

        int cap_sq = (stm_ == Color::White) ? to - 8 : to + 8;
        st.captured = mailbox_[cap_sq];

        remove_piece(st.captured, cap_sq);
        move_piece(moving, from, to);
    } else {
        if (mailbox_[to] != Piece::Empty) {
            st.captured = mailbox_[to];
            remove_piece(st.captured, to);
        }

        move_piece(moving, from, to);

        if (m.promo != PROMO_NONE) {
            remove_piece(moving, to);
            put_piece(promo_piece(stm_, static_cast<Promotion>(m.promo)), to);
        }

        bool king_move = moving == Piece::WK || moving == Piece::BK;
        bool castle = king_move && std::abs(file_of(to) - file_of(from)) == 2;

        if (castle) {
            if (stm_ == Color::White) {
                if (to == make_sq(6, 0))
                    move_piece(Piece::WR, make_sq(7, 0), make_sq(5, 0));
                else
                    move_piece(Piece::WR, make_sq(0, 0), make_sq(3, 0));
            } else {
                if (to == make_sq(6, 7))
                    move_piece(Piece::BR, make_sq(7, 7), make_sq(5, 7));
                else
                    move_piece(Piece::BR, make_sq(0, 7), make_sq(3, 7));
            }
        }
    }

    if (moving == Piece::WK) cr_ &= ~(CR_WK | CR_WQ);
    if (moving == Piece::BK) cr_ &= ~(CR_BK | CR_BQ);

    if (moving == Piece::WR) {
        if (from == make_sq(7, 0)) cr_ &= ~CR_WK;
        if (from == make_sq(0, 0)) cr_ &= ~CR_WQ;
    }

    if (moving == Piece::BR) {
        if (from == make_sq(7, 7)) cr_ &= ~CR_BK;
        if (from == make_sq(0, 7)) cr_ &= ~CR_BQ;
    }

    if (st.captured == Piece::WR) {
        if (to == make_sq(7, 0)) cr_ &= ~CR_WK;
        if (to == make_sq(0, 0)) cr_ &= ~CR_WQ;
    }

    if (st.captured == Piece::BR) {
        if (to == make_sq(7, 7)) cr_ &= ~CR_BK;
        if (to == make_sq(0, 7)) cr_ &= ~CR_BQ;
    }

    ep_ = -1;

    if (moving == Piece::WP && rank_of(to) - rank_of(from) == 2)
        ep_ = make_sq(file_of(from), rank_of(from) + 1);

    if (moving == Piece::BP && rank_of(from) - rank_of(to) == 2)
        ep_ = make_sq(file_of(from), rank_of(from) - 1);

    if (pawn_move || st.captured != Piece::Empty)
        halfmove_ = 0;
    else
        ++halfmove_;

    stm_ = other(stm_);

    history_.push_back(zobrist_key());
    stack_.push_back(st);
}

void BitboardPosition::undo_move() {
    assert(!stack_.empty());

    State st = stack_.back();
    stack_.pop_back();

    if (!history_.empty())
        history_.pop_back();

    stm_      = other(stm_);
    cr_       = st.cr;
    ep_       = st.ep;
    halfmove_ = st.halfmove;

    int from = st.m.from;
    int to   = st.m.to;

    bool king_move = st.moving == Piece::WK || st.moving == Piece::BK;
    bool castle = king_move && std::abs(file_of(to) - file_of(from)) == 2;

    if (castle) {
        if (stm_ == Color::White) {
            if (to == make_sq(6, 0))
                move_piece(Piece::WR, make_sq(5, 0), make_sq(7, 0));
            else
                move_piece(Piece::WR, make_sq(3, 0), make_sq(0, 0));
        } else {
            if (to == make_sq(6, 7))
                move_piece(Piece::BR, make_sq(5, 7), make_sq(7, 7));
            else
                move_piece(Piece::BR, make_sq(3, 7), make_sq(0, 7));
        }
    }

    if (st.m.promo != PROMO_NONE) {
        remove_piece(mailbox_[to], to);
        put_piece(st.moving, from);
    } else {
        move_piece(st.moving, to, from);
    }

    if (st.captured != Piece::Empty) {
        if (st.was_ep) {
            int cap_sq = (stm_ == Color::White) ? to - 8 : to + 8;
            put_piece(st.captured, cap_sq);
        } else {
            put_piece(st.captured, to);
        }
    }
}

int BitboardPosition::piece_ci(Piece p) {
    return (static_cast<int>(p) - 1) / 6;
}

int BitboardPosition::piece_ti(Piece p) {
    return (static_cast<int>(p) - 1) % 6;
}

Piece BitboardPosition::promo_piece(Color c, Promotion pr) {
    if (c == Color::White) {
        switch (pr) {
            case PROMO_Q: return Piece::WQ;
            case PROMO_R: return Piece::WR;
            case PROMO_B: return Piece::WB;
            case PROMO_N: return Piece::WN;
            default:      return Piece::WQ;
        }
    }

    switch (pr) {
        case PROMO_Q: return Piece::BQ;
        case PROMO_R: return Piece::BR;
        case PROMO_B: return Piece::BB;
        case PROMO_N: return Piece::BN;
        default:      return Piece::BQ;
    }
}

void BitboardPosition::put_piece(Piece p, int sq) {
    assert(p != Piece::Empty);

    uint64_t b = 1ULL << sq;

    pcs_[piece_ci(p)][piece_ti(p)] |= b;
    occ_[piece_ci(p)]              |= b;
    occ_all_                       |= b;

    mailbox_[sq] = p;
}

void BitboardPosition::remove_piece(Piece p, int sq) {
    assert(p != Piece::Empty);

    uint64_t b = 1ULL << sq;

    pcs_[piece_ci(p)][piece_ti(p)] &= ~b;
    occ_[piece_ci(p)]              &= ~b;
    occ_all_                       &= ~b;

    mailbox_[sq] = Piece::Empty;
}

void BitboardPosition::move_piece(Piece p, int from, int to) {
    assert(p != Piece::Empty);

    uint64_t bits = (1ULL << from) | (1ULL << to);

    pcs_[piece_ci(p)][piece_ti(p)] ^= bits;
    occ_[piece_ci(p)]              ^= bits;
    occ_all_                       ^= bits;

    mailbox_[from] = Piece::Empty;
    mailbox_[to]   = p;
}