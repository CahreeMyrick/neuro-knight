#include "parse.hpp"
#include <cctype>
#include <sstream>


static std::string trim_copy(const std::string& in) {
    size_t a = 0;
    while (a < in.size() && std::isspace((unsigned char)in[a])) ++a;
    size_t b = in.size();
    while (b > a && std::isspace((unsigned char)in[b - 1])) --b;
    return in.substr(a, b - a);
}

std::optional<int> Parse::square(const std::string& s0) {
    std::string s = trim_copy(s0);
    if (s.size() != 2) return std::nullopt;

    char file = (char)std::tolower((unsigned char)s[0]);
    char rank = s[1];

    if (file < 'a' || file > 'h') return std::nullopt;
    if (rank < '1' || rank > '8') return std::nullopt;

    int f = file - 'a';   // 0..7
    int r = rank - '1';   // 0..7 (rank1=0)
    return make_sq(f, r);
}

std::optional<std::pair<int,int>> Parse::two_squares(const std::string& line0) {
    std::string line = trim_copy(line0);
    if (line.empty()) return std::nullopt;

    std::istringstream iss(line);
    std::string a, b;
    if (!(iss >> a >> b)) return std::nullopt;

    // allow only exactly two tokens 
    std::string extra;
    if (iss >> extra) return std::nullopt;

    auto sqa = square(a);
    auto sqb = square(b);
    if (!sqa || !sqb) return std::nullopt;
    return std::make_pair(*sqa, *sqb);
}


std::string Parse::square_to_string(int sq) {
    char file = static_cast<char>('a' + file_of(sq));
    char rank = static_cast<char>('1' + rank_of(sq));
    std::string out;
    out.push_back(file);
    out.push_back(rank);
    return out;
}

std::string Parse::move_to_string(const Move& m) {
    std::string s = square_to_string(m.from) + square_to_string(m.to);
    if (m.promo != PROMO_NONE) {
        switch (m.promo) {
            case PROMO_Q: s += 'q'; break;
            case PROMO_R: s += 'r'; break;
            case PROMO_B: s += 'b'; break;
            case PROMO_N: s += 'n'; break;
            default:                break;
        }
    }
    return s;

}



