#include "position/bbposition.hpp"
#include "search/search_strategy.hpp"
#include "core/uci.hpp"
#include "openings/opening_book.h"
#include "eval/eval_strategy.hpp"

#include <atomic>
#include <cstdlib>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <vector>

namespace chess {

struct DatasetRow {
    int game_id;
    int ply;
    std::string fen;
    std::string move_uci;
    int engine_score;
    float outcome;
    bool used_book_start;
};

static std::string csv_escape(const std::string& s) {
    std::string out = "\"";
    for (char c : s) {
        if (c == '"') out += "\"\"";
        else out += c;
    }
    out += "\"";
    return out;
}

static float compute_outcome(BitboardPosition& pos) {
    if (BitboardMoveGen::in_check(pos, pos.side_to_move())) {
        return pos.side_to_move() == Color::White ? 0.0f : 1.0f;
    }
    return 0.5f;
}

static BitboardPosition choose_start_position(
    const PositionBook& book,
    bool& used_book_start,
    std::mt19937& rng
) {
    if (!book.empty()) {
        std::uniform_int_distribution<size_t> dist(0, book.size() - 1);
        used_book_start = true;
        return BitboardPosition::from_fen(book[dist(rng)]);
    }

    used_book_start = false;
    return BitboardPosition::startpos();
}

static std::vector<DatasetRow> play_one_game(
    int game_id,
    int depth,
    int max_plies,
    const PositionBook& book,
    std::mt19937& rng
) {
    bool used_book_start = false;
    BitboardPosition pos = choose_start_position(book, used_book_start, rng);

    std::vector<DatasetRow> rows;
    rows.reserve(max_plies);

    MaterialEvaluator evaluator;

    for (int ply = 0; ply < max_plies; ++ply) {
        if (pos.is_draw_by_fifty() || pos.is_draw_by_repetition()) {
            for (auto& row : rows) row.outcome = 0.5f;
            return rows;
        }

        MoveList moves;
        BitboardMoveGen::generate_legal(pos, moves);

        if (moves.size == 0) {
            float final_outcome = compute_outcome(pos);
            for (auto& row : rows) row.outcome = final_outcome;
            return rows;
        }

        Search search(evaluator);
        auto result = search.minimax(pos, depth);

        DatasetRow row;
        row.game_id = game_id;
        row.ply = ply;
        row.fen = pos.to_fen();
        row.move_uci = move_to_uci(result.best);
        row.engine_score = result.score;
        row.outcome = -1.0f;
        row.used_book_start = used_book_start;

        rows.push_back(row);
        pos.make_move(result.best);
    }

    for (auto& row : rows) row.outcome = 0.5f;
    return rows;
}

static void write_header(std::ofstream& out) {
    out << "game_id,ply,fen,move_uci,engine_score,outcome,used_book_start\n";
}

static void write_rows(std::ofstream& out, const std::vector<DatasetRow>& rows) {
    for (const auto& row : rows) {
        out << row.game_id << ","
            << row.ply << ","
            << csv_escape(row.fen) << ","
            << row.move_uci << ","
            << row.engine_score << ","
            << row.outcome << ","
            << (row.used_book_start ? 1 : 0)
            << "\n";
    }
}

} // namespace chess

int main(int argc, char** argv) {
    int target_positions = 10000;
    int depth = 3;
    int max_plies = 300;
    std::string output_path = "selfplay_dataset.csv";
    std::string book_path = "src/openings/popularpos_lichess_v3.epd";
    int num_threads = 4;

    if (argc >= 2) target_positions = std::atoi(argv[1]);
    if (argc >= 3) depth = std::atoi(argv[2]);
    if (argc >= 4) max_plies = std::atoi(argv[3]);
    if (argc >= 5) output_path = argv[4];
    if (argc >= 6) book_path = argv[5];
    if (argc >= 7) num_threads = std::atoi(argv[6]);

    if (num_threads <= 0) {
        num_threads = 1;
    }

    chess::PositionBook book = chess::loadEPD(book_path);

    std::cout << "Loaded " << book.size()
              << " book starting positions from " << book_path << "\n";

    std::cout << "Using " << num_threads << " threads\n";

    std::atomic<int> total_positions{0};
    std::atomic<int> next_game_id{0};
    std::mutex print_mutex;

    auto worker = [&](int thread_id) {
        std::string part_path = output_path + ".part" + std::to_string(thread_id);

        std::ofstream out(part_path);
        if (!out) {
            std::lock_guard<std::mutex> lock(print_mutex);
            std::cerr << "Thread " << thread_id
                      << " failed to open " << part_path << "\n";
            return;
        }

        chess::write_header(out);

        std::random_device rd;
        std::mt19937 rng(rd() + thread_id * 9973);

        while (total_positions.load() < target_positions) {
            int game_id = next_game_id.fetch_add(1);

            auto rows = chess::play_one_game(
                game_id,
                depth,
                max_plies,
                book,
                rng
            );

            chess::write_rows(out, rows);
            out.flush();

            int new_total = total_positions.fetch_add(
                static_cast<int>(rows.size())
            ) + static_cast<int>(rows.size());

            {
                std::lock_guard<std::mutex> lock(print_mutex);
                std::cout << "[thread " << thread_id << "] finished game "
                          << game_id << " with " << rows.size()
                          << " positions | total = "
                          << new_total << "/"
                          << target_positions << "\n";
            }
        }
    };

    std::vector<std::thread> threads;
    threads.reserve(num_threads);

    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back(worker, t);
    }

    for (auto& t : threads) {
        t.join();
    }

    std::ofstream final_out(output_path);
    if (!final_out) {
        std::cerr << "Failed to open final output file: " << output_path << "\n";
        return 1;
    }

    chess::write_header(final_out);

    for (int t = 0; t < num_threads; ++t) {
        std::string part_path = output_path + ".part" + std::to_string(t);
        std::ifstream part(part_path);

        if (!part) {
            continue;
        }

        std::string line;
        bool first_line = true;

        while (std::getline(part, line)) {
            if (first_line) {
                first_line = false;
                continue;
            }

            if (!line.empty()) {
                final_out << line << "\n";
            }
        }

        part.close();
        std::remove(part_path.c_str());
    }

    std::cout << "Wrote " << total_positions.load()
              << " positions to: " << output_path << "\n";

    return 0;
}
