#include "game.hpp"

void Game::run() {

    bool running = true;

    // instantiate board and search
    BitboardPosition pos = BitboardPosition::startpos();
    Search search(_opts.evaluator);

    std::cout << Render::board_ascii(pos) << std::endl;
     
    while (running) {
        if (pos.side_to_move() == _opts.engine_color) {
            auto result = search.minimax(pos, _opts.depth);
            std::cout << "Engine move: " << Parse::move_to_string(result.best) << std::endl;
            pos.make_move(result.best);
            std::cout << Render::board_ascii(pos) << std::endl;
        
        }

        else {
            std::string line;
            std::cout << "Enter move: ";
            std::getline(std::cin, line);
            
            // quit
            if (line == "quit") {
                runnnig = false;
            }

            // parse move
            auto parsed = Parse::two_squares(line);
            Move m;
            m.from = parsed->first;
            m.to = parsed->second;

            // check parsed move
            MoveList legal_moves;
            bool valid = false;
            BitboardMoveGen::generate_legal(pos, legal_moves);

            for (const auto &  lm : legal_moves.moves) {
                if (m.to == lm.to and m.from == lm.from) {
                    valid = true;
                }
            }

            if (!valid) {
                std::cout << "Choose a valid move: ";
                continue;
            }

            pos.make_move(m);
            std::cout << Render::board_ascii(pos) << std::endl;
        

        }


    }

}
