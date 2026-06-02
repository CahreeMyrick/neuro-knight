#include <iostream>
#include "../src/position/bbposition.hpp"
#include "../src/parsing/parse.hpp"
#include "../src/render/render.hpp"
#include "../src/core/movelist.hpp"
#include "../src/movegen/bb_movegen.hpp"
#include "../src/search/search_strategy.hpp"


int main() {

  // initialize board
  BitboardPosition bb = BitboardPosition::startpos();

  // viz board
  std::cout << Render::board_ascii(bb);

  while (true) {
    std::string move_candidate;
    std::cout << "Enter Move: ";
    std::getline(std::cin, move_candidate);

    auto parsed = Parse::two_squares(move_candidate);

    Move m;
    m.from = parsed->first;
    m.to = parsed->second;

    MoveList legal_moves;
    BitboardMoveGen::generate_legal(bb, legal_moves);
    bool valid = false;

    for (auto lm : legal_moves.moves) {
      if (m.to == lm.to & m.from == lm.from) {
        valid = true;
      }
    }

    if (valid == false){
      std::cout << "Choose a valid move. " << std::endl;
      continue;
    }

    bb.make_move(m);
    std::cout << Render::board_ascii(bb);

    if (bb.side_to_move() == Color::Black) {
      //MaterialEvaluator evaluator;
      NeuralEvaluator evaluator;
      Search search(evaluator);

      auto res = search.minimax(bb, 3);
      std::cout << "Engine Move: " << Parse::move_to_string(res.best) << std::endl; 

      bb.make_move(res.best);
      std::cout << Render::board_ascii(bb);
    }

  }

  return 0;
}