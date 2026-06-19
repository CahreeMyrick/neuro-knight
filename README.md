# Neuro-Knight

Neuro-Knight is a chess engine project that combines traditional search techniques with learned evaluation functions. The project includes:

* A C++ chess engine built with CMake
* Material and neural-network evaluation engines
* A backend API for running engine matches
* A web interface for visualizing and replaying games

## Requirements

### Python

Install dependencies into either a Conda environment or a Python virtual environment.

```bash
pip install -r requirements.txt
```

### C++

Requirements:

* CMake 3.16+
* C++20 compatible compiler

---

## Building the Engine

From the repository root:

```bash
cmake -S . -B build
cmake --build build
```

This creates the engine executables inside the `build/` directory.

---

## Running the Web Application

### Start the Backend

```bash
cd apps/match_api
python -m uvicorn main:app --port 8000
```

The backend will start on:

```text
http://localhost:8000
```

### Start the Frontend

Open a second terminal:

```bash
cd apps/web-client
python -m http.server 8080
```

The frontend will be available at:

```text
http://localhost:8080
```

---

## Using the Interface

### Match Configuration

1. Select the maximum number of plies (half-moves) to play.
2. Select a search depth between **1 and 4**.
3. Choose the evaluation engine for White:

   * Material
   * Neural
4. Choose the evaluation engine for Black:

   * Material
   * Neural

### Running a Match

Click **Start Game** to begin a match and load the completed game.

### Game Controls

* **Autoplay** – Automatically plays through the entire game.
* **Next** – Advances one move forward.
* **Previous** – Moves one step backward through the game.

---

## Engine Types

### Material Engine

Uses a traditional hand-crafted material evaluation function.

### Neural Engine

Uses a learned neural-network evaluation function trained on chess positions.

---

## Demo

https://github.com/user-attachments/assets/1c66593d-a097-4b76-95e2-1ef066a461ad









