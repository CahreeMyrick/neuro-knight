// apps/web-client/src/pages/GamePage/GamePage.js

export async function renderGamePage(root) {
  root.innerHTML = `
    <main class="page">
      <section class="main-col">
        <div class="card">
          <div class="card-header">
            <span class="card-label">Engine Match</span>
          </div>

          <div class="match-controls">
            <label>
              White Engine
              <select id="whiteEngine">
                <option value="neural">Neural</option>
                <option value="material">Material</option>
              </select>
            </label>

            <label>
              Black Engine
              <select id="blackEngine">
                <option value="material">Material</option>
                <option value="neural">Neural</option>
              </select>
            </label>

            <label>
              Depth
              <input id="depthInput" type="number" value="3" min="1" max="8" />
            </label>

            <label>
              Max Plies
              <input id="maxPliesInput" type="number" value="120" min="1" max="300" />
            </label>

            <button class="btn" id="startMatch">Start Match</button>
          </div>

          <div class="board-wrap">
            <div id="board"></div>
          </div>

          <div class="board-actions">
            <button class="btn btn-ghost btn-sm" id="prevMove">← Previous</button>
            <button class="btn btn-ghost btn-sm" id="nextMove">Next →</button>
            <button class="btn btn-ghost btn-sm" id="autoPlay">Auto Play</button>
            <button class="btn btn-ghost btn-sm" id="flip">⇅ Flip</button>
            <button class="btn btn-ghost btn-sm" id="reset">Reset</button>
          </div>
        </div>
      </section>

      <aside class="side-col">
        <div class="card">
          <div class="card-header">
            <span class="card-label">Game Info</span>
          </div>
          <div id="gameInfo" class="engine-log">Choose engines and start a match.</div>
        </div>

        <div class="card">
          <div class="card-header">
            <span class="card-label">Move Log</span>
          </div>
          <div id="moveLog" class="engine-log"></div>
        </div>
      </aside>
    </main>
  `;

  const board = Chessboard("board", {
    draggable: false,
    position: "start",
    pieceTheme: "./public/chessboardjs-1.0.0/img/chesspieces/wikipedia/{piece}.png"
  });

  const gameInfo = document.getElementById("gameInfo");
  const moveLog = document.getElementById("moveLog");

  let currentGame = null;
  let currentPly = -1;
  let autoPlayTimer = null;

  function renderGameInfo() {
    if (!currentGame) {
      gameInfo.textContent = "Choose engines and start a match.";
      return;
    }

    const current =
      currentPly >= 0 && currentPly < currentGame.plies.length
        ? currentGame.plies[currentPly]
        : null;

    gameInfo.textContent =
`White: ${currentGame.white_engine}
Black: ${currentGame.black_engine}
Result: ${currentGame.result ?? "In progress / unknown"}

Current Ply: ${currentPly + 1} / ${currentGame.plies.length}

${current ? `Move: ${current.move}
Engine: ${current.engine}
Score: ${current.score}` : "At starting position"}`;
  }

  function renderMoveLog() {
    if (!currentGame) {
      moveLog.textContent = "";
      return;
    }

    moveLog.innerHTML = currentGame.plies
      .map((ply, index) => {
        const active = index === currentPly ? "active-move" : "";

        return `
          <div class="move-row ${active}" data-ply="${index}">
            <span>${index + 1}.</span>
            <span>${ply.side}</span>
            <span>${ply.engine}</span>
            <span>${ply.move}</span>
            <span>${ply.score}</span>
          </div>
        `;
      })
      .join("");

    document.querySelectorAll(".move-row").forEach(row => {
      row.addEventListener("click", () => {
        goToPly(Number(row.dataset.ply));
      });
    });
  }

  function goToPly(plyIndex) {
    if (!currentGame) return;

    currentPly = plyIndex;

    if (currentPly < 0) {
      board.position(currentGame.start_fen || "start");
    } else {
      const ply = currentGame.plies[currentPly];
      board.position(ply.fen_after);
    }

    renderGameInfo();
    renderMoveLog();
  }

  async function startMatch() {
    const whiteEngine = document.getElementById("whiteEngine").value;
    const blackEngine = document.getElementById("blackEngine").value;
    const depth = Number(document.getElementById("depthInput").value);
    const maxPlies = Number(document.getElementById("maxPliesInput").value);

    gameInfo.textContent = "Running match...";
    moveLog.textContent = "";

    try {
      const response = await fetch("http://127.0.0.1:8000/api/match/start", {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({
          white_engine: whiteEngine,
          black_engine: blackEngine,
          depth,
          max_plies: maxPlies
        })
      });

      if (!response.ok) {
        gameInfo.textContent = `Failed to start match. Status: ${response.status}`;
        return;
      }

      currentGame = await response.json();

      if (currentGame.error) {
        gameInfo.textContent =
`Engine failed.

${currentGame.stderr || currentGame.error}`;
        return;
      }

      currentPly = -1;
      board.position(currentGame.start_fen || "start");

      renderGameInfo();
      renderMoveLog();
    } catch (err) {
      gameInfo.textContent =
`Failed to connect to backend.

${err.message}`;
    }
  }

  function nextMove() {
    if (!currentGame) return;
    if (currentPly >= currentGame.plies.length - 1) return;

    goToPly(currentPly + 1);
  }

  function prevMove() {
    if (!currentGame) return;
    if (currentPly <= -1) return;

    goToPly(currentPly - 1);
  }

  function toggleAutoPlay() {
    const btn = document.getElementById("autoPlay");

    if (autoPlayTimer) {
      clearInterval(autoPlayTimer);
      autoPlayTimer = null;
      btn.textContent = "Auto Play";
      return;
    }

    btn.textContent = "Pause";

    autoPlayTimer = setInterval(() => {
      if (!currentGame || currentPly >= currentGame.plies.length - 1) {
        clearInterval(autoPlayTimer);
        autoPlayTimer = null;
        btn.textContent = "Auto Play";
        return;
      }

      nextMove();
    }, 700);
  }

  document.getElementById("startMatch").addEventListener("click", startMatch);
  document.getElementById("nextMove").addEventListener("click", nextMove);
  document.getElementById("prevMove").addEventListener("click", prevMove);
  document.getElementById("autoPlay").addEventListener("click", toggleAutoPlay);

  document.getElementById("flip").addEventListener("click", () => {
    board.flip();
  });

  document.getElementById("reset").addEventListener("click", () => {
    goToPly(-1);
  });
}