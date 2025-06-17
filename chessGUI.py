from PyQt5.QtSvg import QSvgWidget
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QPushButton
import chess.svg

class ChessBoard(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Chessoboard GUI")
        self.setGeometry(320, 320, 800, 800)
        self.layout = QVBoxLayout(self)

        # SVG widget to display the chessboard
        self.widgetSvg = QSvgWidget(parent=self)
        self.layout.addWidget(self.widgetSvg)

        # Label to display the best move
        self.label = QLabel("Best Move:")
        self.layout.addWidget(self.label)

        # Button to load the previous game state
        self.load_previous_button = QPushButton("Reload Previous State")
        self.layout.addWidget(self.load_previous_button)

        self.piece_positions_label = QLabel("Taş Konumları:")
        self.piece_positions_label.setStyleSheet("""
            QLabel {
                font-family: Consolas, monospace;
                font-size: 14px;
                padding: 6px;
                border: 1px solid #ccc;
                background-color: #f9f9f9;
            }
        """)
        self.layout.addWidget(self.piece_positions_label)


        self.chessboard = chess.Board()
        self.previous_state_callback = None

        self.update_chessboard()

        self.load_previous_button.clicked.connect(self.on_load_previous_state)


    def update_chessboard(self, fen=None, best_move=None, grid_pieces=None):
        if fen:
            self.chessboard = chess.Board(fen)

        if best_move == "Game Over":
            return

        arrows = []
        if best_move:
            try:
                move_str = str(best_move)
                start_square = chess.parse_square(move_str[:2])
                end_square = chess.parse_square(move_str[2:4])
                arrows = [(start_square, end_square)]
            except:
                pass

        svg = chess.svg.board(self.chessboard, arrows=arrows)
        self.widgetSvg.load(svg.encode("utf-8"))

        if grid_pieces:
            piece_texts = []
            for i, row in enumerate(grid_pieces):
                for j, piece in enumerate(row):
                    if piece != "":
                        square = chr(ord('a') + j) + str(8 - i)
                        piece_texts.append(f"{piece:>2} @ {square}")
            piece_text_display = "\n".join(piece_texts)
            self.piece_positions_label.setText("Taş Konumları:\n" + piece_text_display)


        self.label.setText(f"Best Move: {best_move}")

    def update_best_move(self, move):
        if move == "Game Over":
            self.label.setText("Game Over")
        else:
            self.label.setText(f"Best Move: {move}")
            # self.update_chessboard(best_move=move)
        
    def on_load_previous_state(self):
        if self.previous_state_callback:
            self.previous_state_callback()