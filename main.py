import cv2
import numpy as np
import chess
import chess.svg
from ultralytics import YOLO
import chess.engine
from chessGUI import ChessBoard
from PyQt5.QtWidgets import QApplication
import sys
import requests

class ChessPlayerHelper:
    def __init__(self, model_path, corner_path, gui):
        self.piece_model = YOLO(model_path) # YOLOv8 model for detecting chess pieces
        self.corners_model = YOLO(corner_path) # YOLOv8 model for detecting chessboard corners
        self.board_corners = None # Store the detected chessboard corners
        self.grid_positions = None # Store the computed grid positions
        self.gui = gui # GUI for displaying the chessboard
        self.valid_fens = [] # History of game states
        self.piece_counts = [] # Store the piece counts for each frame
        self.board = chess.Board() # Initialize a chess board
        self.previous_piece_counts =  {
            'P': 8, 'N': 2, 'B': 2, 'R': 2, 'Q': 1, 'K': 1,
            'p': 8, 'n': 2, 'b': 2, 'r': 2, 'q': 1, 'k': 1
        }

        self.gui.previous_state_callback = self.load_previous_fen # Callback to load previous fen

    def detect_corners(self, frame):
        """
        Use the corner detection model to detect chessboard corners.

        Args:
            frame (np.array): Input frame from the camera.

        Returns:
            np.array: Detected corners if found, otherwise None.
        """
        results = self.corners_model.predict(frame)
        if not results[0].boxes:
            return None
        box = results[0].boxes
        arr = box.xywh.numpy()
        points = arr[:, 0:2]

        if len(points) < 4:
            return None

        corners = self.order_points(points)

        return corners
    
    def order_points(self, pts):
        """
        Order the points in the top-left, top-right, bottom-right, and bottom-left order.

        Args:
            pts (np.array): Array of points to order.

        Returns:
            np.array: Ordered points
        """
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect
    
    def four_point_transform(self, frame, corners, padding=85):
        """
        Perform a perspective transform to get a top-down view of the chessboard with proper padding.

        Args:
            frame (np.array): Input frame from the camera.
            corners (np.array): Detected corners of the chessboard.
            padding (int): Padding to add around the chessboard.

        Returns:
            np.array: Warped image of the chessboard.
            np.array: Transformation matrix.
        """
        # Order the corners to get top-left, top-right, bottom-right, and bottom-left
        rect = self.order_points(corners)
        (tl, tr, br, bl) = rect

        # Calculate the width and height of the warped image
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))

        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))

        # Add padding to each corner point
        tl[0] -= padding
        tl[1] -= padding
        tr[0] += padding
        tr[1] -= padding
        br[0] += padding
        br[1] += padding
        bl[0] -= padding
        bl[1] += padding

        # Define the destination points with added padding
        dst = np.array([
            [0, 0],
            [maxWidth - 1 + 2 * padding, 0],
            [maxWidth - 1 + 2 * padding, maxHeight - 1 + 2 * padding],
            [0, maxHeight - 1 + 2 * padding]
        ], dtype="float32")

        # Compute the perspective transform matrix
        M = cv2.getPerspectiveTransform(rect, dst)

        # Perform the warp perspective transformation
        warped = cv2.warpPerspective(frame, M, (int(maxWidth + 2 * padding), int(maxHeight + 2 * padding)))

        return warped, M

    def transform_corners_on_warped(self, warped_image, original_corners, transformation_matrix):
        """
        Transform the original corners using the transformation matrix.

        Args:
            warped_image (np.array): Warped chessboard image.
            original_corners (list): Detected corners of the chessboard.
            transformation_matrix (np.array): Transformation matrix.

        Returns:
            np.array: Transformed corners.
        """
        # Transform the original corners using the transformation matrix
        original_corners_homogeneous = np.hstack([original_corners, np.ones((4, 1))])  # Add the homogeneous coordinate
        transformed_corners = transformation_matrix.dot(original_corners_homogeneous.T).T
        transformed_corners = transformed_corners[:, :2] / transformed_corners[:, 2:]  # Convert from homogeneous coordinates

        transformed_corners[0][0] -= 4
        transformed_corners[0][1] -= 3

        # Draw the transformed corners on the warped image
        # debug_image = warped_image.copy()
        # for i, corner in enumerate(transformed_corners):
        #     x, y = int(corner[0]), int(corner[1])
        #     cv2.circle(debug_image, (x, y), 5, (0, 0, 255), -1)  # Red circles for corners
        #     cv2.putText(debug_image, f"Corner {i}: ({x}, {y})", (x + 5, y - 5),
        #                 cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
            
        # # # Show the image with transformed corners
        # cv2.imshow("Warped Image with Transformed Corners", debug_image)
        # cv2.waitKey(1)

        return transformed_corners
    
    def resize_warped_image_and_corners(self, warped_image, transformed_corners, target_size=(800, 800)):
        """
        Resize the warped image and scale the transformed corners proportionally.

        Args:
            warped_image (np.array): The warped chessboard image.
            transformed_corners (np.array): Transformed corners in the warped image.
            target_size (tuple): Target dimensions (width, height) for resizing.

        Returns:
            np.array: Resized warped image.
            np.array: Adjusted transformed corners.
        """
        original_height, original_width = warped_image.shape[:2]
        target_width, target_height = target_size

        # Resize the warped image to the target size
        resized_image = cv2.resize(warped_image, (target_width, target_height))

        # Calculate scaling factors for width and height
        scale_x = target_width / original_width
        scale_y = target_height / original_height

        # Adjust the corners based on the scaling factors
        scaled_corners = np.copy(transformed_corners)
        scaled_corners[:, 0] *= scale_x  # Scale x-coordinates
        scaled_corners[:, 1] *= scale_y  # Scale y-coordinates


        # Print the scaled corners at the frame
        # debug_image = resized_image.copy()
        # for i, corner in enumerate(scaled_corners):
        #     x, y = int(corner[0]), int(corner[1])
        #     cv2.circle(debug_image, (x, y), 5, (0, 0, 255), -1)
        #     cv2.putText(debug_image, f"Corner {i}: ({x}, {y})", (x + 5, y - 5),
        #                 cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
            
        # cv2.imshow("Resized Warped Image with Scaled Corners", debug_image)
        # cv2.waitKey(1)
        
        return resized_image, scaled_corners

    def get_grid_positions(self, warped_image, corners):
        """
        Compute the 8x8 grid positions for the chessboard using all four corners.

        Args:
            warped_image (np.array): Resized warped chessboard image.
            corners (np.array): Array of four corners (top-left, top-right, bottom-right, bottom-left).

        Returns:
            list: Grid positions for the 8x8 chessboard.
        """
        # Extract the corners
        top_left, top_right, bottom_right, bottom_left = corners

        # Initialize grid positions
        grid_positions = []

        # Loop through rows and columns to compute square positions
        for row in range(8):
            for col in range(8):
                # Interpolate along the edges for the current square
                row_start = top_left + (bottom_left - top_left) * (row / 8)
                row_end = top_right + (bottom_right - top_right) * (row / 8)
                next_row_start = top_left + (bottom_left - top_left) * ((row + 1) / 8)
                next_row_end = top_right + (bottom_right - top_right) * ((row + 1) / 8)

                square_top_left = row_start + (row_end - row_start) * (col / 8)
                square_top_right = row_start + (row_end - row_start) * ((col + 1) / 8)
                square_bottom_left = next_row_start + (next_row_end - next_row_start) * (col / 8)
                square_bottom_right = next_row_start + (next_row_end - next_row_start) * ((col + 1) / 8)

                # Append the top-left and bottom-right corners of the square
                grid_positions.append((
                    tuple(square_top_left.astype(int)),
                    tuple(square_bottom_right.astype(int))
                ))

        # Debugging: Visualize the grid on the warped image
        # debug_image = warped_image.copy()
        # for top_left, bottom_right in grid_positions:
        #     cv2.rectangle(debug_image, top_left, bottom_right, (0, 255, 0), 1)

        # cv2.imshow("Gridded Chessboard", debug_image)
        # cv2.waitKey(1)

        self.grid_positions = grid_positions
        return grid_positions


    def detect_pieces(self, board):
        """
        Detect chess pieces using YOLOv8 on the aligned chessboard.
        Returns a list of detected pieces and their bounding boxes.

        """

        resized_board = cv2.resize(board, (800, 800))

        results = self.piece_model.predict(resized_board, iou=0.45)
        pieces = []

        if not results[0].boxes:
            return pieces

        piece_counts = {
            'P': 0, 'N': 0, 'B': 0, 'R': 0, 'Q': 0, 'K': 0,
            'p': 0, 'n': 0, 'b': 0, 'r': 0, 'q': 0, 'k': 0
        }

        max_piece_counts = {
            'P': 8, 'N': 2, 'B': 2, 'R': 2, 'Q': 1, 'K': 1,
            'p': 8, 'n': 2, 'b': 2, 'r': 2, 'q': 1, 'k': 1
        }

        for box in results[0].boxes:
            if hasattr(box, 'xyxy') and len(box.xyxy[0]) == 4:
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
                conf = float(box.conf[0]) if hasattr(box, 'conf') else None
                cls = int(box.cls[0]) if hasattr(box, 'cls') else None

                if conf is not None and cls is not None and conf > 0.5:
                    label = self.map_class_to_piece(cls)
                    if piece_counts[label] < max_piece_counts[label]:
                        pieces.append({
                            "bbox": (x1, y1, x2, y2),
                            "label": label,
                            "confidence": conf
                        })
                        piece_counts[label] += 1
                    else:
                        print(f"Skipping {label} due to max count reached.")


        # Compare witn previous piece counts
        for label in piece_counts:
            if piece_counts[label] > self.previous_piece_counts[label]:
                print(f"Detected more {label} pieces than before: {piece_counts[label]} vs {self.previous_piece_counts[label]}")
                excess_count = piece_counts[label] - self.previous_piece_counts[label]
                pieces = [piece for piece in pieces if piece['label'] != label or excess_count <= 0 or (excess_count := excess_count -1)]

        self.previous_piece_counts = piece_counts

        # Debugging: Print detected pieces
        # debug_frame = resized_board.copy()
        # for piece in pieces:
        #     x1, y1, x2, y2 = piece['bbox']
        #     cv2.rectangle(debug_frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
        #     cv2.putText(debug_frame, f"{piece['label']} ({piece['confidence']:.2f})", (x1, y1 - 5),
        #                 cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
            
        # cv2.imshow("Detected Pieces", debug_frame)
        # cv2.waitKey(1)

        return pieces

    def map_class_to_piece(self, cls):
        """
        Map YOLOv8 class index to chess piece labels.

        Args:
            cls (int): YOLOv8 class index.

        Returns:
            str: Chess piece label.
        """
        class_to_piece = {
            0: 'B',  # White bishop
            1: 'K',  # White king
            2: 'N',  # White knight
            3: 'P',  # White pawn
            4: 'Q',  # White queen
            5: 'R',  # White rook
            6: 'b',  # Black bishop
            7: 'k',  # Black king
            8: 'n',  # Black knight
            9: 'p',  # Black pawn
            10: 'q',  # Black queen
            11: 'r'   # Black rook
        }

        return class_to_piece.get(cls, '?')

    def map_pieces_to_grid(self, detected_pieces):
        """
        Map detected pieces to grid positions using self.grid_positions and their bottom coordinates.

        Args:
            detected_pieces (list): List of detected pieces with bounding boxes.

        Returns:
            list: 2D list representing the chessboard with piece labels.
        """
        grid_size = 8
        grid_pieces = [["" for _ in range(grid_size)] for _ in range(grid_size)]

        # Iterate over detected pieces
        for piece in detected_pieces:
            x1, y1, x2, y2 = piece['bbox']
            piece_bottom_x, piece_bottom_y = (x1 + x2) // 2, y2  # Bottom-center of the bounding box

            # Find the square the piece belongs to
            for row in range(grid_size):
                for col in range(grid_size):
                    top_left, bottom_right = self.grid_positions[row * grid_size + col]
                    if top_left[0] <= piece_bottom_x <= bottom_right[0] and top_left[1] <= piece_bottom_y <= bottom_right[1]:
                        grid_pieces[row][col] = piece['label']
                        break

        # Debugging: Print the mapped grid
        # print("Mapped Grid:")
        # for row in grid_pieces:
        #     print(row)

        return grid_pieces
    
    def count_pieces(self, grid_pieces):
        """
        Count the number of pieces on the chessboard.
        """

        return sum(1 for row in grid_pieces for cell in row if cell != "")

    def generate_fen(self, grid_pieces):
        fen_rows = []

        for row in grid_pieces:
            empty_count = 0
            fen_row = ""
            for cell in row:
                if cell == "":
                    empty_count += 1
                else:
                    if empty_count > 0:
                        fen_row += str(empty_count)
                        empty_count = 0
                    fen_row += cell
            if empty_count > 0:
                fen_row += str(empty_count)
            fen_rows.append(fen_row)

        fen_board = "/".join(fen_rows)

        last_fen = self.valid_fens[-1] if self.valid_fens else None
        if last_fen:
            if fen_board == last_fen.split()[0]:
                return last_fen
            last_color = last_fen.split()[1]
            active_color = "b" if last_color == "w" else "w"
        else:
            active_color = "w"

        # 🛠 Bu kısmı güncelledik:
        if "k" in fen_board and "r" in fen_board:
            castling_rights = "KQkq"
        else:
            castling_rights = "-"

        en_passant = "-"
        halfmove_clock = "0"
        fullmove_number = "1"

        return f"{fen_board} {active_color} {castling_rights} {en_passant} {halfmove_clock} {fullmove_number}"
    
    def validate_fen(self, new_fen, piece_count):
        """
        Validate the new FEN based on the previous valid FEN.
        Checks include piece consistency and legal moves.
        """
        try:
            # Create board objects from FEN
            new_board = chess.Board(new_fen)

            if len(self.valid_fens) == 0:
                # No prior FENs, accept the new FEN
                return True

            last_fen = self.valid_fens[-1]
            last_board = chess.Board(last_fen)

            print("Last FEN:", last_fen)
            print("New FEN:", new_fen)

            # Validate active color alternation
            if last_board.turn == new_board.turn:
                print("Turn did not alternate correctly.")
                return False
            
            last_piece_count = self.piece_counts[-1]
            if abs(piece_count - last_piece_count) > 2:
                print(f"Piece count difference is too high: {piece_count} vs {last_piece_count}")
                return False

            return True

        except ValueError as e:
            print(f"FEN validation error: {e}")
            return False
        
    def check_turn(self, fen):
        """
        Check the turn based on the FEN string.
        """
        board = chess.Board(fen)
        if board.turn == chess.WHITE:
            return "White"
        else:
            return "Black"

    def update_valid_fen(self, new_fen):
        """
        Update the list of valid FENs.
        """

        if self.validate_fen(new_fen):
            self.valid_fens.append(new_fen)
            print("FEN validated and updated.")
            return new_fen
        else:
            # print("FEN validation failed.")
            return self.valid_fens[-1]
        
    def get_best_move(self, fen):
        """
        Use Stockfish API to calculate the best move and check for game-over states.
        Args:
            fen (str): The FEN string of the current board state.
        Returns:
            str: Best move in UCI notation or a game-over message.
        """
        try:
            self.board = chess.Board(fen)

            if self.board.is_checkmate():
                print("[INFO] Game over: Checkmate detected!")
                return "Mate"
            elif self.board.is_stalemate():
                print("[INFO] Game over: Stalemate detected!")
                return "Mate"

            url = "https://stockfish.online/api/s/v2.php"
            params = {"fen": fen, "depth": 15}
            response = requests.get(url, params=params)
            response.raise_for_status()

            data = response.json()
            best_move_str = data.get("bestmove")
            if not best_move_str:
                print("[WARNING] No valid bestmove provided by Stockfish.")
                return None

            tokens = best_move_str.strip().split()
            # Eğer UCI çıktı “bestmove e2e4 ponder e7e5” gibiyse ikinci elemanı al
            if tokens[0].lower() == "bestmove" and len(tokens) > 1:
                move = tokens[1]
            else:
                # aksi halde ilk (veya tek) token zaten hamle
                move = tokens[0]

            print(f"[INFO] Best move calculated: {move}")
            return move

        except requests.RequestException as e:
            print(f"[ERROR] Error with Stockfish API: {e}")
            return None
    
    def load_previous_fen(self):
        if len(self.valid_fens) > 1:
            # Şu anki FEN'i çıkar (en son yapılan hamle)
            self.valid_fens.pop()
            self.piece_counts.pop()

            # Bir önceki FEN'i al
            previous_fen = self.valid_fens[-1]
            previous_count = self.piece_counts[-1]

            self.board = chess.Board(previous_fen)
            best_move = self.get_best_move(previous_fen)
            self.gui.update_chessboard(previous_fen, best_move)
            self.gui.update_best_move(best_move)

            print("Önceki FEN yüklendi:", previous_fen)
        else:
            print("Geri dönecek başka FEN bulunamadı.")

    def run(self, process_every_n_frames=3):
        cap = cv2.VideoCapture(1)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        frame_count = 0

        self.board_corners = None  # başlangıçta null

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            key = cv2.waitKey(1) & 0xFF

            # Kullanıcı köşeleri bulmak için 'c'ye bastıysa
            if key == ord("c"):
                chessboard_corners = self.detect_corners(frame)
                if chessboard_corners is not None:
                    self.board_corners = chessboard_corners
                    print("Corners detected and stored.")
                    # Sadece burada FEN üretimi ve hamle yapılır
                    warped, M = self.four_point_transform(frame, self.board_corners)
                    transformed_corners = self.transform_corners_on_warped(warped, self.board_corners, M)
                    resized_warped, resized_corners = self.resize_warped_image_and_corners(warped, transformed_corners)
                    grid_positions = self.get_grid_positions(resized_warped, resized_corners)
                    pieces = self.detect_pieces(resized_warped)
                    grid_pieces = self.map_pieces_to_grid(pieces)
                    piece_count = self.count_pieces(grid_pieces)
                    current_fen = self.generate_fen(grid_pieces)

                    if not self.is_valid_fen_board(current_fen):
                        print("FEN geçersiz.")
                        continue

                    if self.validate_fen(current_fen, piece_count):
                        self.valid_fens.append(current_fen)
                        self.piece_counts.append(piece_count)
                        self.board = chess.Board(current_fen)

                        if self.board.is_checkmate():
                            self.gui.update_chessboard(current_fen)
                            self.gui.update_best_move("Checkmate")
                            continue  # break yerine devam

                        turn = self.check_turn(current_fen)
                        if turn == "White":
                            best_move = self.get_best_move(current_fen)
                            if best_move == "Mate":
                                self.gui.update_chessboard(current_fen)
                                self.gui.update_best_move("Checkmate")
                            else:
                                self.gui.update_chessboard(current_fen, best_move, grid_pieces)
                                self.gui.update_best_move(best_move)
                        else:
                            self.gui.update_chessboard(current_fen)
                            self.gui.update_best_move("Black Move")
                    else:
                        print("FEN doğrulanamadı.")
                else:
                    print("Köşe bulunamadı. Lütfen tekrar deneyin.")

            elif key == ord("q"):
                break

            else:
                # Sadece kullanıcı başka tuşa basana kadar kamerayı göster
                cv2.imshow("Frame", frame)

            frame_count += 1

        cap.release()
        cv2.destroyAllWindows()
        # self.engine.quit()
    def is_valid_fen_board(self, fen):
        try:
            board_part = fen.split()[0]
            rows = board_part.split("/")
            if len(rows) != 8:
                return False
            for row in rows:
                count = 0
                for ch in row:
                    if ch.isdigit():
                        count += int(ch)
                    else:
                        count += 1
                if count != 8:
                    return False
            return True
        except:
            return False
    def process_image(self, image_path):
        """
        Tek bir görüntü üzerinden satranç tahtasını analiz eder.
        Args:
            image_path (str): Görüntü dosyasının yolu.
        """
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"Could not read image from {image_path}")
            return

        attempts = 3
        for attempt in range(attempts):
            print(f"[INFO] Processing attempt {attempt + 1}")
            chessboard_corners = self.detect_corners(frame)
            if chessboard_corners is None:
                print("Köşe tespiti başarısız.")
                continue

            warped, M = self.four_point_transform(frame, chessboard_corners)
            if warped is None:
                print("Perspektif düzeltme başarısız.")
                continue

            transformed_corners = self.transform_corners_on_warped(warped, chessboard_corners, M)
            resized_warped, resized_corners = self.resize_warped_image_and_corners(warped, transformed_corners)
            self.get_grid_positions(resized_warped, resized_corners)

            pieces = self.detect_pieces(resized_warped)
            grid_pieces = self.map_pieces_to_grid(pieces)
            piece_count = self.count_pieces(grid_pieces)
            fen = self.generate_fen(grid_pieces)

            if self.validate_fen(fen, piece_count):
                self.valid_fens.append(fen)
                self.piece_counts.append(piece_count)
                best_move = self.get_best_move(fen)
                self.gui.update_chessboard(fen, best_move, grid_pieces)
                self.gui.update_best_move(best_move)
                return
            else:
                print("FEN doğrulaması başarısız.")
        
        print("Tüm denemeler başarısız.")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = ChessBoard()
    gui.show()

    chesshelper = ChessPlayerHelper("models/best.pt", "models/best_corners.pt", gui)
    chesshelper.run()  # Start the main loop to process the chessboard from the camera feed
    #chesshelper.process_image(r"C:\Users\user\Downloads\bitirme\ChessPlayerHelper-master\ChessPlayerHelper-master\image\image5.jpg")  # Process a single image for testing

    sys.exit(app.exec_())
