# ::IMPORTS -------------------------------------------------------------------------- #
# cross platform path handling - https://docs.python.org/3/library/pathlib.html
from pathlib import Path

# image processing - https://opencv.org/
import cv2

# command line interface - https://typer.tiangolo.com/
import typer

# file system utilities - https://docs.python.org/3/library/os.html
import os

# numpy array manipulation - https://numpy.org/doc/stable/index.html
import numpy as np


# ::UTIL METHODS -------------------------------------------------------------------------- #
def detect_squares_manually(bin_img, min_side_len, max_side_len, side_tolerance=2):
    """
    Detect squares in a binarized (black and white) image by scanning
    horizontal & vertical pixel runs. Returns a list of squares, where each
    square is a tuple of four corner coordinates:
      (top_left, top_right, bottom_left, bottom_right)
    and each corner is (row, col).
    """
    height, width = bin_img.shape
    
    def horizontal_run_length(row, col):
        """Return how many consecutive 'black' pixels (==255) to the right
           of (row, col), including (row, col)."""
        run = 0
        c = col
        while c < width and bin_img[row, c] == 255:
            run += 1
            c += 1
        return run

    def vertical_run_length(row, col):
        """Return how many consecutive 'black' pixels (==255) downward
           from (row, col), including (row, col)."""
        run = 0
        r = row
        while r < height and bin_img[r, col] == 255:
            run += 1
            r += 1
        return run

    squares = []
    
    # Scan all pixels as potential top-left corners of squares
    for x in range(height):
        for y in range(width):
            if bin_img[x, y] == 255:
                # Quick neighbor check: must have black to the right & below
                if (y + 1 < width and bin_img[x, y + 1] == 255) and \
                   (x + 1 < height and bin_img[x + 1, y] == 255):

                    # Measure horizontal & vertical runs
                    lenX = horizontal_run_length(x, y)  # top edge
                    lenY = vertical_run_length(x, y)    # left edge

                    # Check side length constraints
                    if (min_side_len <= lenX <= max_side_len and
                        min_side_len <= lenY <= max_side_len):
                        
                        # Potential top-left corner of a square
                        xTR = x
                        yTR = y + lenX - 1
                        xBL = x + lenY - 1
                        yBL = y

                        if xTR < height and yTR < width and \
                           xBL < height and yBL < width:
                            # Check the other edges
                            vertical_run_TR = vertical_run_length(xTR, yTR)
                            horizontal_run_BL = horizontal_run_length(xBL, yBL)

                            # Ensure lenX ~ lenY
                            if (abs(lenX - lenY) <= side_tolerance and
                                vertical_run_TR >= lenY and
                                horizontal_run_BL >= lenX):
                                
                                # Store the square corners
                                top_left = (x, y)
                                top_right = (xTR, yTR)
                                bottom_left = (xBL, yBL)
                                bottom_right = (xBL, yTR)
                                squares.append((top_left, top_right, 
                                                bottom_left, bottom_right))
    return squares


def canonical_rect(square):
    """
    Given a square as ((row_tl, col_tl), (row_tr, col_tr),
                       (row_bl, col_bl), (row_br, col_br)),
    return a canonical bounding rectangle (row_min, col_min, row_max, col_max).
    """
    (tl, tr, bl, br) = square
    rows = [tl[0], tr[0], bl[0], br[0]]
    cols = [tl[1], tr[1], bl[1], br[1]]

    row_min, row_max = min(rows), max(rows)
    col_min, col_max = min(cols), max(cols)
    return (row_min, col_min, row_max, col_max)


def rect_distance(r1, r2):
    """
    A simple "distance" metric between two rectangles (row_min, col_min, row_max, col_max).
    We sum the absolute differences of each coordinate.
    """
    return abs(r1[0] - r2[0]) + abs(r1[1] - r2[1]) + \
           abs(r1[2] - r2[2]) + abs(r1[3] - r2[3])


def filter_overlapping_squares(squares, overlap_threshold=10):
    """
    Given a list of squares (each as 4 corner coords),
    remove duplicates/overlaps by comparing their canonical bounding boxes.
    If two squares have bounding boxes within 'overlap_threshold' distance,
    we treat them as duplicates and keep only the first one found.
    """
    final_squares = []
    for sq in squares:
        r_sq = canonical_rect(sq)

        # Check if this new rect is close to any rect in final_squares
        is_duplicate = False
        for fsq in final_squares:
            r_fsq = canonical_rect(fsq)
            if rect_distance(r_sq, r_fsq) < overlap_threshold:
                # They're almost the same bounding box => treat as duplicate
                is_duplicate = True
                break
        
        if not is_duplicate:
            final_squares.append(sq)
    
    return final_squares

# ::CLI SETUP -------------------------------------------------------------------------- #
app = typer.Typer(
    add_completion=False,
    no_args_is_help=True
    )

# ::CLI COMMANDS -------------------------------------------------------------------------- #
@app.command()
def extract(
    path: Path,
    min_side_len: int = 15,
    max_side_len: int = 100,
    threshold: int = 150,
    side_tolerance: int = 2,
):
    # 1. Load image & convert to grayscale
    original = cv2.imread(path)
    if original is None:
        print(f"Could not open {path}")
        return
    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)

    # 2. Binarize so that black lines => 255, background => 0 (adjust if needed)
    _, bin_img = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)


    # 3. Detect squares (raw, may have duplicates)
    all_squares = detect_squares_manually(bin_img, min_side_len, max_side_len, side_tolerance)
    print("Raw squares detected:", len(all_squares))

    # 4. Remove overlapping / duplicate squares
    # Increase or decrease overlap_threshold depending on how strict you want to be.
    cleaned_squares = filter_overlapping_squares(all_squares, overlap_threshold=10)
    print("Filtered squares:", len(cleaned_squares))

    # 5. Annotate the final squares
    annotated = original.copy()
    for sq in cleaned_squares:
        (tl, tr, bl, br) = sq
        # Remember each corner is (row, col). 
        # For cv2 drawing, we pass (col, row) as (x,y).
        row_tl, col_tl = tl
        row_br, col_br = br  # bottom-right

        pt1 = (col_tl, row_tl)
        pt2 = (col_br, row_br)
        cv2.rectangle(annotated, pt1, pt2, (255, 0, 0), 2)

        cv2.putText(
            annotated, 
            "square",               # "circle" or "square"
            (col_tl, row_tl - 5),             # text position slightly above top-left
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.5, 
            (255, 0, 0), 
            1
        )

    # 6. Display the annotated result
    cv2.imshow("Detected Squares", annotated)
    cv2.imwrite("annotated.png", annotated)  # save the result

    cv2.waitKey(0)
    cv2.destroyAllWindows()



# ::EXECUTE ------------------------------------------------------------------------ #
def main():
    app()


if __name__ == "__main__":  # ensure importing the script will not execute
    main()