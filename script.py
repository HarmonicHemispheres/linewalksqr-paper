# ----------------------- IMPORTS
import cv2
import numpy as np
import os
import json

# ----------------------- GLOBALS

# ----------------------- UTIL METHODS
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


# ----------------------- PUBLIC METHODS
def extract_square_coords(image, bounding_box_scale=8) -> list:
    """
    Return a list of 4 corner coordinates for each detected square.
    """

    # -- variables
    detections = []

    # -- Setting up Variables
    bounding_box_scale = 8
    min_side_len = 50
    max_side_len = 200
    side_tolerance = 2

    # -- Preprocessing steps

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, bin_img = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)
    all_squares = detect_squares_manually(bin_img, min_side_len, max_side_len, side_tolerance)
    print("Raw squares detected:", len(all_squares))

    # -- Remove overlapping / duplicate squares
    # Increase or decrease overlap_threshold depending on how strict you want to be.
    cleaned_squares = filter_overlapping_squares(all_squares, overlap_threshold=10)
    print("Filtered squares:", len(cleaned_squares))

    # -- Annotate the final squares
    for sq in cleaned_squares:
        (tl, tr, bl, br) = sq
        # Remember each corner is (row, col). 
        # For cv2 drawing, we pass (col, row) as (x,y).
        row_tl, col_tl = tl
        row_br, col_br = br  # bottom-right

        pt1 = (col_tl, row_tl)
        pt2 = (col_br, row_br)
        detections.append({
            "type": "square",
            "bbox": [col_tl-bounding_box_scale, 
                    row_tl-bounding_box_scale,
                    col_br+bounding_box_scale,
                    row_br+bounding_box_scale],
        })
    
    return detections
