import os
import random

import cv2
import numpy as np
from numpy.f2py.auxfuncs import throw_error
from sklearn.cluster import DBSCAN
import pytesseract
import networkx as nx
import matplotlib.pyplot as plt
import pprint
import Levenshtein
from fuzzywuzzy import fuzz

img_input_base_path = 'resources/images/input/'
img_output_base_path = 'resources/images/output/'
imgPaths = [
    'word_search_1.png',
    'word_search_2.png',
    'word_search_3.png',
    'word_search_4.png',
    'word_search_5.png',
    'word_search_6.png',
    'word_search_7.png',
    'word_search_8.png',
]

"""Preprocess the image to improve OCR accuracy (necessary for live feed)."""
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Image not found or unable to load.")

    # grayscale & thresholding
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU) #tresh 128?

    # remove noise
    kernel = np.ones((1, 1), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # Deskew
    """coords = np.column_stack(np.where(binary > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = binary.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    binary = cv2.warpAffine(binary, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT,
                            borderValue=(255, 255, 255))"""

    #cv2.imshow('Original Image', img)
    #cv2.imshow('Binary Image', binary)
    #cv2.waitKey(0)
    return img, binary

""" return (letters, words) """
def detect_text_regions(binary):
    """Detect potential text regions using contours."""
    #contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #edges = cv2.Canny(binary, 50, 150)
    edges = cv2.Canny(binary, 128, 128)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #print(len(contours))

    scale_factor = 2
    boxed_regions = [
        (x + (w - int(h * scale_factor)) // 2, y + (h - int(h * scale_factor)) // 2, int(h * scale_factor), int(h * scale_factor))
        for cnt in contours
        for x, y, w, h in [cv2.boundingRect(cnt)]
    ]


    letters = set()
    words = []

    merged_regions = merge_intersecting_regions(boxed_regions)

    for group in merged_regions:
        if len(group) > 1:
            words.append(group)
        else:
            letters.update(group)

    #letters = prune_outliers(letters)

    word_bounding_boxes = [
        (
            min(region[0] for region in word_group),
            min(region[1] for region in word_group),
            max(region[0] + region[2] for region in word_group) - min(region[0] for region in word_group),
            max(region[1] + region[3] for region in word_group) - min(region[1] for region in word_group)
        )
        for word_group in words
    ]

    #print({h for x, y, w, h in letters})

    print(f"Letters: {len(letters)}, Words: {len(word_bounding_boxes)}")

    output_picture = binary.copy()
    #cv2.imshow('binary with regions', output_picture)
    #cv2.waitKey(0)

    # Draw rectangles around letters
    for (x, y, w, h) in letters:
        cv2.rectangle(output_picture, (x, y), (x + w, y + h), (255, 255, 255), 2)

    # Draw rectangles around words
    for (x, y, w, h) in word_bounding_boxes:
        cv2.rectangle(output_picture, (x, y), (x + w, y + h), (255, 255, 255), 2)

    # Save the resulting image
    cv2.imwrite(img_output_base_path + "regions/out.png", output_picture)

    #cv2.imshow('binary with regions', output_picture)
    #cv2.waitKey(0)

    return letters, word_bounding_boxes

"""Prune outlier regions based on size using IQR - Interquartile Range."""
def prune_outliers(regions):

    if not regions:
        return regions

    widths = np.array([w for _, _, w, _ in regions])
    heights = np.array([h for _, _, _, h in regions])

    # IQR for widths and heights
    q1_w, q3_w = np.percentile(widths, [25, 75])
    iqr_w = q3_w - q1_w
    lower_bound_w = q1_w - 1.5 * iqr_w
    upper_bound_w = q3_w + 1.5 * iqr_w

    q1_h, q3_h = np.percentile(heights, [25, 75])
    iqr_h = q3_h - q1_h
    lower_bound_h = q1_h - 1.5 * iqr_h
    upper_bound_h = q3_h + 1.5 * iqr_h

    pruned_regions = [
        region for region in regions
        if lower_bound_w <= region[2] <= upper_bound_w and lower_bound_h <= region[3] <= upper_bound_h
    ]

    return pruned_regions

def are_regions_intersecting(region1, region2):
    x1, y1, w1, h1 = region1
    x2, y2, w2, h2 = region2

    return not (x1 > x2 + w2 or x1 + w1 < x2 or y1 > y2 + h2 or y1 + h1 < y2)

"""Find the root of the set in which element i is."""
def find(parent, i):

    if parent[i] == i:
        return i
    else:
        parent[i] = find(parent, parent[i])
        return parent[i]

"""Union of two sets of x and y."""
def union(parent, rank, x, y):
    root_x = find(parent, x)
    root_y = find(parent, y)

    if root_x != root_y:
        if rank[root_x] > rank[root_y]:
            parent[root_y] = root_x
        elif rank[root_x] < rank[root_y]:
            parent[root_x] = root_y
        else:
            parent[root_y] = root_x
            rank[root_x] += 1

"""Merge intersecting regions into groups."""
def merge_intersecting_regions(regions):

    n = len(regions)
    parent = list(range(n))
    rank = [0] * n

    for i in range(n):
        for j in range(i + 1, n):
            if are_regions_intersecting(regions[i], regions[j]):
                union(parent, rank, i, j)

    groups = {}
    for i in range(n):
        root = find(parent, i)
        if root not in groups:
            groups[root] = set()
        groups[root].add(regions[i])

    return list(groups.values())

def cluster_regions(img, regions, axis, eps=10):
    """Cluster regions based on their positions along a specified axis."""
    positions = np.array([[region[axis]] for region in regions])
    clustering = DBSCAN(eps=eps, min_samples=1).fit(positions)
    labels = clustering.labels_

    #visualize_clusters(img, regions, labels, img_output_base_path + "clusters.png")

    clustered_regions = []
    for label in np.unique(labels):
        clustered_regions.append([region for region, lbl in zip(regions, labels) if lbl == label])
    #print("cluster_regions ->", len(clustered_regions))
    return clustered_regions

"""
Clusters text regions by spatial proximity and returns the regions in the largest cluster.
Todo: remove (the idea was to further cleanup the detected regions by removing outliers (ie: regions very far rom the rest)
"""
def get_largest_cluster(text_regions):
    if not text_regions:
        return np.array([])

    # centers of bounding boxes
    centers = np.array([(x + w / 2, y + h / 2) for x, y, w, h in text_regions])


    avg_width = np.mean([w for _, _, w, _ in text_regions])
    eps = avg_width * 2

    clustering = DBSCAN(eps=eps, min_samples=5).fit(centers)
    labels = clustering.labels_

    # Find the largest cluster
    unique_labels, counts = np.unique(labels, return_counts=True)
    largest_cluster_label = unique_labels[np.argmax(counts)]

    # Filter regions belonging to the largest cluster
    largest_cluster_regions = [region for region, label in zip(text_regions, labels) if label == largest_cluster_label]

    return largest_cluster_regions

def visualize_clusters(img, regions, labels, output_path):
    """Visualize clusters by drawing bounding boxes with unique colors."""
    # BGR format
    color_palette = [
        (0, 255, 0),  # Green for grid
        (255, 0, 0),  # Blue for other clusters
        (0, 0, 255),  # Red for noise
        (0, 255, 255),
        (255, 0, 255),
        (255, 255, 0),
        (0, 0, 0),
        (90, 90, 0),
        (90, 90, 200),
        (90, 10, 150),
        (90, 170, 0),
        (90, 170, 200),
        (190, 170, 0),
    ]

    # Assign colors to cluster labels
    unique_labels = sorted(set(labels))
    print(f"{len(labels)} labels")
    print(f"{len(unique_labels)} unique labels")
    color_map = {
        label: color_palette[i % len(color_palette)] for i, label in enumerate(unique_labels) if label != -1
    }
    color_map[-1] = (0, 0, 255)  # Red for noise

    # Draw bounding boxes for each cluster
    for i, (x, y, w, h) in enumerate(regions):
        label = labels[i]
        color = color_map.get(label, (0, 0, 255))  # Default to red for noise
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)

    # Save the output image
    cv2.imwrite(output_path, img)
    print(f"Clusters visualized and saved to: {output_path}")

def extract_letters_from_grid_by_letter(img, letter_regions):
    #letter_regions = sorted(letter_regions, key=lambda r: (r[1], r[0]))
    rows = cluster_regions(img, letter_regions, axis=1)
    letter_grid = []

    for row in rows:
        letter_row = []
        row = sorted(row, key=lambda r: r[1])
        for x, y, w, h in row:
            letter_img = img[y:y + h, x:x + w]
            letter = pytesseract.image_to_string(letter_img, config='--psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ').strip()
            if letter == '':
                letter = '*'
            letter_row.append(letter)
        print(' '.join(letter_row))
        letter_grid.append(letter_row)

    return letter_grid

def extract_letters_from_grid_by_row(img, letter_regions):
    #letter_regions = sorted(letter_regions, key=lambda r: (r[1], r[0]))
    rows = cluster_regions(img, letter_regions, axis=1)
    letter_grid = []

    for row in rows:
        row = sorted(row, key=lambda r: r[1])
        x, y, w, h = generate_bounding_box(row)
        #row_img = img[y:y + h, x:x + w]
        #letter_row = pytesseract.image_to_string(row_img, config='--psm 10 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ').strip()
        letter_row = extract_words_from_regions(img, [(x, y, w, h)])
        letter_grid.append(letter_row)

    return letter_grid

def extract_letters_from_grid_by_letter(img, letter_regions):
    rows = cluster_regions(img, letter_regions, axis=1)
    letter_grid = []

    for row in rows:
        letter_row = []
        row = sorted(row, key=lambda r: r[1])
        for x, y, w, h in row:
            letter_img = img[y:y + h, x:x + w]
            letter = pytesseract.image_to_string(letter_img, config='--psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ').strip()
            if letter == '':
                letter = '*'
            letter_row.append(letter)
        print(' '.join(letter_row))
        letter_grid.append(letter_row)

    return letter_grid

def extract_letters_from_grid_by_row(img, letter_regions):
    rows = cluster_regions(img, letter_regions, axis=1)
    letter_grid = []

    for row in rows:
        row = sorted(row, key=lambda r: r[1])
        x, y, w, h = generate_bounding_box(row)
        letter_row = extract_words_from_regions(img, [(x, y, w, h)])
        letter_grid.append(letter_row)

    return letter_grid

"""Save each row region as an image (for verification)."""
def save_row_regions(img, letter_regions, output_dir, file_name):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    rows = cluster_regions(img, letter_regions, axis=1)

    if file_name is None:
        file_name = "row"

    for idx, row in enumerate(rows):
        row = sorted(row, key=lambda r: r[0])
        x, y, w, h = generate_bounding_box(row)
        row_img = img[y:y + h, x:x + w]

        row_img_path = os.path.join(output_dir, f'{file_name}_{idx + 1}.png')
        cv2.imwrite(row_img_path, row_img)
        #print(f'Saved row image: {row_img_path}')

"""Generates the coordinates of the box encompassing all input regions"""
def generate_bounding_box(regions):
    x_min = min(region[0] for region in regions)
    y_min = min(region[1] for region in regions)
    x_max = max(region[0] + region[2] for region in regions)
    y_max = max(region[1] + region[3] for region in regions)

    width = x_max - x_min
    height = y_max - y_min

    return (x_min, y_min, width, height)

"""Extract words from the image and return them as a list of strings."""
def extract_words_from_regions(img, word_regions):
    words = []

    for (x, y, w, h) in word_regions:
        word_img = img[y:y + h, x:x + w]
        word = pytesseract.image_to_string(word_img, config='--psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ').strip()
        words.append(word)

    return words

"""Pretty print the 2D grid of letters."""
def pretty_print_grid(grid):

    for row in grid:
        print(len(row), "-", row)
        #print(' '.join(row))

"""Visually cross out words inside the grid for all detected_words."""
def draw_detected_words(img, detected_words):

    for word_coords in detected_words:
        for i in range(len(word_coords) - 1):
            start_point = word_coords[i]
            end_point = word_coords[i + 1]
            cv2.line(img, start_point, end_point, (0, 0, 255), 2)  # Red lines for detected words

"""Draw bounding boxes for the grid, grid letters and word list."""
def visualize_results(output_filename, img, letter_regions, word_regions):

    grid = [(
        min(region[0] for region in letter_regions),
        min(region[1] for region in letter_regions),
        max(region[0] + region[2] for region in letter_regions) - min(region[0] for region in letter_regions),
        max(region[1] + region[3] for region in letter_regions) - min(region[1] for region in letter_regions)
    )]

    for x, y, w, h in letter_regions:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)  # Green for letters
    for x, y, w, h in word_regions:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 3)  # Blue for word list
    for x, y, w, h in grid:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2)  # Purple for word grid

    if output_filename is None:
        output_filename = "output.jpg"

    cv2.imwrite(img_output_base_path + output_filename, img)

def generate_random_color():
    return [random.randint(0, 255) for _ in range(3)]

def visualize_solved_puzzle(output_filename, img, letter_regions, word_regions, letter_grid, word_list, found_words):

    if output_filename is None:
        output_filename = "output.jpg"

    overlay = img.copy()
    alpha = 0.7  # Transparency factor
    line_thickness = 20

    i=0
    for word, coords in found_words.items():
        i += 1
        try:
            color = generate_random_color()
            line_start = coords[0]
            line_end = coords[-1]

            grid_region_start = letter_regions[line_start[0] * len(letter_grid[0]) + line_start[1]]
            grid_region_end = letter_regions[line_end[0] * len(letter_grid[0]) + line_end[1]]

            x1, y1, w1, h1 = grid_region_start
            x2, y2, w2, h2 = grid_region_end


            #print("-----")
            #print("letter_grid", "=>", len(letter_grid))
            #print("letter_grid[]", "=>", len(letter_grid[0]))
            #print(word, "=>", grid_region_start, "-", grid_region_end)
            #print("-----")

            save_row_regions(img, [grid_region_start, grid_region_end], img_output_base_path + "solved/details/" + output_filename + "-", str(i) + "_" + word)

            #cv2.line(img, (x1, y1), (x2 + w2, y2 + h2), (0, 0, 255), 2)

            x1_center = x1 + w1 // 2
            y1_center = y1 + h1 // 2
            x2_center = x2 + w2 // 2
            y2_center = y2 + h2 // 2
            cv2.line(img, (x1_center, y1_center), (x2_center, y2_center), color, line_thickness)

            word_list_index = word_list.index(word)
            word_region = word_regions[word_list_index]

            x, y, w, h = word_region
            y_center = y + h // 2
            #cv2.line(img, (x, y), (x + w, y + h), color, line_thickness)
            cv2.line(img, (x, y_center), (x + w, y_center), color, line_thickness)
        except Exception as e:
            print(f"Error processing word '{word}': {e}")

    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    cv2.imwrite(img_output_base_path + "solved/" + output_filename, img)

def find_word_in_grid_basic(grid, word):
    """Find a word in the grid and return the coordinates of the word."""
    directions = [
        (0, 1), (0, -1),  # Horizontal
        (1, 0), (-1, 0),  # Vertical
        (1, 1), (-1, -1),  # Diagonal top-left to bottom-right and bottom-right to top-left
        (1, -1), (-1, 1)  # Diagonal top-right to bottom-left and bottom-left to top-right
    ]
    rows, cols = len(grid), len(grid[0])

    def is_valid(x, y):
        return 0 <= x < rows and 0 <= y < cols

    def search_from(x, y, dx, dy):
        coords = []
        for i in range(len(word)):
            nx, ny = x + i * dx, y + i * dy
            if not is_valid(nx, ny) or grid[nx][ny] != word[i]:
                return []
            coords.append((nx, ny))
        return coords

    for x in range(rows):
        for y in range(cols):
            if grid[x][y] == word[0]:
                for dx, dy in directions:
                    coords = search_from(x, y, dx, dy)
                    if coords:
                        return coords
    return []

def find_word_in_grid_levenshtein(grid, word):
    """Find the closest matching word in the grid and return the coordinates of the word."""
    directions = [
        (0, 1), (0, -1),  # Horizontal
        (1, 0), (-1, 0),  # Vertical
        (1, 1), (-1, -1),  # Diagonal top-left to bottom-right and bottom-right to top-left
        (1, -1), (-1, 1)  # Diagonal top-right to bottom-left and bottom-left to top-right
    ]
    rows, cols = len(grid), len(grid[0])

    def is_valid(x, y):
        return 0 <= x < rows and 0 <= y < cols

    def search_from(x, y, dx, dy):
        coords = []
        found_word = ""
        for i in range(len(word)):
            nx, ny = x + i * dx, y + i * dy
            if not is_valid(nx, ny):
                return "", []
            found_word += grid[nx][ny]
            coords.append((nx, ny))
        return found_word, coords

    closest_match = None
    min_distance = float('inf')
    best_coords = []

    for x in range(rows):
        for y in range(cols):
            for dx, dy in directions:
                found_word, coords = search_from(x, y, dx, dy)
                if found_word:
                    distance = Levenshtein.distance(word, found_word)
                    print(f"{word} =? {found_word} (dist:{distance})")
                    if distance < min_distance:
                        min_distance = distance
                        closest_match = found_word
                        best_coords = coords

    print(f">> {word} => {closest_match} (dist:{min_distance})")

    return best_coords

def find_word_in_grid_fuzzy_wuzzy(grid, word):
    """Find the closest matching word in the grid and return the coordinates of the word."""
    directions = [
        (0, 1), (0, -1),  # Horizontal
        (1, 0), (-1, 0),  # Vertical
        (1, 1), (-1, -1),  # Diagonal top-left to bottom-right and bottom-right to top-left
        (1, -1), (-1, 1)  # Diagonal top-right to bottom-left and bottom-left to top-right
    ]
    rows, cols = len(grid), len(grid[0])

    def is_valid(x, y):
        return 0 <= x < rows and 0 <= y < cols

    def search_from(x, y, dx, dy):
        coords = []
        found_word = ""
        for i in range(len(word)):
            nx, ny = x + i * dx, y + i * dy
            if not is_valid(nx, ny):
                return "", []
            found_word += grid[nx][ny]
            coords.append((nx, ny))
        return found_word, coords

    closest_match = None
    highest_score = 0
    best_coords = []

    for x in range(rows):
        for y in range(cols):
            for dx, dy in directions:
                found_word, coords = search_from(x, y, dx, dy)
                if found_word:
                    score = fuzz.ratio(word, found_word)
                    #if word == "OROID" and score > 20:
                    #    print(f"{word} =? {found_word} (score:{score})")
                    if score > highest_score:
                        highest_score = score
                        closest_match = found_word
                        best_coords = coords

    print(f">> {word} => {closest_match} (score:{highest_score})")

    return best_coords

def solve_grid(grid, word_list):
    """Solve the word search puzzle and return the coordinates of the found words."""
    found_words = {}
    for word in word_list:
        coords = find_word_in_grid_fuzzy_wuzzy(grid, word)
        if coords:
            found_words[word] = coords
    return found_words

# Main function
def main():
    #for i in range(len(imgPaths)):
    #for i in range(1):
    #    img_path = imgPaths[i]
    #for img_path in [imgPaths[0], imgPaths[6], ]:
    #for img_path in [imgPaths[0], ]:
    for img_path in [imgPaths[6], ]:
        try:
            i = imgPaths.index(img_path)
            #output_file = img_output_base_path + 'text_regions_' + str(i + 1) + ".png"
            img, binary = preprocess_image(img_input_base_path + img_path)
            letters, words = detect_text_regions(binary)

            if not letters and not words:
                print(f"Nothing detected in {img_path}.")
                continueletters, words = detect_text_regions(binary)

            if not letters:
                print(f"No letters detected in {img_path}.")
                continue

            if not words:
                print(f"No words detected in {img_path}.")
                continue

            letters = get_largest_cluster(letters)
            letters = sorted(letters, key=lambda r: (r[1], r[0]))
            words = get_largest_cluster(words)

            print(f"word list of {len(words)}")

            save_row_regions(binary, letters, img_output_base_path + "rows/", None)

            # Extract letters and words
            letter_grid = extract_letters_from_grid_by_letter(binary, letters)

            #letter_grid = extract_words_from_regions(img, [(
            #    min(region[0] for region in letters),
            #    min(region[1] for region in letters),
            #    max(region[0] + region[2] for region in letters) - min(region[0] for region in letters),
            #    max(region[1] + region[3] for region in letters) - min(region[1] for region in letters)
            #)])
            word_list = extract_words_from_regions(binary, words)

            # Pretty print the grid for manual validation
            print(f"Letter Grid for {img_path}:")
            #pretty_print_grid(letter_grid)
            print(f"Word List for {img_path}: {word_list}")

            found_words = solve_grid(letter_grid, word_list)
            print("Found words", found_words.keys())
            #visualize_clusters(img, words, output_file)
            #cv2.imwrite(output_file, img)
            #if len(labels) == 0:
            #    print(f"No clusters detected in {img_path}.")
            #    continue

            # Visualize and save the results
            #output_file_name = f'puzzle_regions{i + 1}.png'
            #visualize_puzzle_regions(output_file_name, img, letters, words)
            output_file_name = f'solved_{i + 1}.png'
            visualize_solved_puzzle(output_file_name, img, letters, words, letter_grid, word_list, found_words)
            print(f"Processed {img_path} successfully.")
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            raise e

main()
