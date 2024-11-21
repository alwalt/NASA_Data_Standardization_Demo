# image_path = "static/P242_73665006707-A6_001_004_proj.tif"
from shiny import App, ui, reactive, render
import matplotlib.pyplot as plt
import numpy as np
import cv2
import json
import os
import glob
import pandas as pd

# Load the metadata from the CSV file
meta_file = "metadata/meta.csv"
metadata_df = pd.read_csv(meta_file)

# Function to get metadata for the current image
def get_image_metadata(image_name):
    # Extract metadata for the given image name
    row = metadata_df[metadata_df["filename"] == image_name]
    if not row.empty:
        return {
            "Dose (Gy)": row.iloc[0]["dose_Gy"],
            "Particle Type": row.iloc[0]["particle_type"],
            "Hours Post Exposure": row.iloc[0]["hr_post_exposure"],
        }
    else:
        return {
            "Dose (Gy)": "Unknown",
            "Particle Type": "Unknown",
            "Hours Post Exposure": "Unknown",
        }

# Load all .tif images in the folder
image_folder = "static"
image_list = sorted(glob.glob(os.path.join(image_folder, "*.tif")))  # Sorted list of images
current_image_index = reactive.Value(0)  # Track current image index

# Load the current image path and data
def get_current_image_path():
    return image_list[current_image_index()]

# Function to load and preprocess the image
def load_image(file_path, target_size=None):
    # Load the image using OpenCV
    image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)

    # Clip pixel values to a specified range (400, 4000)
    image = np.clip(image, 400, 4000)

    # Normalize to [0, 1]
    image = (image - 400) / (4000 - 400)

    # Optionally resize the image for display
    if target_size:
        image = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
    
    # Ensure the image is a float32 numpy array
    return image.astype(np.float32)

def load_current_image():
    return load_image(get_current_image_path())

# Save results as JSON
def save_results(image_name, selected_points, corrupted, output_dir="results"):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    # Define the output file path
    output_path = os.path.join(output_dir, f"{os.path.splitext(image_name)[0]}.json")
    # Create the data dictionary
    data = {
        "image_name": image_name,
        "selected_points": [[int(x), int(y)] for x, y in selected_points],  # Convert coordinates to Python int
        "corrupted": bool(corrupted),  # Ensure boolean value is properly serialized
    }
    # Save as JSON
    with open(output_path, "w") as f:
        json.dump(data, f, indent=4)
    print(f"Results saved to {output_path}")

# Function to get a local window around a point
def get_window(image, x, y, window_size):
    half_window = window_size // 2
    return image[
        max(0, y - half_window): min(image.shape[0], y + half_window + 1),
        max(0, x - half_window): min(image.shape[1], x + half_window + 1)
    ]

# Function to find the local maximum in the window
def find_local_max(image, x, y, window_size):
    window = get_window(image, x, y, window_size)
    local_max = np.unravel_index(np.argmax(window), window.shape)
    global_y = y - (window.shape[0] // 2) + local_max[0]
    global_x = x - (window.shape[1] // 2) + local_max[1]
    return global_x, global_y


# Shiny UI
app_ui = ui.page_sidebar(
    ui.sidebar(
        ui.h3("Image Info"),
        ui.output_text("image_info"),

        ui.h3("Options"),  # New header for options
        ui.input_checkbox("enable_local_max", "Enable Find Local Max", value=False),

    ),
    ui.div(
        ui.output_ui("dynamic_card"),  # Render the card dynamically
        ui.div(
            ui.input_action_button("reset", "Reset Selections", class_="btn-primary"),
            ui.input_action_button("undo", "Undo Last Selection", class_="btn-secondary"),
            ui.input_action_button("report", "Report as Corrupted", class_="btn-danger"),
            ui.input_action_button("submit", "Submit", class_="btn-success"),
            style="display: flex; gap: 10px; margin-top: 15px;"  # Buttons below the card
        ),
    )
)


# Shiny Server
def server(input, output, session):
    # Store selected points
    selected_points = reactive.Value([])
    corrupted = reactive.Value(False)

    # Reactive value to store the current image data
    current_image_data = reactive.Value(None)

    # Load the current image data when the image index changes
    @reactive.Effect
    def update_image_data():
        current_image_data.set(load_current_image())


    # Dynamically render the card
    @output
    @render.ui
    def dynamic_card():
        current_image_name = os.path.basename(get_current_image_path())
        metadata = get_image_metadata(current_image_name)
        return ui.card(
            ui.card_header(f"Image: {current_image_name}"),  # Dynamically set header
            ui.output_plot(
                "image_plot",
                click=True,
                width="800px",
                height="800px"
            ),
            ui.card_footer(
                f"Type: {metadata['Particle Type']} | "
                f"Dose: {metadata['Dose (Gy)']} Gy | "
                f"Hours Post Exposure: {metadata['Hours Post Exposure']}"
            ),
            width="800px",
            height="800px",
        )

    # Render the plot
    @output
    @render.plot
    def image_plot():
        fig, ax = plt.subplots()
        image_data = current_image_data()  # Load the current image
        ax.imshow(image_data, cmap="gray")
        for x, y in selected_points():
            ax.plot(x, y, "ro")  # Plot selected points in red
        # ax.set_title(f"Image: {os.path.basename(get_current_image_path())}")
        ax.axis("off")
        return fig
    
    @output
    @render.text
    def image_info():
        # Get the current image file name
        current_image_name = os.path.basename(get_current_image_path())
        
        # Get metadata for the current image
        current_metadata = get_image_metadata(current_image_name)
        
        # Manually format each piece of metadata on a new line
        info = (
            f"File Name: {current_image_name}\n"
            f"Dose (Gy): {current_metadata['Dose (Gy)']}\n"
            f"Particle Type: {current_metadata['Particle Type']}\n"
            f"Hours Post Exposure: {current_metadata['Hours Post Exposure']}"
        )
        
        return info


    # Handle click events
    @reactive.Effect
    @reactive.event(input.image_plot_click)
    def record_click():
        image_data = current_image_data()
        if image_data is None:
            return
        click = input.image_plot_click()
        if click is not None:
            clicked_x, clicked_y = int(click["x"]), int(click["y"])
            height, width = image_data.shape[:2]

            # Check if click is within bounds
            if 0 <= clicked_x < width and 0 <= clicked_y < height:
                print(f"Clicked coordinates: ({clicked_x}, {clicked_y})")

                # Apply local max adjustment if enabled
                if input.enable_local_max():
                    adjusted_x, adjusted_y = find_local_max(image_data, clicked_x, clicked_y, window_size=7)
                    if 0 <= adjusted_x < width and 0 <= adjusted_y < height:
                        print(f"Adjusted coordinates: ({adjusted_x}, {adjusted_y})")
                        selected_points.set(selected_points() + [(adjusted_x, adjusted_y)])
                    else:
                        print(f"Adjusted coordinates ({adjusted_x}, {adjusted_y}) are out of bounds. Ignoring.")
                else:
                    selected_points.set(selected_points() + [(clicked_x, clicked_y)])
            else:
                print(f"Click ({clicked_x}, {clicked_y}) is outside image bounds. Ignoring.")

    # Handle reset button
    @reactive.Effect
    @reactive.event(input.reset)
    def reset_points():
        selected_points.set([])

    # Handle undo button
    @reactive.Effect
    @reactive.event(input.undo)
    def undo_last_point():
        if selected_points():
            print("Undoing last selection")  # Debugging
            selected_points.set(selected_points()[:-1])

    # Handle report as corrupted button
    @reactive.Effect
    @reactive.event(input.report)
    def report_corrupted():
        print("Reporting image as corrupted...")  # Debugging
        save_results(
            os.path.basename(get_current_image_path()),  # Save current image
            [],
            True
        )
        move_to_next_image()  # Move to next image

    # Handle submit button
    @reactive.Effect
    @reactive.event(input.submit)
    def submit_data():
        save_results(
            os.path.basename(get_current_image_path()),
            selected_points(),
            False
        )
        move_to_next_image()

    def move_to_next_image():
        if current_image_index() + 1 < len(image_list):
            # Move to the next image
            current_image_index.set(current_image_index() + 1)
        else:
            # Loop back to the first image
            print("All images processed! Looping back to the first image.")
            current_image_index.set(0)
        # Reset state for the new image
        selected_points.set([])
        corrupted.set(False)
        print(f"Loading image: {get_current_image_path()}")


    # Display selected points
    @output
    @render.text
    def selected_points_output():
        return f"Selected Points: {selected_points()}"

# Create the app
app = App(app_ui, server)
