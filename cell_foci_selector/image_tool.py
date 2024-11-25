# image_path = "static/P242_73665006707-A6_001_004_proj.tif"
from shiny import App, ui, reactive, render
import matplotlib.pyplot as plt
import numpy as np
import cv2
import json
import os
import glob
import shutil

# Load all .tif images in the folder
image_folder = "static"
# Reactive image list to store the list of images
image_list = reactive.Value([])  # Track list of images reactively
current_image_index = reactive.Value(0)  # Track current image index

# Function to refresh the list of images
def refresh_image_list():
    updated_list = sorted(glob.glob(os.path.join(image_folder, "*")))
    image_list.set(updated_list)  # Update the reactive value

    if len(updated_list) == 0:
        print("Image list is empty. Please upload images.")
    else:
        print(f"Refreshed image list: {updated_list}")


# Function to load the current image path safely
def get_current_image_path():
    images = image_list()
    if len(images) == 0:
        return None  # Return None if no images are available
    return images[current_image_index()]

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

# Function to load the current image safely
def load_current_image():
    image_path = get_current_image_path()
    if image_path is None:
        print("No image path available.")
        return None
    return load_image(image_path)

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


# Image UI
image_ui = ui.div(
    ui.div(
        ui.card(
            ui.h2("Imaging Tools", style="text-align: center; margin-bottom: 10px;"),
            ui.input_file(
                "upload_image",
                label="Upload Image(s)",
                multiple=True,  # Allow multiple files
                accept=[".tif", ".png", ".jpg"],  # Accepted image formats
            ),
            # ui.input_action_button("refresh_images", "Refresh Image List", class_="btn-info"),
            ui.tags.hr(style="margin-top: 10px; margin-bottom: 10px;"),
            ui.h3("Foci Selection", style="margin-top: 5px; margin-bottom: 5px;"),
            ui.input_radio_buttons(
                "selection_mode", "Modes",
                choices={"click": "Point", "brush": "Area"},
                selected="click"
            ),
            ui.input_checkbox("enable_local_max", "Enable Find Local Max", value=False),
            ui.tags.hr(style="margin-top: 10px; margin-bottom: 10px;"),
            ui.h3("Image Info", style="margin-top: 5px; margin-bottom: 5px;"),
            ui.output_text_verbatim("image_info"),
            style="padding: 20px;  width: 340px;"
        ),
        style="flex: 0 0 auto; padding-right: 10px;"
    ),
    ui.div(
        ui.output_ui("dynamic_card"),  # Render the dynamic card (main content)
        ui.div(
            ui.input_action_button("reset", "Reset Selections", class_="btn-primary"),
            ui.input_action_button("undo", "Undo Last Selection", class_="btn-secondary"),
            ui.input_action_button("report", "Report as Corrupted", class_="btn-danger"),
            ui.input_action_button("submit", "Submit", class_="btn-success"),
            style="display: flex; gap: 10px; margin-top: 15px;"
        ),
        style="flex: 2;"
    ),
    style="display: flex; gap: 10px; align-items: flex-start; justify-content: space-between; padding: 20px;"
)

# Shiny Server
def image_server(input, output, session):
    # Store selected points
    selected_points = reactive.Value([])
    corrupted = reactive.Value(False)
    current_image_data = reactive.Value(None)  # Current image data

    # # Refresh the image list on app initialization
    # refresh_image_list()  # This updates the image_list reactive value

    # Refresh the image list on app initialization using a reactive context
    @reactive.Effect
    def initialize_image_list():
        refresh_image_list()  # This updates the reactive value image_list
        if len(image_list()) > 0:
            current_image_index.set(0)

    # Load the current image data when the image index or image list changes
    @reactive.Effect
    def update_image_data():
        if current_image_index() >= len(image_list()):
            print("No more images to display.")
            return

        images = image_list()
        if len(images) == 0:
            current_image_data.set(None)
            print("No images available. Please upload images.")
        else:
            current_image_data.set(load_current_image())
            print(f"Image data updated for image: {get_current_image_path()}")



    # Dynamically render the card
    @output
    @render.ui
    def dynamic_card():
        # Make dynamic_card depend on both current_image_index and image_list
        _ = current_image_index()  # Track reactive value of current_image_index
        images = image_list()  # Track reactive value of image_list
        if len(images) == 0:
            # Show a placeholder card when no images are available
            return ui.card(
                ui.card_header("Upload Images"),
                ui.div("Please upload images to start annotating."),
                style="width: 820px; height: 700px; max-width: 820px; margin-left: 0;"
            )
        
        current_image_path = get_current_image_path()
        if current_image_path is None:
            # If no valid image is available, return placeholder
            return ui.card(
                ui.card_header("Upload Images"),
                ui.div("Please upload images to start annotating."),
                style="width: 820px; height: 700px; max-width: 820px; margin-left: 0;"
            )

        # Render the card with the image information
        current_image_name = os.path.basename(current_image_path)
        return ui.card(
            ui.card_header(f"Image: {current_image_name}"),
            ui.output_plot(
                "image_plot",
                click=True,
                brush=True,
                width="800px",
                height="800px",
            ),
            style="width: 820px; height: 700px; max-width: 820px; margin-left: 0;"
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

    # Handle click events
    @reactive.Effect
    @reactive.event(input.image_plot_click)
    def record_click():
        if input.selection_mode() != "click":
            return
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

    @reactive.Effect
    @reactive.event(input.image_plot_brush)
    def record_brush():
        if input.selection_mode() != "brush":
            return
        
        brush = input.image_plot_brush()
        image_data = current_image_data()
        
        if brush is not None and image_data is not None:
            # Get brushed region bounds
            x_min, x_max = int(brush["xmin"]), int(brush["xmax"])
            y_min, y_max = int(brush["ymin"]), int(brush["ymax"])
            
            # Clip the coordinates to stay within image bounds
            height, width = image_data.shape
            x_min = max(0, min(x_min, width - 1))
            x_max = max(0, min(x_max, width - 1))
            y_min = max(0, min(y_min, height - 1))
            y_max = max(0, min(y_max, height - 1))
            
            # Extract the brushed region
            brushed_region = image_data[y_min:y_max + 1, x_min:x_max + 1]
            
            if input.enable_local_max():
                # Find the brightest point in the brushed region
                local_max_idx = np.unravel_index(np.argmax(brushed_region), brushed_region.shape)
                adjusted_x = x_min + local_max_idx[1]  # Add offset to global coordinates
                adjusted_y = y_min + local_max_idx[0]
                print(f"Brightest point in brush: ({adjusted_x}, {adjusted_y})")
                selected_points.set(selected_points() + [(adjusted_x, adjusted_y)])
            else:
                # Default: Use the center of the brushed region
                center_x = (x_min + x_max) // 2
                center_y = (y_min + y_max) // 2
                print(f"Center of brush: ({center_x}, {center_y})")
                selected_points.set(selected_points() + [(center_x, center_y)])


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
        if current_image_index() + 1 < len(image_list()):
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

    
    # Handle image uploads
    @reactive.Effect
    @reactive.event(input.upload_image)
    def handle_upload():
        uploaded_files = input.upload_image()
        if uploaded_files:
            # Ensure the static directory exists
            if not os.path.exists(image_folder):
                os.makedirs(image_folder)
                print(f"Created directory: {image_folder}")

            for file_info in uploaded_files:
                file_path = file_info["datapath"]  # Temporary file path
                dest_path = os.path.join(image_folder, file_info["name"])  # Destination path
                shutil.move(file_path, dest_path)  # Save file to the static directory

            # Refresh the image list after uploading files
            refresh_image_list()

            # Set the current image index to the first image after the list is refreshed
            if len(image_list()) > 0:
                current_image_index.set(0)  # Ensure it starts from the first image after refresh

            # Print debugging information
            print(f"Uploaded {len(uploaded_files)} files.")
            print(f"Current image list: {image_list()}")


    # Handle refresh button
    @reactive.Effect
    @reactive.event(input.refresh_images)
    def handle_refresh():
        refresh_image_list()
        if len(image_list()) > 0:
            current_image_index.set(0)  # Ensure it starts from the first image after refresh
        print("Image list refreshed.")

    # Refresh the image list on app initialization
    refresh_image_list()


# Create a function that returns the UI and server parts for this tool
def create_image_tool_app():
    return image_ui, image_server