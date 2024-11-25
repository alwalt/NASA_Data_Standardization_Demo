from shiny import App, Inputs, Outputs, Session, render, ui
from image_tool import create_image_tool_app
from chatbot import create_chatbot_app
from shiny.types import ImgData
from pathlib import Path

# Import other tools as you add them
# from table_tool import create_table_tool_app

# Create individual tool UIs
image_ui, image_server = create_image_tool_app()
chatbot_ui, chatbot_server = create_chatbot_app()

# Main UI for the app using a navigation bar with a custom title and logo in the header
app_ui = ui.page_navbar(
    ui.nav_panel(
        "Image Annotation Tool",
        image_ui  # UI for the Image Tool
    ),
    ui.nav_panel(
        "Table Analysis Tool",
        ui.div("Table analysis content goes here")  # Placeholder for Table Tool UI
    ),
    ui.nav_panel(
        "Chatbot Assistant",
        chatbot_ui  # Placeholder for Chatbot Tool UI
    ),
    title="NASA Scientific Data Curation",  # Simple title to avoid layout issues
    # header=ui.tags.div(
    #     # Container for the logo and the title to ensure alignment
    #     ui.output_image("nasa_logo"),
    #     style="display: inline-flex; align-items: center; gap: 15px; padding: 10px;"  # Flex properties for alignment and spacing
    # )
)


# Main server function
def server(input, output, session):
    # Render the NASA logo image
    @render.image
    def nasa_logo():
        from pathlib import Path

        dir = Path(__file__).resolve().parent
        img: ImgData = {"src": str(dir / 'www' / "nasa-logo.svg"), "width": "100px"}
        print(img)
        return img
    
    # Use the image server logic
    image_server(input, output, session)

    # Use the chatbot server logic
    chatbot_server(input, output, session)

    # You can similarly add server logic for other tools once they are defined
    # Example:
    # table_ui, table_server = create_table_tool_app()

# Create the main app
app = App(app_ui, server)
