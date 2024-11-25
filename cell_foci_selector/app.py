from shiny import App, Inputs, Outputs, Session, render, ui, reactive
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
    sidebar=ui.sidebar(
            ui.output_ui("chat_ui"),  # Dynamically render the chat UI in the sidebar
            width=300,
            position="right",
            style="height: 100%; background-color: #f8f9fa; border-left: 1px solid #ddd;",
        ),
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

    # Reactive value to store chat messages
    chat_messages = reactive.Value([
        {"content": "Hello! How can I assist you today?", "role": "assistant"}
    ])

    # Dynamically render the chat UI in the sidebar
    @output
    @render.ui
    def chat_ui():
        return ui.div(
            ui.Chat(
                id="chat",
                messages=chat_messages(),
            ).ui(height="100%"),
            # style="height: 100%; position: fixed; right: 0; background-color: #f8f9fa; border-left: 1px solid #ddd;",
        )

    # Handle chat submission
    @ui.Chat(id="chat").on_user_submit
    async def handle_chat_submit():
        user_message = input.chat()
        if user_message:
            messages = chat_messages()
            messages.append({"role": "user", "content": user_message})
            messages.append({"role": "assistant", "content": f"You said: {user_message}"})
            chat_messages.set(messages)

    # You can similarly add server logic for other tools once they are defined
    # Example:
    # table_ui, table_server = create_table_tool_app()

# Create the main app
app = App(app_ui, server)
