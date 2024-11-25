from shiny import ui

chatbot_ui = ui.page_fillable(
    ui.panel_title("Hello Shiny Chat"),
    ui.chat_ui("chat"),
    fillable_mobile=True,
)

# Create a welcome message
welcome = ui.markdown(
    """
    Hi! This is a simple Shiny `Chat` UI. Enter a message below and I will
    simply repeat it back to you. For more examples, see this
    [folder of examples](https://github.com/posit-dev/py-shiny/tree/main/examples/chat).
    """
)


def chatbot_server(input, output, session):
    chat = ui.Chat(id="chat", messages=[welcome])

    # Define a callback to run when the user submits a message
    @chat.on_user_submit
    async def _():
        # Get the user's input
        user = chat.user_input()
        # Append a response to the chat
        await chat.append_message(f"You said: {user}")

# Create a function that returns the UI and server parts for this tool
def create_chatbot_app():
    return chatbot_ui, chatbot_server