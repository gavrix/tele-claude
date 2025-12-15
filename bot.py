from telegram import Update
from telegram.ext import Application, MessageHandler, CommandHandler, CallbackQueryHandler, filters

from config import BOT_TOKEN
from handlers import handle_new_topic, handle_callback, handle_message, handle_topic_created


def main() -> None:
    if not BOT_TOKEN:
        raise ValueError("BOT_TOKEN not found in environment")

    app = Application.builder().token(BOT_TOKEN).build()

    # Handle new forum topic created (auto-detect)
    app.add_handler(MessageHandler(
        filters.StatusUpdate.FORUM_TOPIC_CREATED & filters.ChatType.SUPERGROUP,
        handle_topic_created
    ))

    # Handle /new command in groups (manual fallback)
    app.add_handler(CommandHandler(
        "new",
        handle_new_topic,
        filters=filters.ChatType.SUPERGROUP
    ))

    # Handle inline keyboard button clicks
    app.add_handler(CallbackQueryHandler(handle_callback))

    # Handle all text messages in groups
    app.add_handler(MessageHandler(
        filters.TEXT & filters.ChatType.SUPERGROUP & ~filters.COMMAND,
        handle_message
    ))

    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
