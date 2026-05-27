import os
import logging
from telegram import Update
from telegram.ext import Application, MessageHandler, filters, ContextTypes

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID = int(os.getenv("CHAT_ID", "0"))
OWNER_ID = int(os.getenv("OWNER_ID", "0"))

if not TOKEN or not CHAT_ID or not OWNER_ID:
    raise ValueError("BOT_TOKEN, CHAT_ID и OWNER_ID должны быть заданы")

TRIGGER_WORD = "ошибка"

def get_user_info(update: Update) -> str:
    user = update.effective_user
    username = f"@{user.username}" if user.username else "без username"
    return f"{user.full_name} ({username}, id: {user.id})"

def get_chat_info(update: Update) -> str:
    chat = update.effective_chat
    return chat.title if chat.title else f"чат {chat.id}"

async def monitor_messages(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_chat.id != CHAT_ID:
        return

    text = ""
    if update.message.text:
        text = update.message.text
    elif update.message.caption:
        text = update.message.caption
    else:
        return

    if TRIGGER_WORD in text.lower():
        author = get_user_info(update)
        chat = get_chat_info(update)
        message_text = text[:1000]
        notification = (
            f"🚨 Обнаружено слово «ошибка»\n"
            f"Чат: {chat}\n"
            f"Автор: {author}\n"
            f"Сообщение:\n{message_text}"
        )

        try:
            await context.bot.send_message(
                chat_id=OWNER_ID,
                text=notification,
                disable_web_page_preview=True
            )
            logger.info(f"Уведомление отправлено владельцу: {update.message.message_id}")
        except Exception as e:
            logger.error(f"Ошибка отправки уведомления: {e}")

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_chat.type == "private":
        await update.message.reply_text(
            f"Ваш Telegram ID: {update.effective_user.id}\n"
            "Бот отслеживает сообщения со словом «ошибка» в указанном чате."
        )

def main():
    application = Application.builder().token(TOKEN).build()
    application.add_handler(MessageHandler(
        (filters.TEXT | filters.CAPTION) & ~filters.COMMAND,
        monitor_messages
    ))
    application.add_handler(MessageHandler(filters.COMMAND & filters.Regex("^/start$"), start))
    logger.info("Бот запущен...")
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()
