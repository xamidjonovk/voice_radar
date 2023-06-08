from telegram import Update, ReplyKeyboardMarkup, ReplyKeyboardRemove
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, ConversationHandler
import sqlite3

# Define conversation states
STATE_MAIN_MENU = 0
STATE_ADD_PERSON = 1
STATE_UPDATE_PERSON = 2
STATE_DELETE_PERSON = 3

# Database file path
DB_FILE = 'people.db'


def get_connection():
    """Get the SQLite connection"""
    return sqlite3.connect(DB_FILE)


def create_table():
    """Create the people table if it doesn't exist"""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS people (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    first_name TEXT,
                    last_name TEXT,
                    folder TEXT,
                    about TEXT,
                    photo_path TEXT
                    )''')
    conn.commit()
    conn.close()


def start(update: Update, context):
    """Handler for the /start command"""
    reply_keyboard = [['Add Person', 'Update Person'], ['Delete Person', 'Get People']]
    markup = ReplyKeyboardMarkup(reply_keyboard, one_time_keyboard=True)

    update.message.reply_text('Welcome to the CRUD bot!\n\nPlease select an operation:', reply_markup=markup)
    return STATE_MAIN_MENU


def main_menu(update: Update, context):
    """Handler for the main menu selection"""
    selected_option = update.message.text

    if selected_option == 'Add Person':
        update.message.reply_text('Enter the First Name:')
        return STATE_ADD_PERSON
    elif selected_option == 'Update Person':
        update.message.reply_text('Enter the ID of the person to update:')
        return STATE_UPDATE_PERSON
    elif selected_option == 'Delete Person':
        update.message.reply_text('Enter the ID of the person to delete:')
        return STATE_DELETE_PERSON
    elif selected_option == 'Get People':
        return get_people(update, context)


def add_person(update: Update, context):
    """Handler for adding a person"""
    first_name = update.message.text

    update.message.reply_text('Enter the Last Name:')
    context.user_data['first_name'] = first_name
    return STATE_ADD_PERSON


def add_person_last_name(update: Update, context):
    """Handler for adding the last name of a person"""
    last_name = update.message.text

    # Get the first name from the context
    first_name = context.user_data.get('first_name')

    # Save the person to the database
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("INSERT INTO people (first_name, last_name) VALUES (?, ?)", (first_name, last_name))
    conn.commit()
    conn.close()

    update.message.reply_text('Person added successfully!')
    return start(update, context)


def update_person(update: Update, context):
    """Handler for updating a person"""
    person_id = update.message.text

    # Check if the person ID exists in the database
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM people WHERE id=?", (person_id,))
    person = cursor.fetchone()
    conn.close()

    if person:
        update.message.reply_text('Select the field to update: First Name, Last Name')
        context.user_data['person_id'] = person_id
        return STATE_UPDATE_PERSON
    else:
        update.message.reply_text('Invalid person ID. Please try again.')
        return start(update, context)


def perform_update(update: Update, context):
    """Handler for selecting the field to update"""
    field = update.message.text

    if field == 'First Name':
        update.message.reply_text('Enter the new First Name:')
        return STATE_UPDATE_PERSON
    elif field == 'Last Name':
        update.message.reply_text('Enter the new Last Name:')
        return STATE_UPDATE_PERSON


def perform_update_field(update: Update, context):
    """Handler for updating the selected field"""
    new_value = update.message.text
    field = context.user_data['field']
    person_id = context.user_data['person_id']

    # Update the selected field in the database
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(f"UPDATE people SET {field}=? WHERE id=?", (new_value, person_id))
    conn.commit()
    conn.close()

    update.message.reply_text(f'{field} updated successfully!')
    return start(update, context)


def delete_person(update: Update, context):
    """Handler for deleting a person"""
    person_id = update.message.text

    # Check if the person ID exists in the database
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM people WHERE id=?", (person_id,))
    person = cursor.fetchone()
    conn.close()

    if person:
        # Delete the person from the database
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM people WHERE id=?", (person_id,))
        conn.commit()
        conn.close()

        update.message.reply_text('Person deleted successfully!')
        return start(update, context)
    else:
        update.message.reply_text('Invalid person ID. Please try again.')
        return start(update, context)


def get_people(update: Update, context):
    """Handler for getting all people"""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM people")
    people = cursor.fetchall()
    conn.close()

    if people:
        people_list = '\n'.join([f'{person[0]}: {person[1]} {person[2]}' for person in people])
        update.message.reply_text(f'People:\n{people_list}')
    else:
        update.message.reply_text('No people found.')

    return start(update, context)


def main():
    # Create the people table if it doesn't exist
    create_table()

    # Create the Telegram bot updater and dispatcher
    updater = Updater("6282188294:AAEohy5_HdMfG4Ygoh3vufotvkky0Uy-B-0", use_context=True)
    dp = updater.dispatcher

    # Add conversation handler
    conv_handler = ConversationHandler(
        entry_points=[CommandHandler('start', start)],
        states={
            STATE_MAIN_MENU: [MessageHandler(Filters.text, main_menu)],
            STATE_ADD_PERSON: [MessageHandler(Filters.text, add_person_last_name)],
            STATE_UPDATE_PERSON: [MessageHandler(Filters.text, perform_update)],
            STATE_DELETE_PERSON: [MessageHandler(Filters.text, delete_person)],
        },
        fallbacks=[CommandHandler('start', start)],
    )
    dp.add_handler(conv_handler)

    # Start the bot
    updater.start_polling()
    updater.idle()


if __name__ == '__main__':
    main()
