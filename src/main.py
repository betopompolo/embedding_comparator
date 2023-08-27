from typing import cast
from simple_term_menu import TerminalMenu

from create_mongo_db import CreateMongoDb

def main():
    menu = {
        "Create MongoDB": CreateMongoDb(pair_filters=[
            { "language": 'python', "partition": 'test' },
            { "language": 'python', "partition": 'train' },
            { "language": 'python', "partition": 'valid' },
        ]).run,
    }

    terminal_menu = TerminalMenu(title="Select an option", menu_entries=menu.keys())
    selected_option_index: int = cast(int, terminal_menu.show())
    assert isinstance(selected_option_index, int), "Invalid option"
    
    option_handler_key = list(menu.keys())[selected_option_index]
    menu[option_handler_key]()


if __name__ == "__main__":
    main()