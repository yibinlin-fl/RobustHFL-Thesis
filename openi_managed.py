"""Root-level fallback for OpenI screens that only index root startup files."""
from scripts.openi_entry import arguments, run


if __name__ == "__main__":
    run(arguments(None))
