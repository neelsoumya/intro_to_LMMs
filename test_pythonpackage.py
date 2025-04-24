# pip install -e .

from intro_to_LMMs import __version__, greet

if __name__ == "__main__":
    print(f"Welcome to my first python package! The intro_to_LMMs package, version: {__version__}")
    
    # Call the greet function
    print("Calling the greet function:")

    print(greet())

