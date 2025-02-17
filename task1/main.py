import os
import sys

# Make sure the current directory (ml_project) is added to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import train
import process
import predect

def main():
    # Run preprocessing
    print("Preprocessing data...")
    process.run()

    # Train the model
    print("Training the model...")
    train.run()

    # Make predictions
    print("Making predictions...")
    predect.run()

if __name__ == "__main__":
    main()
