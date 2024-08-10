from dashboard import app_layout, generate_pseudo_testing_data, generate_pseudo_testing_data_with_truth, load_data, merge_data
import os
import gc
from datetime import datetime
def print_with_timestamp(message):
    """Prints a message with a timestamp."""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{timestamp}] {message}")

def main():

    generate_pseudo_testing_data('data/input/training_data/pseudo_testing_data_with_truth',
                                 'data/input/training_data/degradation_data')

    generate_pseudo_testing_data_with_truth('data/input/training_data/pseudo_testing_data_with_truth',
                                            'data/input/training_data/pseudo_testing_data')

    load_data()
    print_with_timestamp("New random data generated: 'pseudo_testing_data' & 'pseudo_testing_data_with_truth' ")

    app_layout()
    gc.collect()

    #os.system('clear')

if __name__ == '__main__':
    main()
