from dashboard import app_layout, generate_pseudo_testing_data, generate_pseudo_testing_data_with_truth, load_data
import gc
import os
def main():

    generate_pseudo_testing_data('data/input/training_data/pseudo_testing_data_with_truth',
                                 'data/input/training_data/degradation_data')

    generate_pseudo_testing_data_with_truth('data/input/training_data/pseudo_testing_data_with_truth',
                                            'data/input/training_data/pseudo_testing_data')


    app_layout()


    #os.system('clear')

if __name__ == '__main__':
    main()
