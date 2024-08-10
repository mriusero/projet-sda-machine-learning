from dashboard import app_layout, generate_pseudo_testing_data, generate_pseudo_testing_data_with_truth, load_data
import gc

def main():

    generate_pseudo_testing_data('data/input/training_data/pseudo_testing_data_with_truth',
                                 'data/input/training_data/degradation_data')

    generate_pseudo_testing_data_with_truth('data/input/training_data/pseudo_testing_data_with_truth',
                                            'data/input/training_data/pseudo_testing_data')

    update_message = load_data()
    print(update_message)

    app_layout()
    gc.collect()

    #os.system('clear')

if __name__ == '__main__':
    main()
