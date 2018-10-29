For online perceptron training and validation, run:
$ python perceptron.py [online/average] [training file] [validation file] [iteration number] [y/n for predicting a test file] [test file] [best iteration number]

For example:
Running online perceptron (with predicting a test file)
$ python perceptron.py o pa2_train.csv pa2_valid.csv 15 y pa2_test_no_label.csv 7

Running average perceptron (without predicting a test file)
$ python perceptron.py a pa2_train.csv pa2_valid.csv 15 n

For kernel perceptron training and validation, run:
$ python perceptron.py [kernel] [training file] [validation file] [iteration number] [p value] [y/n for predicting a test file] [test file] [best iteration number]

For example:
Running kernel perceptron (with predicting a test file)
$ python perceptron.py k pa2_train.csv pa2_valid.csv 15 3 y pa2_test_no_label.csv 4

Running kernel perceptron (without predicting a test file)
$ python perceptron.py k pa2_train.csv pa2_valid.csv 15 3 n

This code will output an accuracy file (accuracy for each iteration), and a prediction file if you put y in test option.