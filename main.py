def choice():
	print("Please select one option")
	print("1. Logistic Regression")
	print("2. Multilayer Perceptron")
	print("3. Random Forest Classifier")
	print("4. SVM Classifier")
	print("5. Ensemble Classifier")

	option = int(input("\nSelect any of the above option (1..5): "))
	if option == 1:
		from logistic_implementation import logistic_regression
		logistic_regression()
	elif option == 2:
		from neural_mnist import neural_network
		neural_network()
	elif option == 3:
		from rsv_implementation import random_forest
		random_forest()
	elif option == 4:
		from svm_implementation import svm_implement
		svm_implement()
	elif option == 5:
		import ensemble
	return option
	
i = 0
while(i<6):
	i = choice()