from create_model import create, train

def main():
	model = create()
	train(model, train_list = [2], lr = 0.0005, epochs = 20, patience = 20, clipnorm = 5, clipvalue = 5, checkpoint = 5)

if __name__=="__main__":
	main()
