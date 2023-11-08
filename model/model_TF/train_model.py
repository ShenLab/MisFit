from create_model import create, train

def main():
	model = create()
	train(model, train_list = [0], lr = 1e-4, epochs = 30, patience = 30, clipnorm = None, clipvalue = None, checkpoint = 5)

if __name__=="__main__":
	main()
