from create_model import create, train

def main():
	model = create()
	train(model, train_list = [0], lr = 1e-2, epochs = 25, patience = 25, clipnorm = None, clipvalue = None, checkpoint = 1)

if __name__=="__main__":
	main()
