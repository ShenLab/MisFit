from create_model import create, train

def main():
	model = create()
	train(model, train_list = [2], lr = 1e-3, epochs = 25, patience = 25, clipnorm = 3, clipvalue = 3, checkpoint = 0)

if __name__=="__main__":
	main()
