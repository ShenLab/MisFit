from create_post import create_post, train_post

def main():
	model = create_post()
	train_post(model, [2], epochs = 20, patience = 10)

if __name__=="__main__":
	main()

