from create_post import create_post, train_post

def main():
	model = create_post()
	train_post(model, [2], epochs = 30, patience = 5)

if __name__=="__main__":
	main()

