from create_model import output, combine

def main():
	test_list = 0
	output(test_list)
	combine(test_list, "d", "model_damage")
	combine(test_list, "s", "model_selection")

if __name__=="__main__":
	main()
