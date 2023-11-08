from create_model import combine
from create_post import output_post

def main():
	test_list = 0
	output_post(test_list, output_d = True)
	combine(test_list, "d", "model_damage")
	combine(test_list, "s", "model_selection")

if __name__=="__main__":
	main()
