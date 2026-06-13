	# SparkPrediction

	train.csv data format:
	id: unique numeric identifier for each row
	user_id: ID of the user who posted the review (u_xxxx, where xxxx is a 16-digit combination of letters and numbers)
	prod_id: ID of the reviewed product (a_yyyy, where yyyy is a 16-digit combination of letters and numbers)
	parent_prod_id: Parent ID of the product (category of the product) (a_yyyy, where yyyy is a 16-digit combination of letters and numbers)
	title: the review title
	comment: the review text
	time: timestamp when the review was posted
	votes: number of users who found the review helpful (non-negative integer)
	purchased: whether the user purchased the product (TRUE or FALSE)
	rating: the rating given by the user (ground truth label for training) (integer from 1 to 5) 

