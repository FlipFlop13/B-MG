DATA/
The data folder is where all  the data iss and where the scripts will add data to. 
	CSV/
		This folder contains all the raw data, these are directly from the Worlld Bank or from the World Healt 
		Organization. These are only used to create the knowledge graphs.
	GRAPHS/
		Conatins the knowledge graphs.
	SIMILARITY_MEASURMENT/
		Contains the resulting files from the similarity measures. It contains a file with the indicator ppairs with t
		with the largest correlations. Furthermore it contains the matrices representing the country similarities. 
		the file containing the dense matrix represents the similarity estimates and the file containinng the 
		sparse matrix contains the true similarity for the top 5 similar countries. Finally, the correlation predicition
		has the correlaton between prediciton and true series values.
csv_to_kg.py Loads all the csv data and transforms it into a knowledge graph. (Takes about 5 minutes on my pc)
find_correlations_between_categories.py gets the knowledge graphs and looks for correlated series then it measures th similarity
	between the countries. (Takes about 2 hours on 5 cores (more has been causing memory issues))
predict.py Attempts to run the prediction method and saves the results ass a text file as seen in data/similarity_meassurments/
s (takes ~1 hour)


	
